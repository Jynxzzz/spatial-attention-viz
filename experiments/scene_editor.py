"""
Scene editor for counterfactual experiments on Waymo pkl scene files.

Provides a fluent API for loading, inspecting, modifying, and saving scene
dictionaries used by Scenario Dreamer. Designed for A/B counterfactual
experiments where we compare model attention between an original scene and
a systematically modified variant.

Scene structure (Waymo pkl via Scenario Dreamer):
    scene['objects']         -- list of agent dicts (position, heading, velocity, valid, type, width, length)
    scene['av_idx']          -- int index of the ego vehicle
    scene['lane_graph']      -- dict with lanes, suc_pairs, pre_pairs, left_pairs, right_pairs, etc.
    scene['traffic_lights']  -- list of 91 frames, each a list of light dicts

All trajectories have 91 frames at 10 Hz (9.1 seconds total).
Traffic light states: 0 = Red/Unknown, 4 = Green.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_FRAMES = 91
FRAME_RATE_HZ = 10

# Default agent dimensions (width, length) in meters
DEFAULT_DIMENSIONS: dict[str, tuple[float, float]] = {
    'vehicle': (2.0, 4.5),
    'pedestrian': (0.6, 0.6),
    'cyclist': (0.8, 1.8),
}

# Traffic light state codes (Waymo convention)
TL_RED = 0
TL_GREEN = 4

VALID_AGENT_TYPES = {'vehicle', 'pedestrian', 'cyclist', 'other'}


# ---------------------------------------------------------------------------
# SceneEditor
# ---------------------------------------------------------------------------

class SceneEditor:
    """Scene editing for counterfactual attention experiments.

    Provides methods to inspect and modify Waymo scene pkl files used by
    Scenario Dreamer.  All mutation methods return ``self`` so calls can be
    chained:

        >>> editor = SceneEditor(path).remove_agents_by_type('pedestrian').flip_traffic_lights()

    The editor always works on a deep copy of the original data so the
    source dict / file is never mutated.
    """

    # ------------------------------------------------------------------ init
    def __init__(self, scene_path_or_dict: Union[str, Path, dict]) -> None:
        """Load a scene from a pkl path or accept a dict directly.

        Args:
            scene_path_or_dict: Either a filesystem path to a ``.pkl`` file
                containing a serialised scene dict, or a scene dict itself.

        Raises:
            FileNotFoundError: If a path is given and the file does not exist.
            ValueError: If the loaded object is not a dict with the expected
                top-level keys.
        """
        if isinstance(scene_path_or_dict, (str, Path)):
            path = Path(scene_path_or_dict)
            if not path.exists():
                raise FileNotFoundError(f"Scene file not found: {path}")
            logger.info("Loading scene from %s", path)
            with open(path, 'rb') as fh:
                raw = pickle.load(fh)
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Expected dict in pkl file, got {type(raw).__name__}"
                )
            self._scene: dict = copy.deepcopy(raw)
            self._source_path: Optional[str] = str(path)
        elif isinstance(scene_path_or_dict, dict):
            self._scene = copy.deepcopy(scene_path_or_dict)
            self._source_path = None
        else:
            raise TypeError(
                f"Expected str, Path, or dict; got {type(scene_path_or_dict).__name__}"
            )

        self._validate_scene()
        logger.info(
            "Scene loaded: %d agents, ego_idx=%d, %d lanes, %d traffic-light frames",
            self.num_agents,
            self.ego_idx,
            len(self._scene.get('lane_graph', {}).get('lanes', {})),
            len(self._scene.get('traffic_lights', [])),
        )

    # ------------------------------------------------------------ validation
    def _validate_scene(self) -> None:
        """Check that the scene dict has the minimum required structure."""
        required_keys = {'objects', 'av_idx', 'lane_graph', 'traffic_lights'}
        missing = required_keys - set(self._scene.keys())
        if missing:
            raise ValueError(f"Scene dict missing required keys: {missing}")

        objects = self._scene['objects']
        if not isinstance(objects, list) or len(objects) == 0:
            raise ValueError("Scene must contain at least one agent in 'objects'")

        av_idx = self._scene['av_idx']
        if not (0 <= av_idx < len(objects)):
            raise ValueError(
                f"av_idx={av_idx} out of range for {len(objects)} agents"
            )

    # ----------------------------------------------------------- properties
    @property
    def num_agents(self) -> int:
        """Return the number of agents (objects) in the scene."""
        return len(self._scene['objects'])

    @property
    def ego_idx(self) -> int:
        """Return the index of the ego vehicle (AV)."""
        return self._scene['av_idx']

    # --------------------------------------------------------- agent queries
    def get_agent_info(self, agent_idx: int, frame: int = 30) -> dict:
        """Get agent position, heading, velocity, type, and dimensions at a frame.

        Args:
            agent_idx: Index into the objects list.
            frame: Frame number (0-90).  Default 30 (the 3-second mark, often
                used as the "current" observation boundary).

        Returns:
            Dict with keys: index, type, position (x, y), heading, velocity
            (x, y), valid, width, length.

        Raises:
            IndexError: If agent_idx or frame is out of range.
        """
        if agent_idx < 0 or agent_idx >= self.num_agents:
            raise IndexError(
                f"agent_idx={agent_idx} out of range [0, {self.num_agents})"
            )
        if frame < 0 or frame >= NUM_FRAMES:
            raise IndexError(f"frame={frame} out of range [0, {NUM_FRAMES})")

        obj = self._scene['objects'][agent_idx]
        pos = obj['position'][frame]
        vel = obj['velocity'][frame]
        return {
            'index': agent_idx,
            'type': obj['type'],
            'position': (pos['x'], pos['y']),
            'heading': obj['heading'][frame],
            'velocity': (vel['x'], vel['y']),
            'valid': obj['valid'][frame],
            'width': obj['width'],
            'length': obj['length'],
            'is_ego': agent_idx == self.ego_idx,
        }

    def get_agents_by_type(self, agent_type: str) -> list[int]:
        """Get indices of all agents of a given type.

        Args:
            agent_type: One of 'vehicle', 'pedestrian', 'cyclist', 'other'.

        Returns:
            List of agent indices matching the type.
        """
        agent_type = agent_type.lower()
        return [
            i for i, obj in enumerate(self._scene['objects'])
            if obj['type'] == agent_type
        ]

    def get_nearby_agents(
        self,
        center_pos: tuple[float, float],
        radius: float,
        frame: int = 30,
    ) -> list[int]:
        """Get agent indices within *radius* metres of *center_pos* at *frame*.

        Only agents with ``valid[frame] == True`` are considered.

        Args:
            center_pos: (x, y) reference position in world coordinates.
            radius: Search radius in metres.
            frame: Frame index (0-90).

        Returns:
            List of agent indices sorted by distance (nearest first).
        """
        if frame < 0 or frame >= NUM_FRAMES:
            raise IndexError(f"frame={frame} out of range [0, {NUM_FRAMES})")

        cx, cy = center_pos
        r2 = radius * radius
        results: list[tuple[float, int]] = []

        for i, obj in enumerate(self._scene['objects']):
            if not obj['valid'][frame]:
                continue
            pos = obj['position'][frame]
            dx = pos['x'] - cx
            dy = pos['y'] - cy
            dist2 = dx * dx + dy * dy
            if dist2 <= r2:
                results.append((dist2, i))

        results.sort(key=lambda t: t[0])
        return [idx for _, idx in results]

    # ------------------------------------------------------- agent removal
    def remove_agent(self, agent_idx: int) -> SceneEditor:
        """Remove a single agent from the scene.

        The ego vehicle (``av_idx``) cannot be removed.  If the removed agent
        has an index lower than the ego, ``av_idx`` is decremented so it
        continues to point at the correct object.

        Args:
            agent_idx: Index of the agent to remove.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If attempting to remove the ego vehicle.
            IndexError: If agent_idx is out of range.
        """
        if agent_idx < 0 or agent_idx >= self.num_agents:
            raise IndexError(
                f"agent_idx={agent_idx} out of range [0, {self.num_agents})"
            )
        if agent_idx == self.ego_idx:
            raise ValueError("Cannot remove the ego vehicle (av_idx)")

        del self._scene['objects'][agent_idx]

        # Adjust ego index if necessary
        if agent_idx < self._scene['av_idx']:
            self._scene['av_idx'] -= 1

        logger.debug(
            "Removed agent %d. %d agents remain, ego_idx=%d",
            agent_idx, self.num_agents, self.ego_idx,
        )
        return self

    def remove_agents(self, agent_indices: list[int]) -> SceneEditor:
        """Remove multiple agents by index.

        Agents are removed in descending index order so that earlier removals
        do not shift the indices of later ones.

        Args:
            agent_indices: List of indices to remove.

        Returns:
            ``self`` for method chaining.
        """
        # De-duplicate and sort descending so removals don't shift indices
        for idx in sorted(set(agent_indices), reverse=True):
            self.remove_agent(idx)
        return self

    def remove_agents_by_type(self, agent_type: str) -> SceneEditor:
        """Remove all agents of *agent_type* (except the ego).

        Args:
            agent_type: Agent type string, e.g. ``'pedestrian'``.

        Returns:
            ``self`` for method chaining.
        """
        agent_type = agent_type.lower()
        indices = [
            i for i, obj in enumerate(self._scene['objects'])
            if obj['type'] == agent_type and i != self.ego_idx
        ]
        if not indices:
            logger.info("No non-ego agents of type '%s' to remove.", agent_type)
            return self
        logger.info(
            "Removing %d agents of type '%s'.", len(indices), agent_type,
        )
        return self.remove_agents(indices)

    # -------------------------------------------------- traffic light edits
    def set_traffic_light(
        self,
        lane_id: int,
        state: int,
        frames: Optional[range] = None,
    ) -> SceneEditor:
        """Set traffic light state for a specific lane across frames.

        Args:
            lane_id: The lane identifier the traffic light controls.
            state: New state value (0 = Red/Unknown, 4 = Green).
            frames: Range of frame indices to modify.  If ``None``, all 91
                frames are modified.

        Returns:
            ``self`` for method chaining.
        """
        tl_list = self._scene['traffic_lights']
        if frames is None:
            frames = range(len(tl_list))

        changed = 0
        for fi in frames:
            if fi < 0 or fi >= len(tl_list):
                logger.warning("Frame %d out of range, skipping.", fi)
                continue
            for light in tl_list[fi]:
                if light['lane'] == lane_id:
                    light['state'] = state
                    changed += 1

        logger.info(
            "Set traffic light lane=%d to state=%d across %d frame entries.",
            lane_id, state, changed,
        )
        return self

    def flip_traffic_lights(self) -> SceneEditor:
        """Flip all traffic lights: red (0) becomes green (4) and vice versa.

        Lights with states other than 0 or 4 are left unchanged.

        Returns:
            ``self`` for method chaining.
        """
        flipped = 0
        for frame_lights in self._scene['traffic_lights']:
            for light in frame_lights:
                if light['state'] == TL_RED:
                    light['state'] = TL_GREEN
                    flipped += 1
                elif light['state'] == TL_GREEN:
                    light['state'] = TL_RED
                    flipped += 1
        logger.info("Flipped %d traffic light entries.", flipped)
        return self

    # --------------------------------------------------- agent insertion
    @staticmethod
    def _default_dims(agent_type: str) -> tuple[float, float]:
        """Return (width, length) defaults for an agent type."""
        return DEFAULT_DIMENSIONS.get(agent_type, DEFAULT_DIMENSIONS['vehicle'])

    def add_stationary_agent(
        self,
        agent_type: str,
        position: tuple[float, float],
        heading: float,
        width: Optional[float] = None,
        length: Optional[float] = None,
    ) -> SceneEditor:
        """Add a stationary agent at a fixed position for all 91 frames.

        The agent is valid at every frame, with zero velocity.

        Args:
            agent_type: One of 'vehicle', 'pedestrian', 'cyclist', 'other'.
            position: (x, y) in world coordinates.
            heading: Heading angle in degrees.
            width: Agent width in metres; uses type default if ``None``.
            length: Agent length in metres; uses type default if ``None``.

        Returns:
            ``self`` for method chaining.
        """
        default_w, default_l = self._default_dims(agent_type)
        if width is None:
            width = default_w
        if length is None:
            length = default_l

        px, py = position
        agent: dict[str, Any] = {
            'type': agent_type,
            'position': [{'x': px, 'y': py}] * NUM_FRAMES,
            'heading': [heading] * NUM_FRAMES,
            'velocity': [{'x': 0.0, 'y': 0.0}] * NUM_FRAMES,
            'valid': [True] * NUM_FRAMES,
            'width': float(width),
            'length': float(length),
        }
        self._scene['objects'].append(agent)
        new_idx = self.num_agents - 1
        logger.info(
            "Added stationary %s at (%.1f, %.1f) heading=%.1f as agent %d.",
            agent_type, px, py, heading, new_idx,
        )
        return self

    def add_moving_agent(
        self,
        agent_type: str,
        start_pos: tuple[float, float],
        end_pos: tuple[float, float],
        heading: float,
        speed: float,
        width: Optional[float] = None,
        length: Optional[float] = None,
        start_frame: int = 0,
    ) -> SceneEditor:
        """Add an agent moving in a straight line from *start_pos* toward *end_pos*.

        The agent appears at *start_frame* and moves at *speed* m/s.  If the
        agent reaches *end_pos* before frame 90 it stays at *end_pos* for the
        remaining frames.  Frames before *start_frame* are marked invalid.

        The velocity vector is derived from the direction of
        ``end_pos - start_pos`` scaled to *speed*.

        Args:
            agent_type: Agent type string.
            start_pos: (x, y) starting position.
            end_pos: (x, y) ending position.
            heading: Heading angle in degrees.
            speed: Scalar speed in metres per second.
            width: Agent width; type default if ``None``.
            length: Agent length; type default if ``None``.
            start_frame: Frame at which the agent first appears (default 0).

        Returns:
            ``self`` for method chaining.
        """
        default_w, default_l = self._default_dims(agent_type)
        if width is None:
            width = default_w
        if length is None:
            length = default_l

        sx, sy = start_pos
        ex, ey = end_pos
        dx = ex - sx
        dy = ey - sy
        total_dist = math.sqrt(dx * dx + dy * dy)

        if total_dist < 1e-6:
            # start == end: treat as stationary
            return self.add_stationary_agent(
                agent_type, start_pos, heading, width, length,
            )

        # Unit direction vector
        ux = dx / total_dist
        uy = dy / total_dist
        vx = ux * speed
        vy = uy * speed
        dt = 1.0 / FRAME_RATE_HZ  # seconds per frame

        positions: list[dict[str, float]] = []
        velocities: list[dict[str, float]] = []
        headings: list[float] = []
        valids: list[bool] = []

        for frame in range(NUM_FRAMES):
            if frame < start_frame:
                # Not yet appeared
                positions.append({'x': sx, 'y': sy})
                velocities.append({'x': 0.0, 'y': 0.0})
                headings.append(heading)
                valids.append(False)
            else:
                elapsed = (frame - start_frame) * dt
                travel = speed * elapsed
                if travel >= total_dist:
                    # Arrived at destination
                    positions.append({'x': ex, 'y': ey})
                    velocities.append({'x': 0.0, 'y': 0.0})
                else:
                    positions.append({
                        'x': sx + ux * travel,
                        'y': sy + uy * travel,
                    })
                    velocities.append({'x': vx, 'y': vy})
                headings.append(heading)
                valids.append(True)

        agent: dict[str, Any] = {
            'type': agent_type,
            'position': positions,
            'heading': headings,
            'velocity': velocities,
            'valid': valids,
            'width': float(width),
            'length': float(length),
        }
        self._scene['objects'].append(agent)
        new_idx = self.num_agents - 1
        logger.info(
            "Added moving %s from (%.1f, %.1f) to (%.1f, %.1f) speed=%.1f m/s "
            "starting frame %d as agent %d.",
            agent_type, sx, sy, ex, ey, speed, start_frame, new_idx,
        )
        return self

    # --------------------------------------------------------- scene access
    def get_scene(self) -> dict:
        """Return the (potentially modified) scene dictionary.

        The returned dict is a reference to the internal state -- callers
        should use ``copy.deepcopy`` if they need an independent copy.
        """
        return self._scene

    def save(self, output_path: str) -> None:
        """Serialise the scene to a pkl file.

        Args:
            output_path: Destination file path.  Parent directories are
                created automatically.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'wb') as fh:
            pickle.dump(self._scene, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Scene saved to %s", out)

    # ------------------------------------------------------ describe / diff
    def describe(self) -> str:
        """Return a human-readable multi-line summary of the scene."""
        scene = self._scene
        objects = scene['objects']
        lanes = scene.get('lane_graph', {}).get('lanes', {})
        tl_frames = scene.get('traffic_lights', [])

        # Count agent types
        type_counts: dict[str, int] = {}
        for obj in objects:
            t = obj['type']
            type_counts[t] = type_counts.get(t, 0) + 1

        # Count valid agents at frame 30
        valid_at_30 = sum(1 for o in objects if o['valid'][30])

        # Traffic light summary
        tl_states: dict[int, int] = {}
        if tl_frames:
            for light in tl_frames[0]:
                tl_states[light['state']] = tl_states.get(light['state'], 0) + 1

        # Ego info
        ego = self.get_agent_info(self.ego_idx, frame=30)

        lines = [
            "=== Scene Summary ===",
            f"Source: {self._source_path or '(in-memory)'}",
            f"Total agents: {self.num_agents}",
            f"  by type: {type_counts}",
            f"  valid at frame 30: {valid_at_30}",
            f"Ego vehicle (idx={self.ego_idx}):",
            f"  position: ({ego['position'][0]:.1f}, {ego['position'][1]:.1f})",
            f"  heading: {ego['heading']:.1f} deg",
            f"  velocity: ({ego['velocity'][0]:.2f}, {ego['velocity'][1]:.2f}) m/s",
            f"Lanes: {len(lanes)}",
            f"Traffic light frames: {len(tl_frames)}",
        ]
        if tl_states:
            state_names = {0: 'Red/Unknown', 4: 'Green'}
            state_str = ', '.join(
                f"{state_names.get(s, f'state={s}')}: {c}"
                for s, c in sorted(tl_states.items())
            )
            lines.append(f"  lights at frame 0: {state_str}")

        # Extra lane_graph features
        lg = scene.get('lane_graph', {})
        for extra_key in ('road_edges', 'crosswalks', 'stop_signs'):
            if extra_key in lg:
                lines.append(f"  {extra_key}: {len(lg[extra_key])}")

        return '\n'.join(lines)

    def diff(self, other: SceneEditor) -> dict:
        """Compare this scene with *other* and return a summary of differences.

        Args:
            other: Another ``SceneEditor`` instance.

        Returns:
            Dict with keys describing detected differences:
            - ``agent_count``: (self_count, other_count) if different
            - ``agents_added``: count of agents present only in other
            - ``agents_removed``: count of agents present only in self
            - ``ego_idx``: (self_idx, other_idx) if different
            - ``traffic_light_changes``: number of frame/light entries that
              differ in state
            - ``lane_count``: (self, other) if different
        """
        result: dict[str, Any] = {}
        s_obj = self._scene['objects']
        o_obj = other._scene['objects']

        if len(s_obj) != len(o_obj):
            result['agent_count'] = (len(s_obj), len(o_obj))
            if len(o_obj) > len(s_obj):
                result['agents_added'] = len(o_obj) - len(s_obj)
            else:
                result['agents_removed'] = len(s_obj) - len(o_obj)

        if self.ego_idx != other.ego_idx:
            result['ego_idx'] = (self.ego_idx, other.ego_idx)

        # Traffic light diffs
        s_tl = self._scene.get('traffic_lights', [])
        o_tl = other._scene.get('traffic_lights', [])
        tl_changes = 0
        for fi in range(min(len(s_tl), len(o_tl))):
            s_lights = {l['lane']: l['state'] for l in s_tl[fi]}
            o_lights = {l['lane']: l['state'] for l in o_tl[fi]}
            for lane_id in set(s_lights) | set(o_lights):
                if s_lights.get(lane_id) != o_lights.get(lane_id):
                    tl_changes += 1
        if tl_changes:
            result['traffic_light_changes'] = tl_changes

        # Lane count comparison
        s_lanes = len(self._scene.get('lane_graph', {}).get('lanes', {}))
        o_lanes = len(other._scene.get('lane_graph', {}).get('lanes', {}))
        if s_lanes != o_lanes:
            result['lane_count'] = (s_lanes, o_lanes)

        # Agent type distribution comparison
        def _type_dist(objs):
            d: dict[str, int] = {}
            for o in objs:
                d[o['type']] = d.get(o['type'], 0) + 1
            return d

        s_types = _type_dist(s_obj)
        o_types = _type_dist(o_obj)
        if s_types != o_types:
            result['agent_types'] = {'self': s_types, 'other': o_types}

        if not result:
            result['identical'] = True

        return result

    # ------------------------------------------------ counterfactual factory
    @staticmethod
    def create_counterfactual_pair(
        scene_path: str,
        modification_fn: Callable[[SceneEditor], None],
    ) -> tuple[dict, dict]:
        """Create an original and modified scene pair for A/B comparison.

        The *modification_fn* receives a ``SceneEditor`` wrapping a deep copy
        of the original scene and should mutate it in place (the editor's
        chaining API is convenient here).

        Args:
            scene_path: Path to the original scene ``.pkl`` file.
            modification_fn: A callable ``f(editor: SceneEditor) -> None``
                that applies the desired counterfactual modification.

        Returns:
            ``(original_scene_dict, modified_scene_dict)`` -- both are
            independent deep copies.
        """
        original = SceneEditor(scene_path)
        modified = SceneEditor(scene_path)
        modification_fn(modified)
        return original.get_scene(), modified.get_scene()

    # --------------------------------------------------------------- repr
    def __repr__(self) -> str:
        src = self._source_path or 'in-memory'
        return (
            f"<SceneEditor agents={self.num_agents} ego={self.ego_idx} "
            f"src='{src}'>"
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def batch_counterfactual_experiment(
    scene_paths: list[str],
    modification_fn: Callable[[SceneEditor], None],
    model_fn: Optional[Callable[[dict], Any]] = None,
    output_dir: Optional[str] = None,
) -> list[dict]:
    """Run counterfactual experiments on multiple scenes.

    For each scene the function:

    1. Loads the original scene.
    2. Applies *modification_fn* to create a counterfactual variant.
    3. Optionally runs *model_fn* on both scenes.
    4. Optionally saves both scenes to *output_dir*.

    Args:
        scene_paths: List of paths to original scene ``.pkl`` files.
        modification_fn: Callable that takes a ``SceneEditor`` and modifies
            it in place.
        model_fn: Optional callable that takes a scene dict and returns a
            model output (e.g. predicted trajectory, attention maps).  If
            provided, both original and modified scenes are passed through.
        output_dir: Optional directory to save modified scenes.  Files are
            saved as ``<original_stem>_modified.pkl``.

    Returns:
        List of result dicts, one per scene, each containing:
        - ``scene_path``: the original path
        - ``original_scene``: original scene dict
        - ``modified_scene``: modified scene dict
        - ``diff``: dict of differences between original and modified
        - ``original_output``: model output on original (if model_fn given)
        - ``modified_output``: model output on modified (if model_fn given)
    """
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for i, sp in enumerate(scene_paths):
        logger.info(
            "[%d/%d] Processing %s", i + 1, len(scene_paths), sp,
        )
        try:
            original_scene, modified_scene = SceneEditor.create_counterfactual_pair(
                sp, modification_fn,
            )

            # Compute diff
            orig_editor = SceneEditor(original_scene)
            mod_editor = SceneEditor(modified_scene)
            diff = orig_editor.diff(mod_editor)

            entry: dict[str, Any] = {
                'scene_path': sp,
                'original_scene': original_scene,
                'modified_scene': modified_scene,
                'diff': diff,
            }

            # Optional model inference
            if model_fn is not None:
                logger.info("  Running model on original...")
                entry['original_output'] = model_fn(original_scene)
                logger.info("  Running model on modified...")
                entry['modified_output'] = model_fn(modified_scene)

            # Optional save
            if output_dir is not None:
                stem = Path(sp).stem
                mod_path = os.path.join(output_dir, f"{stem}_modified.pkl")
                mod_editor.save(mod_path)
                entry['modified_path'] = mod_path

            results.append(entry)

        except Exception:
            logger.exception("Failed to process %s", sp)
            results.append({
                'scene_path': sp,
                'error': True,
            })

    logger.info(
        "Batch complete: %d/%d scenes processed successfully.",
        sum(1 for r in results if 'error' not in r),
        len(scene_paths),
    )
    return results


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import glob

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s  %(message)s',
    )

    TRAIN_DIR = '/home/xingnan/workspace/scenario_dreamer_waymo_big/train'

    # Find one sample scene
    scene_files = sorted(glob.glob(os.path.join(TRAIN_DIR, '*.pkl')))
    if not scene_files:
        logger.error("No pkl files found in %s", TRAIN_DIR)
        raise SystemExit(1)

    sample_path = scene_files[0]
    logger.info("Using sample scene: %s", sample_path)

    # --- Basic inspection ---
    editor = SceneEditor(sample_path)
    print(editor.describe())
    print()
    print(repr(editor))
    print()

    # --- Ego info ---
    ego_info = editor.get_agent_info(editor.ego_idx)
    print(f"Ego agent info at frame 30: {ego_info}")
    print()

    # --- Nearby agents ---
    ego_pos = ego_info['position']
    nearby = editor.get_nearby_agents(ego_pos, radius=20.0, frame=30)
    print(f"Agents within 20 m of ego at frame 30: {nearby}")
    for idx in nearby[:5]:
        info = editor.get_agent_info(idx, frame=30)
        print(f"  agent {idx}: {info['type']}, pos=({info['position'][0]:.1f}, {info['position'][1]:.1f})")
    print()

    # --- Agents by type ---
    for atype in ('vehicle', 'pedestrian', 'cyclist'):
        indices = editor.get_agents_by_type(atype)
        print(f"{atype}s: {len(indices)} agents")
    print()

    # --- Counterfactual 1: Remove all pedestrians ---
    print("--- Counterfactual: Remove all pedestrians ---")
    orig_scene, no_ped_scene = SceneEditor.create_counterfactual_pair(
        sample_path,
        lambda e: e.remove_agents_by_type('pedestrian'),
    )
    orig_ed = SceneEditor(orig_scene)
    mod_ed = SceneEditor(no_ped_scene)
    diff = orig_ed.diff(mod_ed)
    print(f"Diff: {diff}")
    print(f"Original agents: {orig_ed.num_agents}, Modified agents: {mod_ed.num_agents}")
    print()

    # --- Counterfactual 2: Flip traffic lights ---
    print("--- Counterfactual: Flip traffic lights ---")
    orig_scene2, flipped_scene = SceneEditor.create_counterfactual_pair(
        sample_path,
        lambda e: e.flip_traffic_lights(),
    )
    orig_ed2 = SceneEditor(orig_scene2)
    flip_ed = SceneEditor(flipped_scene)
    diff2 = orig_ed2.diff(flip_ed)
    print(f"Diff: {diff2}")
    print()

    # --- Counterfactual 3: Add a stationary vehicle in front of ego ---
    print("--- Counterfactual: Add stationary vehicle near ego ---")
    def add_blocker(e: SceneEditor):
        ego = e.get_agent_info(e.ego_idx, frame=30)
        ex, ey = ego['position']
        heading_rad = math.radians(ego['heading'])
        # Place a car 10 m ahead of ego
        bx = ex + 10.0 * math.cos(heading_rad)
        by = ey + 10.0 * math.sin(heading_rad)
        e.add_stationary_agent('vehicle', (bx, by), ego['heading'])

    orig_scene3, blocked_scene = SceneEditor.create_counterfactual_pair(
        sample_path, add_blocker,
    )
    orig_ed3 = SceneEditor(orig_scene3)
    block_ed = SceneEditor(blocked_scene)
    diff3 = orig_ed3.diff(block_ed)
    print(f"Diff: {diff3}")
    print(f"Original agents: {orig_ed3.num_agents}, Blocked agents: {block_ed.num_agents}")
    print()

    # --- Counterfactual 4: Add a moving pedestrian crossing in front ---
    print("--- Counterfactual: Add moving pedestrian crossing ---")
    def add_jaywalker(e: SceneEditor):
        ego = e.get_agent_info(e.ego_idx, frame=30)
        ex, ey = ego['position']
        heading_rad = math.radians(ego['heading'])
        # Perpendicular direction (to the right of ego's heading)
        perp_x = math.sin(heading_rad)
        perp_y = -math.cos(heading_rad)
        # Start 15 m ahead, 10 m to the right; walk across to 10 m to the left
        ahead_x = ex + 15.0 * math.cos(heading_rad)
        ahead_y = ey + 15.0 * math.sin(heading_rad)
        start = (ahead_x + 10.0 * perp_x, ahead_y + 10.0 * perp_y)
        end = (ahead_x - 10.0 * perp_x, ahead_y - 10.0 * perp_y)
        cross_heading = math.degrees(math.atan2(-perp_y, -perp_x))
        e.add_moving_agent(
            'pedestrian', start, end,
            heading=cross_heading, speed=1.4,  # typical walking speed
            start_frame=20,
        )

    orig_scene4, jaywalker_scene = SceneEditor.create_counterfactual_pair(
        sample_path, add_jaywalker,
    )
    jw_ed = SceneEditor(jaywalker_scene)
    diff4 = SceneEditor(orig_scene4).diff(jw_ed)
    print(f"Diff: {diff4}")
    print(f"Modified scene agents: {jw_ed.num_agents}")
    print()

    # --- Batch experiment (no model, just scene creation) ---
    print("--- Batch counterfactual (first 3 scenes, remove pedestrians) ---")
    batch_paths = scene_files[:3]
    batch_results = batch_counterfactual_experiment(
        batch_paths,
        modification_fn=lambda e: e.remove_agents_by_type('pedestrian'),
    )
    for r in batch_results:
        if 'error' in r:
            print(f"  {r['scene_path']}: ERROR")
        else:
            print(f"  {r['scene_path']}: diff={r['diff']}")

    print("\nDone.")
