"""Token bookkeeper: maps token indices to physical entities.

In the scene encoder, agent tokens and map tokens are concatenated:
  tokens[0:A] = agent tokens
  tokens[A:A+M] = map tokens

This module provides clean mapping between token indices and
the actual agents/lanes they represent, essential for attention visualization.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TokenBookkeeper:
    """Maintains mapping between token indices and physical entities.

    Attributes:
        num_agents: number of agent tokens (A)
        num_map: number of map tokens (M)
        agent_obj_ids: list of original scene object indices for each agent token
        lane_ids: list of original lane IDs for each map token
        agent_mask: (A,) bool - which agent slots are valid
        map_mask: (M,) bool - which map slots are valid
        target_agent_indices: list of agent token indices selected as prediction targets
    """
    num_agents: int = 0
    num_map: int = 0
    agent_obj_ids: list = field(default_factory=list)
    lane_ids: list = field(default_factory=list)
    agent_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    map_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    target_agent_indices: list = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.num_agents + self.num_map

    @property
    def agent_range(self) -> tuple:
        """Token index range for agent tokens: [start, end)."""
        return (0, self.num_agents)

    @property
    def map_range(self) -> tuple:
        """Token index range for map tokens: [start, end)."""
        return (self.num_agents, self.num_agents + self.num_map)

    def is_agent_token(self, token_idx: int) -> bool:
        return 0 <= token_idx < self.num_agents

    def is_map_token(self, token_idx: int) -> bool:
        return self.num_agents <= token_idx < self.total_tokens

    def token_to_agent_idx(self, token_idx: int) -> int:
        """Convert token index to agent slot index."""
        assert self.is_agent_token(token_idx), f"Token {token_idx} is not an agent"
        return token_idx

    def token_to_map_idx(self, token_idx: int) -> int:
        """Convert token index to map slot index."""
        assert self.is_map_token(token_idx), f"Token {token_idx} is not a map token"
        return token_idx - self.num_agents

    def token_to_obj_id(self, token_idx: int):
        """Convert token index to original scene object ID."""
        agent_idx = self.token_to_agent_idx(token_idx)
        if agent_idx < len(self.agent_obj_ids):
            return self.agent_obj_ids[agent_idx]
        return None

    def token_to_lane_id(self, token_idx: int):
        """Convert token index to original lane ID."""
        map_idx = self.token_to_map_idx(token_idx)
        if map_idx < len(self.lane_ids):
            return self.lane_ids[map_idx]
        return None

    def get_agent_tokens(self) -> list:
        """Return list of valid agent token indices."""
        return [i for i in range(self.num_agents) if self.agent_mask[i]]

    def get_map_tokens(self) -> list:
        """Return list of valid map token indices."""
        return [i + self.num_agents for i in range(self.num_map) if self.map_mask[i]]

    def describe_token(self, token_idx: int) -> str:
        """Human-readable description of what a token represents."""
        if self.is_agent_token(token_idx):
            aidx = self.token_to_agent_idx(token_idx)
            obj_id = self.agent_obj_ids[aidx] if aidx < len(self.agent_obj_ids) else "?"
            return f"Agent[{aidx}] (obj={obj_id})"
        elif self.is_map_token(token_idx):
            midx = self.token_to_map_idx(token_idx)
            lid = self.lane_ids[midx] if midx < len(self.lane_ids) else "?"
            return f"Map[{midx}] (lane={lid})"
        return f"Unknown[{token_idx}]"

    @staticmethod
    def from_batch_sample(
        agent_obj_ids: list,
        lane_ids: list,
        agent_mask: np.ndarray,
        map_mask: np.ndarray,
        max_agents: int,
        max_map: int,
        target_agent_indices: list = None,
    ) -> "TokenBookkeeper":
        """Create bookkeeper from a single batch sample's metadata."""
        return TokenBookkeeper(
            num_agents=max_agents,
            num_map=max_map,
            agent_obj_ids=agent_obj_ids,
            lane_ids=lane_ids,
            agent_mask=agent_mask,
            map_mask=map_mask,
            target_agent_indices=target_agent_indices or [],
        )
