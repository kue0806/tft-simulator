"""
Agent Pool for Self-Play Training.

Manages past versions of the agent for stable self-play training.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import shutil


@dataclass
class AgentVersion:
    """Metadata for a stored agent version."""
    version_id: int
    timestep: int
    save_time: str
    path: str
    win_rate_vs_random: Optional[float] = None
    avg_placement_vs_random: Optional[float] = None
    elo_rating: float = 1000.0
    games_played: int = 0
    is_strong: bool = True  # Can be filtered out if weak


class AgentPool:
    """
    Agent Pool for Self-Play Training.

    Manages past versions of the agent:
    - Saves current agent periodically
    - Maintains a fixed-size pool (10-20 agents)
    - Provides random sampling for opponent selection
    - Tracks performance metrics (ELO ratings)

    Usage:
        pool = AgentPool(save_dir="models/agent_pool", max_size=20)

        # During training
        if timestep % save_interval == 0:
            pool.save_agent(model, timestep)

        # Get opponent
        opponent = pool.sample_opponent()
    """

    def __init__(
        self,
        save_dir: str = "models/agent_pool",
        max_size: int = 20,
        min_size: int = 10,
        save_interval: int = 50000,
        initial_elo: float = 1000.0,
        elo_k_factor: float = 32.0,
    ):
        """
        Initialize Agent Pool.

        Args:
            save_dir: Directory to store agent checkpoints
            max_size: Maximum number of agents to keep
            min_size: Minimum agents before pruning weak ones
            save_interval: Steps between saves (for reference)
            initial_elo: Starting ELO rating
            elo_k_factor: K-factor for ELO updates
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.max_size = max_size
        self.min_size = min_size
        self.save_interval = save_interval
        self.initial_elo = initial_elo
        self.elo_k_factor = elo_k_factor

        self.versions: List[AgentVersion] = []
        self._next_version_id = 0

        # Load existing pool if available
        self._load_metadata()

    def save_agent(
        self,
        model,
        timestep: int,
        win_rate_vs_random: Optional[float] = None,
        avg_placement_vs_random: Optional[float] = None,
    ) -> AgentVersion:
        """
        Save current agent to the pool.

        Args:
            model: The stable-baselines3 model to save
            timestep: Current training timestep
            win_rate_vs_random: Optional win rate metric
            avg_placement_vs_random: Optional placement metric

        Returns:
            AgentVersion metadata for the saved agent
        """
        version_id = self._next_version_id
        self._next_version_id += 1

        # Create save path
        save_path = self.save_dir / f"agent_v{version_id:04d}"
        model.save(str(save_path))

        # Create version metadata
        version = AgentVersion(
            version_id=version_id,
            timestep=timestep,
            save_time=datetime.now().isoformat(),
            path=str(save_path),
            win_rate_vs_random=win_rate_vs_random,
            avg_placement_vs_random=avg_placement_vs_random,
            elo_rating=self.initial_elo,
            games_played=0,
            is_strong=True,
        )

        self.versions.append(version)

        # Prune if over max size
        if len(self.versions) > self.max_size:
            self._prune_pool()

        # Save metadata
        self._save_metadata()

        return version

    def load_agent(self, version: AgentVersion, model_class):
        """
        Load an agent from the pool.

        Args:
            version: AgentVersion to load
            model_class: The model class (e.g., PPO)

        Returns:
            Loaded model
        """
        return model_class.load(version.path)

    def sample_opponent(
        self,
        exclude_latest: bool = False,
        only_strong: bool = False,
        weighted_by_elo: bool = False,
    ) -> Optional[AgentVersion]:
        """
        Sample a random opponent from the pool.

        Args:
            exclude_latest: If True, don't sample the most recent version
            only_strong: If True, only sample from strong agents
            weighted_by_elo: If True, weight sampling by ELO rating

        Returns:
            Sampled AgentVersion or None if pool is empty
        """
        candidates = self.versions.copy()

        if exclude_latest and len(candidates) > 1:
            candidates = candidates[:-1]

        if only_strong:
            candidates = [v for v in candidates if v.is_strong]

        if not candidates:
            return None

        if weighted_by_elo:
            # Higher ELO = higher probability
            elos = np.array([v.elo_rating for v in candidates])
            # Softmax-style weighting
            probs = np.exp((elos - elos.max()) / 100)
            probs = probs / probs.sum()
            idx = np.random.choice(len(candidates), p=probs)
        else:
            idx = np.random.randint(len(candidates))

        return candidates[idx]

    def sample_opponents(self, n: int, **kwargs) -> List[AgentVersion]:
        """Sample multiple opponents (with replacement)."""
        opponents = []
        for _ in range(n):
            opp = self.sample_opponent(**kwargs)
            if opp is not None:
                opponents.append(opp)
        return opponents

    def get_latest(self) -> Optional[AgentVersion]:
        """Get the most recent agent version."""
        return self.versions[-1] if self.versions else None

    def get_by_version_id(self, version_id: int) -> Optional[AgentVersion]:
        """Get agent by version ID."""
        for v in self.versions:
            if v.version_id == version_id:
                return v
        return None

    def update_elo(
        self,
        winner_version: AgentVersion,
        loser_version: AgentVersion,
        is_draw: bool = False,
    ):
        """
        Update ELO ratings after a game.

        Args:
            winner_version: The winning agent's version
            loser_version: The losing agent's version
            is_draw: If True, treat as a draw (0.5 - 0.5)
        """
        # Expected scores
        exp_winner = 1 / (1 + 10 ** ((loser_version.elo_rating - winner_version.elo_rating) / 400))
        exp_loser = 1 - exp_winner

        # Actual scores
        if is_draw:
            actual_winner = 0.5
            actual_loser = 0.5
        else:
            actual_winner = 1.0
            actual_loser = 0.0

        # Update ratings
        winner_version.elo_rating += self.elo_k_factor * (actual_winner - exp_winner)
        loser_version.elo_rating += self.elo_k_factor * (actual_loser - exp_loser)

        winner_version.games_played += 1
        loser_version.games_played += 1

        self._save_metadata()

    def update_metrics(
        self,
        version: AgentVersion,
        win_rate_vs_random: Optional[float] = None,
        avg_placement_vs_random: Optional[float] = None,
    ):
        """Update metrics for a version."""
        if win_rate_vs_random is not None:
            version.win_rate_vs_random = win_rate_vs_random
        if avg_placement_vs_random is not None:
            version.avg_placement_vs_random = avg_placement_vs_random
        self._save_metadata()

    def mark_weak(self, version: AgentVersion):
        """Mark a version as weak (may be pruned)."""
        version.is_strong = False
        self._save_metadata()

    def get_average_elo(self) -> float:
        """Get average ELO of the pool."""
        if not self.versions:
            return self.initial_elo
        return np.mean([v.elo_rating for v in self.versions])

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent pool."""
        if not self.versions:
            return {
                "size": 0,
                "avg_elo": self.initial_elo,
                "min_elo": self.initial_elo,
                "max_elo": self.initial_elo,
                "strong_count": 0,
            }

        elos = [v.elo_rating for v in self.versions]
        return {
            "size": len(self.versions),
            "avg_elo": np.mean(elos),
            "min_elo": np.min(elos),
            "max_elo": np.max(elos),
            "strong_count": sum(1 for v in self.versions if v.is_strong),
            "total_games": sum(v.games_played for v in self.versions),
            "latest_timestep": self.versions[-1].timestep if self.versions else 0,
        }

    def _prune_pool(self):
        """Remove excess agents, keeping strong ones."""
        if len(self.versions) <= self.max_size:
            return

        # Separate strong and weak
        strong = [v for v in self.versions if v.is_strong]
        weak = [v for v in self.versions if not v.is_strong]

        # First remove weak agents
        while len(self.versions) > self.max_size and weak:
            to_remove = weak.pop(0)
            self._remove_version(to_remove)

        # If still over max, remove oldest strong agents (but keep min_size)
        while len(self.versions) > self.max_size:
            # Keep the latest and some older ones
            # Remove the oldest that isn't in the top performers
            sorted_by_elo = sorted(self.versions, key=lambda v: v.elo_rating, reverse=True)
            # Keep top performers and latest
            protected = set(v.version_id for v in sorted_by_elo[:self.min_size])
            protected.add(self.versions[-1].version_id)

            # Find oldest unprotected
            for v in self.versions:
                if v.version_id not in protected:
                    self._remove_version(v)
                    break
            else:
                # All protected, remove oldest
                self._remove_version(self.versions[0])

    def _remove_version(self, version: AgentVersion):
        """Remove a version from the pool."""
        # Delete file
        path = Path(version.path)
        if path.exists():
            os.remove(str(path) + ".zip")

        # Remove from list
        self.versions = [v for v in self.versions if v.version_id != version.version_id]

    def _save_metadata(self):
        """Save pool metadata to disk."""
        metadata = {
            "next_version_id": self._next_version_id,
            "versions": [asdict(v) for v in self.versions],
        }

        metadata_path = self.save_dir / "pool_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self):
        """Load pool metadata from disk."""
        metadata_path = self.save_dir / "pool_metadata.json"
        if not metadata_path.exists():
            return

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self._next_version_id = metadata.get("next_version_id", 0)
            self.versions = [
                AgentVersion(**v) for v in metadata.get("versions", [])
            ]
        except Exception as e:
            print(f"Warning: Failed to load agent pool metadata: {e}")

    def clear(self):
        """Clear the entire pool (use with caution)."""
        for v in self.versions:
            path = Path(v.path)
            if path.exists():
                os.remove(str(path) + ".zip")

        self.versions = []
        self._next_version_id = 0
        self._save_metadata()

    def __len__(self) -> int:
        return len(self.versions)

    def __repr__(self) -> str:
        stats = self.get_pool_stats()
        return (
            f"AgentPool(size={stats['size']}, "
            f"avg_elo={stats['avg_elo']:.1f}, "
            f"strong={stats['strong_count']})"
        )
