"""
State Encoding: Game State -> Neural Network Input Vector.

Encodes the TFT game state into a fixed-size vector suitable for
neural network input. Includes board state, bench, shop, economy,
synergies, and opponent information.
"""

import numpy as np
from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

from src.data.loaders import load_champions

if TYPE_CHECKING:
    from src.core.game_state import PlayerState, GameState


@dataclass
class EncoderConfig:
    """State encoder configuration."""

    # Embedding dimensions
    champion_embed_dim: int = 32
    item_embed_dim: int = 16
    trait_embed_dim: int = 8

    # Board/bench sizes
    board_size: int = 28  # 4x7 hexes
    bench_size: int = 9
    shop_size: int = 5

    # Game constants
    num_champions: int = 100
    num_items: int = 46
    num_traits: int = 44
    max_players: int = 8

    # Normalization
    max_gold: int = 100
    max_hp: int = 100
    max_level: int = 10

    # Opponent encoding options (OPTIMIZED: summary instead of full units)
    encode_opponent_units: bool = False  # Disabled - use summary instead
    opponent_unit_dim: int = 8  # Compact encoding per unit (legacy)
    opponent_summary_dim: int = 15  # Summary encoding per opponent (much smaller!)

    # Champion contest encoding (track how many of each champion opponents have)
    encode_champion_contest: bool = True  # Track opponent champion counts
    champion_contest_dim: int = 60  # Top N champions to track (reduced from num_champions)

    # Augment encoding
    num_augment_choices: int = 3
    augment_embed_dim: int = 20  # Per-augment embedding dimension
    num_augment_categories: int = 5  # combat, economy, trait, utility, etc.


class StateEncoder:
    """
    Encodes game state for neural network input.

    Usage:
        encoder = StateEncoder()
        state_vector = encoder.encode(player_state, game_state)
    """

    def __init__(self, config: Optional[EncoderConfig] = None):
        self.config = config or EncoderConfig()

        # Champion ID -> index mapping
        self._champion_to_idx: Dict[str, int] = {}
        self._build_mappings()

        # State dimension calculation
        self.state_dim = self._calculate_state_dim()

    def _build_mappings(self):
        """Build ID -> index mappings."""
        try:
            champions = load_champions()
            for idx, champ in enumerate(champions):
                self._champion_to_idx[champ.id] = idx
        except Exception:
            # Fallback if data loading fails
            pass

    def _calculate_state_dim(self) -> int:
        """Calculate total state dimension."""
        c = self.config

        # Simplified version
        unit_dim = c.champion_embed_dim + 3 + c.item_embed_dim  # embed + star + items

        board_dim = c.board_size * unit_dim
        bench_dim = c.bench_size * unit_dim
        shop_dim = c.shop_size * (c.champion_embed_dim + 5)  # embed + cost one-hot
        economy_dim = 10  # gold, HP, level, XP, streak, etc.
        synergy_dim = c.num_traits * 2  # count + active
        stage_dim = 10
        other_players_dim = (c.max_players - 1) * 5  # HP, level, strength, etc.

        # Opponent encoding (OPTIMIZED: summary instead of full units)
        if c.encode_opponent_units:
            # Legacy: full unit encoding (expensive!)
            opponent_units_per_player = (c.board_size + c.bench_size) * c.opponent_unit_dim
            opponent_units_dim = (c.max_players - 1) * opponent_units_per_player
        else:
            # New: compact summary per opponent (7 * 15 = 105 dims vs 2,072!)
            opponent_units_dim = (c.max_players - 1) * c.opponent_summary_dim

        # Champion contest encoding (how many of each champion opponents have)
        # This helps the agent understand which champions are contested
        champion_contest_dim = c.champion_contest_dim if c.encode_champion_contest else 0

        # Augment choices encoding (3 choices × 20 dims = 60 dims)
        # + 1 flag for "is augment selection phase"
        augment_dim = c.num_augment_choices * c.augment_embed_dim + 1

        return (
            board_dim
            + bench_dim
            + shop_dim
            + economy_dim
            + synergy_dim
            + stage_dim
            + other_players_dim
            + opponent_units_dim
            + champion_contest_dim
            + augment_dim
        )

    def encode(
        self,
        player: "PlayerState",
        game: "GameState",
        player_idx: int = 0,
        augment_choices: Optional[List] = None,
    ) -> np.ndarray:
        """
        Encode full state.

        Args:
            player: Player state to encode.
            game: Full game state.
            player_idx: Index of the player.
            augment_choices: List of augment options during augment selection phase.

        Returns:
            np.ndarray: State vector (float32).
        """
        parts = []

        # 1. Board encoding
        parts.append(self._encode_board(player))

        # 2. Bench encoding
        parts.append(self._encode_bench(player))

        # 3. Shop encoding
        parts.append(self._encode_shop(player, game))

        # 4. Economy state
        parts.append(self._encode_economy(player))

        # 5. Synergies
        parts.append(self._encode_synergies(player))

        # 6. Stage info
        parts.append(self._encode_stage(game))

        # 7. Other players (basic info)
        parts.append(self._encode_other_players(game, player_idx))

        # 8. Other players' units (board + bench)
        if self.config.encode_opponent_units:
            # Legacy: full unit encoding
            parts.append(self._encode_opponent_units(game, player_idx))
        else:
            # Optimized: compact summary encoding
            parts.append(self._encode_opponent_summary(game, player_idx))

        # 9. Champion contest encoding (opponent champion counts)
        if self.config.encode_champion_contest:
            parts.append(self._encode_champion_contest(game, player_idx))

        # 10. Augment choices (during augment selection phase)
        parts.append(self._encode_augment_choices(augment_choices))

        return np.concatenate(parts).astype(np.float32)

    def _encode_board(self, player: "PlayerState") -> np.ndarray:
        """Encode board units."""
        c = self.config
        unit_dim = c.champion_embed_dim + 3 + c.item_embed_dim
        board = np.zeros(c.board_size * unit_dim, dtype=np.float32)

        # Get board from player
        board_units = getattr(player.units, "board", {})
        if isinstance(board_units, dict):
            for pos, instance in board_units.items():
                # Position -> index (row * 7 + col)
                if isinstance(pos, tuple) and len(pos) == 2:
                    idx = pos[0] * 7 + pos[1]
                else:
                    continue

                if idx >= c.board_size:
                    continue

                start = idx * unit_dim

                # Champion embedding
                champ_id = getattr(instance.champion, "id", "")
                champ_idx = self._champion_to_idx.get(champ_id, 0)
                board[start : start + c.champion_embed_dim] = self._get_champion_embedding(
                    champ_idx
                )

                # Star level (one-hot)
                star = min(getattr(instance, "star_level", 1) - 1, 2)
                board[start + c.champion_embed_dim + star] = 1.0

                # Items
                items = getattr(instance, "items", [])
                item_start = start + c.champion_embed_dim + 3
                board[item_start : item_start + c.item_embed_dim] = self._encode_items(
                    items
                )

        return board

    def _encode_bench(self, player: "PlayerState") -> np.ndarray:
        """Encode bench units."""
        c = self.config
        unit_dim = c.champion_embed_dim + 3 + c.item_embed_dim
        bench = np.zeros(c.bench_size * unit_dim, dtype=np.float32)

        bench_units = getattr(player.units, "bench", [])
        for idx, instance in enumerate(bench_units):
            if instance is None or idx >= c.bench_size:
                continue

            start = idx * unit_dim

            champ_id = getattr(instance.champion, "id", "")
            champ_idx = self._champion_to_idx.get(champ_id, 0)
            bench[start : start + c.champion_embed_dim] = self._get_champion_embedding(
                champ_idx
            )

            star = min(getattr(instance, "star_level", 1) - 1, 2)
            bench[start + c.champion_embed_dim + star] = 1.0

            items = getattr(instance, "items", [])
            item_start = start + c.champion_embed_dim + 3
            bench[item_start : item_start + c.item_embed_dim] = self._encode_items(items)

        return bench

    def _encode_shop(self, player: "PlayerState", game: "GameState") -> np.ndarray:
        """Encode shop state."""
        c = self.config
        slot_dim = c.champion_embed_dim + 5  # embed + cost one-hot (1-5)
        shop = np.zeros(c.shop_size * slot_dim, dtype=np.float32)

        # Get shop from game
        shop_state = None
        if hasattr(game, "get_shop_for_player"):
            shop_state = game.get_shop_for_player(getattr(player, "player_id", 0))
        elif hasattr(game, "shops"):
            shop_state = game.shops.get(getattr(player, "player_id", 0))

        if shop_state is None:
            return shop

        slots = getattr(shop_state, "slots", [])
        for idx, slot in enumerate(slots[: c.shop_size]):
            if slot is None:
                continue

            # Check if purchased
            is_purchased = getattr(slot, "is_purchased", False)
            if is_purchased:
                continue

            # Get champion from slot
            champion = getattr(slot, "champion", slot)
            if champion is None:
                continue

            start = idx * slot_dim

            champ_id = getattr(champion, "id", getattr(champion, "champion_id", ""))
            champ_idx = self._champion_to_idx.get(champ_id, 0)
            shop[start : start + c.champion_embed_dim] = self._get_champion_embedding(
                champ_idx
            )

            # Cost one-hot (1-5)
            cost = min(getattr(champion, "cost", 1) - 1, 4)
            shop[start + c.champion_embed_dim + cost] = 1.0

        return shop

    def _encode_economy(self, player: "PlayerState") -> np.ndarray:
        """Encode economy state."""
        c = self.config
        economy = np.zeros(10, dtype=np.float32)

        gold = getattr(player, "gold", 0)
        hp = getattr(player, "hp", 100)
        level = getattr(player, "level", 1)
        xp = getattr(player, "xp", 0)
        streak = getattr(player, "streak", 0)

        economy[0] = gold / c.max_gold  # Gold (normalized)
        economy[1] = hp / c.max_hp  # HP (normalized)
        economy[2] = level / c.max_level  # Level (normalized)
        economy[3] = xp / 100  # XP (normalized)
        economy[4] = min(streak, 5) / 5  # Win streak (normalized)
        economy[5] = min(-streak if streak < 0 else 0, 5) / 5  # Loss streak
        economy[6] = 1.0 if streak > 0 else 0.0  # Is on win streak?
        economy[7] = len(getattr(player.units, "board", {})) / 10  # Board unit count
        economy[8] = (
            sum(1 for b in getattr(player.units, "bench", []) if b) / 9
        )  # Bench count
        economy[9] = level / 10  # Level (raw normalized)

        return economy

    def _encode_synergies(self, player: "PlayerState") -> np.ndarray:
        """Encode synergies."""
        c = self.config
        synergies = np.zeros(c.num_traits * 2, dtype=np.float32)

        # Get active synergies
        if hasattr(player.units, "get_active_synergies"):
            active_synergies = player.units.get_active_synergies()
        else:
            active_synergies = {}

        for trait_id, data in active_synergies.items():
            # Trait index (simple hash)
            trait_idx = hash(trait_id) % c.num_traits

            if isinstance(data, dict):
                count = data.get("count", 0)
                is_active = data.get("is_active", False)
            else:
                count = getattr(data, "count", 0)
                is_active = getattr(data, "is_active", False)

            synergies[trait_idx * 2] = count / 10  # Count (normalized)
            synergies[trait_idx * 2 + 1] = 1.0 if is_active else 0.0

        return synergies

    def _encode_stage(self, game: "GameState") -> np.ndarray:
        """Encode stage information."""
        stage = np.zeros(10, dtype=np.float32)

        sm = getattr(game, "stage_manager", None)
        if sm is None:
            return stage

        stage_num = getattr(sm, "stage", 1)
        round_num = getattr(sm, "round", 1)

        stage[0] = stage_num / 7  # Stage (normalized)
        stage[1] = round_num / 7  # Round (normalized)

        # Round type flags
        if hasattr(sm, "is_pvp_round"):
            stage[2] = 1.0 if sm.is_pvp_round() else 0.0
        if hasattr(sm, "is_carousel_round"):
            stage[3] = 1.0 if sm.is_carousel_round() else 0.0
        if hasattr(sm, "is_augment_round"):
            stage[4] = 1.0 if sm.is_augment_round() else 0.0
        if hasattr(sm, "get_total_rounds"):
            stage[5] = sm.get_total_rounds() / 50

        return stage

    def _encode_other_players(self, game: "GameState", my_idx: int) -> np.ndarray:
        """Encode other players' information."""
        c = self.config
        others = np.zeros((c.max_players - 1) * 5, dtype=np.float32)

        players = getattr(game, "players", [])
        other_idx = 0

        for idx, player in enumerate(players):
            if idx == my_idx:
                continue

            is_alive = getattr(player, "is_alive", True)
            if not is_alive:
                continue

            if other_idx >= c.max_players - 1:
                break

            start = other_idx * 5
            others[start] = getattr(player, "hp", 100) / c.max_hp
            others[start + 1] = getattr(player, "level", 1) / c.max_level
            others[start + 2] = len(getattr(player.units, "board", {})) / 10
            others[start + 3] = getattr(player, "gold", 0) / c.max_gold

            # Approximate board strength
            if hasattr(player.units, "get_active_synergies"):
                synergy_count = len(player.units.get_active_synergies())
            else:
                synergy_count = 0
            others[start + 4] = synergy_count / 10

            other_idx += 1

        return others

    def _get_champion_embedding(self, champ_idx: int) -> np.ndarray:
        """Get champion embedding (simple version)."""
        embed = np.zeros(self.config.champion_embed_dim, dtype=np.float32)
        embed[champ_idx % self.config.champion_embed_dim] = 1.0
        return embed

    def _encode_items(self, items: List) -> np.ndarray:
        """Encode items."""
        embed = np.zeros(self.config.item_embed_dim, dtype=np.float32)
        for item in items[:3]:
            item_id = getattr(item, "id", getattr(item, "item_id", str(item)))
            item_idx = hash(item_id) % self.config.item_embed_dim
            embed[item_idx] = 1.0
        return embed

    def _encode_opponent_units(self, game: "GameState", my_idx: int) -> np.ndarray:
        """
        Encode other players' board and bench units.

        Each unit is encoded compactly:
        - Champion type (one-hot, reduced dimension)
        - Star level (normalized)
        - Cost tier (normalized)
        - Has items flag
        """
        c = self.config
        slots_per_player = c.board_size + c.bench_size  # 28 + 9 = 37 slots
        opponent_units_per_player = slots_per_player * c.opponent_unit_dim
        total_dim = (c.max_players - 1) * opponent_units_per_player

        result = np.zeros(total_dim, dtype=np.float32)

        players = getattr(game, "players", [])
        opponent_idx = 0

        for idx, player in enumerate(players):
            if idx == my_idx:
                continue

            is_alive = getattr(player, "is_alive", True)
            if not is_alive:
                opponent_idx += 1
                continue

            if opponent_idx >= c.max_players - 1:
                break

            player_start = opponent_idx * opponent_units_per_player

            # Encode board units
            board_units = getattr(player.units, "board", {})
            if isinstance(board_units, dict):
                for pos, instance in board_units.items():
                    if isinstance(pos, tuple) and len(pos) == 2:
                        slot_idx = pos[0] * 7 + pos[1]
                    else:
                        continue

                    if slot_idx >= c.board_size:
                        continue

                    unit_start = player_start + slot_idx * c.opponent_unit_dim
                    self._encode_opponent_unit(result, unit_start, instance)

            # Encode bench units (after board slots)
            bench_units = getattr(player.units, "bench", [])
            for bench_idx, instance in enumerate(bench_units):
                if instance is None or bench_idx >= c.bench_size:
                    continue

                slot_idx = c.board_size + bench_idx
                unit_start = player_start + slot_idx * c.opponent_unit_dim
                self._encode_opponent_unit(result, unit_start, instance)

            opponent_idx += 1

        return result

    def _encode_opponent_unit(
        self, result: np.ndarray, start: int, instance
    ) -> None:
        """
        Encode a single opponent unit into the result array.

        Encoding (8 dimensions):
        - [0-4]: Champion type (cost-based one-hot: 1-5 cost)
        - [5]: Star level (0.33, 0.67, 1.0 for 1/2/3 star)
        - [6]: Is melee (attack range <= 1)
        - [7]: Has items flag
        """
        c = self.config

        # Get champion info
        champion = getattr(instance, "champion", None)
        if champion is None:
            return

        # Cost-based encoding (one-hot for 1-5 cost)
        cost = getattr(champion, "cost", 1)
        cost_idx = min(max(cost - 1, 0), 4)
        result[start + cost_idx] = 1.0

        # Star level (normalized)
        star = getattr(instance, "star_level", 1)
        result[start + 5] = star / 3.0

        # Is melee
        attack_range = getattr(champion.stats, "attack_range", 1) if hasattr(champion, "stats") else 1
        result[start + 6] = 1.0 if attack_range <= 1 else 0.0

        # Has items
        items = getattr(instance, "items", [])
        result[start + 7] = 1.0 if len(items) > 0 else 0.0

    def _encode_opponent_summary(self, game: "GameState", my_idx: int) -> np.ndarray:
        """
        Encode compact summary of other players' boards.

        Much more efficient than full unit encoding!
        Reduces from 2,072 dimensions to 105 (7 opponents × 15 dims).

        Per-opponent encoding (15 dimensions):
        - [0-4]: Unit count by cost tier (1-5 cost)
        - [5-7]: Star level distribution (1/2/3 star counts)
        - [8]: Total board units (normalized)
        - [9]: Total bench units (normalized)
        - [10]: Total items equipped
        - [11]: Average star level
        - [12]: Board strength estimate
        - [13]: Melee/ranged ratio
        - [14]: Is alive flag
        """
        c = self.config
        total_dim = (c.max_players - 1) * c.opponent_summary_dim
        result = np.zeros(total_dim, dtype=np.float32)

        players = getattr(game, "players", [])
        opponent_idx = 0

        for idx, player in enumerate(players):
            if idx == my_idx:
                continue

            if opponent_idx >= c.max_players - 1:
                break

            start = opponent_idx * c.opponent_summary_dim

            is_alive = getattr(player, "is_alive", True)
            result[start + 14] = 1.0 if is_alive else 0.0

            if not is_alive:
                opponent_idx += 1
                continue

            # Collect stats from board and bench
            cost_counts = [0, 0, 0, 0, 0]  # 1-5 cost
            star_counts = [0, 0, 0]  # 1-3 star
            total_items = 0
            total_stars = 0
            unit_count = 0
            melee_count = 0

            # Process board units
            board_units = getattr(player.units, "board", {})
            if isinstance(board_units, dict):
                for instance in board_units.values():
                    self._update_unit_stats(
                        instance, cost_counts, star_counts,
                        total_items, total_stars, unit_count, melee_count
                    )
                    champion = getattr(instance, "champion", None)
                    if champion:
                        cost = getattr(champion, "cost", 1)
                        cost_counts[min(max(cost - 1, 0), 4)] += 1
                        star = getattr(instance, "star_level", 1)
                        star_counts[min(star - 1, 2)] += 1
                        total_stars += star
                        unit_count += 1
                        items = getattr(instance, "items", [])
                        total_items += len(items)
                        attack_range = 1
                        if hasattr(champion, "stats"):
                            attack_range = getattr(champion.stats, "attack_range", 1)
                        if attack_range <= 1:
                            melee_count += 1

            board_count = len(board_units) if isinstance(board_units, dict) else 0

            # Process bench units
            bench_units = getattr(player.units, "bench", [])
            bench_count = sum(1 for b in bench_units if b is not None)

            # Encode summary
            # Cost distribution (normalized)
            for i in range(5):
                result[start + i] = cost_counts[i] / 10.0

            # Star distribution (normalized)
            for i in range(3):
                result[start + 5 + i] = star_counts[i] / 10.0

            # Board/bench counts
            result[start + 8] = board_count / 10.0
            result[start + 9] = bench_count / 9.0

            # Items
            result[start + 10] = total_items / 30.0

            # Average star level
            if unit_count > 0:
                result[start + 11] = total_stars / unit_count / 3.0
            else:
                result[start + 11] = 0.0

            # Board strength estimate (simple: sum of cost * star)
            strength = sum(
                (i + 1) * cost_counts[i] * (sum(star_counts) / max(unit_count, 1))
                for i in range(5)
            )
            result[start + 12] = strength / 50.0

            # Melee ratio
            if unit_count > 0:
                result[start + 13] = melee_count / unit_count
            else:
                result[start + 13] = 0.5

            opponent_idx += 1

        return result

    def _update_unit_stats(
        self, instance, cost_counts, star_counts,
        total_items, total_stars, unit_count, melee_count
    ):
        """Helper to update unit statistics (unused, inline version above)."""
        pass

    def _encode_champion_contest(self, game: "GameState", my_idx: int) -> np.ndarray:
        """
        Encode how many of each champion all opponents have combined.

        This helps the agent understand:
        - Which champions are being contested (hard to 3-star)
        - Which champions are open (easy to collect)
        - Whether to pivot away from contested champions

        Returns:
            np.ndarray: Champion count vector (60 dims for top 60 champions by index)
        """
        c = self.config
        result = np.zeros(c.champion_contest_dim, dtype=np.float32)

        # Count champions across all opponents (board + bench)
        champion_counts: Dict[str, int] = {}

        players = getattr(game, "players", [])
        for idx, player in enumerate(players):
            if idx == my_idx:
                continue

            is_alive = getattr(player, "is_alive", True)
            if not is_alive:
                continue

            units = getattr(player, "units", None)
            if units is None:
                continue

            # Count board units
            board_units = getattr(units, "board", {})
            if isinstance(board_units, dict):
                for instance in board_units.values():
                    champion = getattr(instance, "champion", None)
                    if champion:
                        champ_id = getattr(champion, "id", "")
                        star_level = getattr(instance, "star_level", 1)
                        # Count copies (1-star=1, 2-star=3, 3-star=9)
                        copies = 3 ** (star_level - 1)
                        champion_counts[champ_id] = champion_counts.get(champ_id, 0) + copies

            # Count bench units
            bench_units = getattr(units, "bench", [])
            for instance in bench_units:
                if instance is None:
                    continue
                champion = getattr(instance, "champion", None)
                if champion:
                    champ_id = getattr(champion, "id", "")
                    star_level = getattr(instance, "star_level", 1)
                    copies = 3 ** (star_level - 1)
                    champion_counts[champ_id] = champion_counts.get(champ_id, 0) + copies

        # Encode counts using champion index mapping
        for champ_id, count in champion_counts.items():
            champ_idx = self._champion_to_idx.get(champ_id)
            if champ_idx is not None and champ_idx < c.champion_contest_dim:
                # Normalize: max pool is usually 29 for 1-cost, 13 for 5-cost
                # Use 30 as max for normalization
                result[champ_idx] = min(count / 30.0, 1.0)

        return result

    def _encode_augment_choices(self, augment_choices: Optional[List]) -> np.ndarray:
        """
        Encode augment choices during augment selection phase.

        Per-augment encoding (20 dimensions):
        - [0]: Is augment selection phase (1.0 if choices available)
        - For each of 3 augments:
          - [1-5]: Category one-hot (combat, economy, trait, utility, other)
          - [6]: Tier (silver=0.33, gold=0.67, prismatic=1.0)
          - [7-10]: Effect type flags (damage, defense, utility, economy)
          - [11]: Gold-related flag
          - [12]: Trait-related flag
          - [13-19]: Reserved for specific augment ID hashing

        Args:
            augment_choices: List of augment options (usually 3).

        Returns:
            np.ndarray: Augment encoding vector.
        """
        c = self.config
        # Total: 1 (phase flag) + 3 augments × 20 dims = 61 dims
        total_dim = c.num_augment_choices * c.augment_embed_dim + 1
        result = np.zeros(total_dim, dtype=np.float32)

        if augment_choices is None or len(augment_choices) == 0:
            # Not in augment selection phase
            result[0] = 0.0
            return result

        # In augment selection phase
        result[0] = 1.0

        # Category mapping
        category_map = {
            "combat": 0,
            "economy": 1,
            "trait": 2,
            "utility": 3,
            "other": 4,
        }

        # Tier mapping
        tier_map = {
            "silver": 0.33,
            "gold": 0.67,
            "prismatic": 1.0,
        }

        for idx, augment in enumerate(augment_choices[:c.num_augment_choices]):
            if augment is None:
                continue

            start = 1 + idx * c.augment_embed_dim

            # Category one-hot
            category = getattr(augment, "category", None)
            if category:
                cat_name = category.value if hasattr(category, "value") else str(category)
                cat_idx = category_map.get(cat_name.lower(), 4)
                result[start + cat_idx] = 1.0

            # Tier
            tier = getattr(augment, "tier", None)
            if tier:
                tier_name = tier.value if hasattr(tier, "value") else str(tier)
                result[start + 5] = tier_map.get(tier_name.lower(), 0.5)

            # Effect analysis
            effects = getattr(augment, "effects", {})
            if isinstance(effects, dict):
                # Damage-related
                if any(k in effects for k in ["damage_amp", "ad_bonus", "ap_bonus", "attack_speed"]):
                    result[start + 6] = 1.0
                # Defense-related
                if any(k in effects for k in ["armor", "magic_resist", "health_bonus", "shield"]):
                    result[start + 7] = 1.0
                # Utility-related
                if any(k in effects for k in ["mana", "crit_chance", "crit_damage"]):
                    result[start + 8] = 1.0
                # Economy-related
                if any(k in effects for k in ["gold", "interest", "instant_gold", "income"]):
                    result[start + 9] = 1.0
                    result[start + 10] = 1.0

            # Trait-related (check name or effects)
            name = getattr(augment, "name", "")
            if "emblem" in name.lower() or "heart" in name.lower() or "soul" in name.lower():
                result[start + 11] = 1.0

            # Augment ID hash for uniqueness
            aug_id = getattr(augment, "id", str(augment))
            hash_val = hash(aug_id)
            for i in range(7):
                result[start + 12 + i] = ((hash_val >> i) & 1) * 0.5

        return result
