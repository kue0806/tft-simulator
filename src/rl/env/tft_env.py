"""
TFT Gymnasium Environment.

OpenAI Gym compatible environment for TFT reinforcement learning.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from src.core.game_state import GameState, PlayerState
from src.core.shop import Shop
from src.core.unlock_manager import UnlockManager, reset_unlock_manager
from src.core.synergy_calculator import SynergyCalculator
from src.core.stage_manager import StageManager, RoundType, RoundPhase
from src.core.pve_system import PvESystem
from src.core.carousel import CarouselSystem
from src.core.augment import AugmentSystem
from src.combat.combat_engine import CombatEngine
from src.combat.hex_grid import HexPosition, Team

from .state_encoder import StateEncoder, EncoderConfig
from .action_space import ActionSpace, ActionType, ActionConfig
from .reward_calculator import RewardCalculator, RewardConfig


class TFTEnv(gym.Env):
    """
    TFT Gymnasium Environment.

    Single-agent environment where other players use bots/heuristics.

    Usage:
        env = TFTEnv()
        obs, info = env.reset()

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        num_players: int = 8,
        agent_player_idx: int = 0,
        max_rounds: int = 50,
        render_mode: Optional[str] = None,
        encoder_config: Optional[EncoderConfig] = None,
        action_config: Optional[ActionConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ):
        super().__init__()

        self.num_players = num_players
        self.agent_player_idx = agent_player_idx
        self.max_rounds = max_rounds
        self.render_mode = render_mode

        # Components
        self.state_encoder = StateEncoder(encoder_config)
        self.action_space_handler = ActionSpace(action_config)
        self.reward_calculator = RewardCalculator(reward_config)

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_encoder.state_dim,),
            dtype=np.float32,
        )
        self.action_space = self.action_space_handler.gym_space

        # Game state (initialized in reset)
        self.game: Optional[GameState] = None
        self.shops: Dict[int, Shop] = {}
        self.unlock_manager: Optional[UnlockManager] = None
        self.synergy_calculator: Optional[SynergyCalculator] = None
        self.stage_manager: Optional[StageManager] = None
        self.pve_system: Optional[PvESystem] = None
        self.carousel_system: Optional[CarouselSystem] = None
        self.augment_system: Optional[AugmentSystem] = None

        # Episode tracking
        self.current_round = 0
        self.done = False
        self._action_count = 0
        self._max_actions_per_round = 60  # Prevent infinite loops

        # Augment selection state
        self._pending_augment_choices = None  # List of augment options during selection
        self._in_augment_selection = False  # Flag for augment selection phase

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        # Initialize game (GameState creates its own pool and players)
        self.game = GameState(num_players=self.num_players)

        # Give players starting gold
        for player in self.game.players:
            player.gold = 10

        # Initialize unlock manager (reset singleton first)
        reset_unlock_manager()
        self.unlock_manager = UnlockManager(self.game.pool)

        # Initialize synergy calculator
        self.synergy_calculator = SynergyCalculator()

        # Initialize stage manager
        self.stage_manager = StageManager()

        # Initialize PVE system
        self.pve_system = PvESystem()

        # Initialize carousel system
        self.carousel_system = CarouselSystem()

        # Initialize augment system
        self.augment_system = AugmentSystem()

        # Initialize shops for each player using the game's pool
        self.shops = {}
        for i in range(self.num_players):
            player = self.game.players[i]
            player_level = getattr(player, "level", 1)
            shop = Shop(self.game.pool, player_level)
            shop.refresh()  # Fill shop with champions
            self.shops[i] = shop

        # Make shops accessible via game.shops for action space handler
        self.game.shops = self.shops

        # Reset reward calculator
        self.reward_calculator.reset()

        # Episode state
        self.current_round = 0
        self.done = False
        self._action_count = 0

        # Reset augment selection state
        self._pending_augment_choices = None
        self._in_augment_selection = False

        # Initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute action.

        Args:
            action: Action index.

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.done:
            return self._get_observation(), 0.0, True, False, self._get_info()

        player = self.game.players[self.agent_player_idx]

        # Decode and execute action
        action_type, params = self.action_space_handler.decode_action(action)

        # Handle augment selection phase
        if self._in_augment_selection:
            if action_type == ActionType.SELECT_AUGMENT:
                action_valid = self._execute_augment_selection(player, params)
            else:
                # Invalid action during augment selection - treat as selecting first option
                action_valid = False
                self._execute_augment_selection(player, 0)

            # Augment selection done
            self._in_augment_selection = False
            self._pending_augment_choices = None

            # Advance the stage manager after augment selection
            if self.stage_manager:
                self.stage_manager.advance_round()

            # Get observation and info
            obs = self._get_observation()
            info = self._get_info()
            info["action_valid"] = action_valid
            info["augment_selected"] = True

            # Small reward for valid augment selection
            reward = 0.1 if action_valid else 0.0

            return obs, reward, False, False, info

        action_valid = self._execute_action(player, action_type, params)

        self._action_count += 1

        # Check round end (PASS or too many actions)
        round_result = None
        if action_type == ActionType.PASS or self._action_count >= self._max_actions_per_round:
            round_result = self._simulate_round()
            self.current_round += 1
            self._action_count = 0

        # Check game over
        terminated = self._check_game_over()
        truncated = self.current_round >= self.max_rounds
        self.done = terminated or truncated

        # Calculate placement (if game over)
        placement = None
        if self.done:
            placement = self._calculate_placement()

        # Calculate reward
        reward = self.reward_calculator.calculate(
            player=player,
            action_type=action_type,
            action_valid=action_valid,
            round_result=round_result,
            done=self.done,
            placement=placement,
        )

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info["action_valid"] = action_valid
        info["round_result"] = round_result
        info["placement"] = placement

        return obs, reward, terminated, truncated, info

    def _execute_action(
        self, player: PlayerState, action_type: ActionType, params: Any
    ) -> bool:
        """Execute action."""
        c = self.action_space_handler.config
        player_id = getattr(player, "player_id", 0)
        shop = self.shops.get(player_id)

        if action_type == ActionType.PASS:
            return True

        elif action_type == ActionType.BUY:
            slot_idx = params
            if shop and 0 <= slot_idx < c.shop_size:
                return self._execute_buy(shop, slot_idx, player)
            return False

        elif action_type == ActionType.SELL_BENCH:
            bench_idx = params
            if 0 <= bench_idx < c.bench_size:
                return self._execute_sell_bench(player, bench_idx)
            return False

        elif action_type == ActionType.SELL_BOARD:
            board_pos = params
            pos = self.action_space_handler.idx_to_pos(board_pos)
            return self._execute_sell_board(player, pos)

        elif action_type == ActionType.PLACE:
            bench_idx, board_pos = params
            pos = self.action_space_handler.idx_to_pos(board_pos)
            return self._execute_place(player, bench_idx, pos)

        elif action_type == ActionType.REFRESH:
            gold = getattr(player, "gold", 0)
            if gold >= 2 and shop:
                player.gold = gold - 2
                shop.refresh()
                # Record reroll for unlock tracking
                player_id = getattr(player, "player_id", 0)
                if self.unlock_manager:
                    self.unlock_manager.record_reroll(player_id)
                return True
            return False

        elif action_type == ActionType.BUY_XP:
            gold = getattr(player, "gold", 0)
            level = getattr(player, "level", 1)
            if gold >= 4 and level < 10:
                player.gold = gold - 4
                if hasattr(player, "add_xp"):
                    player.add_xp(4)
                else:
                    player.xp = getattr(player, "xp", 0) + 4
                if shop:
                    shop.player_level = getattr(player, "level", 1)
                return True
            return False

        return False

    def _execute_buy(self, shop: Shop, slot_idx: int, player: PlayerState) -> bool:
        """Execute buy action."""
        slots = getattr(shop, "slots", [])
        if slot_idx >= len(slots):
            return False

        slot = slots[slot_idx]
        if slot is None:
            return False

        is_purchased = getattr(slot, "is_purchased", False)
        if is_purchased:
            return False

        champion = getattr(slot, "champion", slot)
        if champion is None:
            return False

        cost = getattr(champion, "cost", 999)
        gold = getattr(player, "gold", 0)

        if gold < cost:
            return False

        # Check bench space
        units = getattr(player, "units", None)
        if units is None:
            return False

        bench = getattr(units, "bench", [])
        bench_space = sum(1 for b in bench if b is None)
        if bench_space <= 0:
            return False

        # Purchase through shop (Shop.purchase only takes slot_idx)
        if hasattr(shop, "purchase"):
            purchased = shop.purchase(slot_idx)
            if purchased:
                player.gold = gold - cost
                # Add to bench
                if hasattr(units, "add_to_bench"):
                    units.add_to_bench(purchased)
                return True
            return False

        # Manual purchase (fallback)
        player.gold = gold - cost

        # Add to bench
        if hasattr(units, "add_to_bench"):
            return units.add_to_bench(champion)

        return True

    def _execute_sell_bench(self, player: PlayerState, bench_idx: int) -> bool:
        """Execute sell bench action."""
        units = getattr(player, "units", None)
        if units is None:
            return False

        # Get unit info before selling for unlock tracking
        bench = getattr(units, "bench", [])
        if bench_idx >= len(bench) or bench[bench_idx] is None:
            return False

        unit = bench[bench_idx]
        champion = getattr(unit, "champion", unit)
        champion_id = getattr(champion, "id", None)
        star_level = getattr(unit, "star_level", 1)

        if hasattr(units, "sell_bench"):
            result = units.sell_bench(bench_idx, self.game.pool)
            if result and self.unlock_manager and champion_id:
                player_id = getattr(player, "player_id", 0)
                self.unlock_manager.record_unit_sold(player_id, champion_id, star_level)
            return result

        # Manual sell
        cost = getattr(champion, "cost", 1)

        # Sell value based on star level
        sell_value = cost * (3 ** (star_level - 1))
        player.gold = getattr(player, "gold", 0) + sell_value

        bench[bench_idx] = None

        # Record for unlock tracking
        if self.unlock_manager and champion_id:
            player_id = getattr(player, "player_id", 0)
            self.unlock_manager.record_unit_sold(player_id, champion_id, star_level)

        return True

    def _execute_sell_board(
        self, player: PlayerState, pos: Tuple[int, int]
    ) -> bool:
        """Execute sell board action."""
        units = getattr(player, "units", None)
        if units is None:
            return False

        # Get unit info before selling for unlock tracking
        board = getattr(units, "board", {})
        if pos not in board:
            return False

        unit = board[pos]
        champion = getattr(unit, "champion", unit)
        champion_id = getattr(champion, "id", None)
        star_level = getattr(unit, "star_level", 1)

        if hasattr(units, "sell_board"):
            result = units.sell_board(pos, self.game.pool)
            if result and self.unlock_manager and champion_id:
                player_id = getattr(player, "player_id", 0)
                self.unlock_manager.record_unit_sold(player_id, champion_id, star_level)
            return result

        # Manual sell
        cost = getattr(champion, "cost", 1)

        sell_value = cost * (3 ** (star_level - 1))
        player.gold = getattr(player, "gold", 0) + sell_value

        del board[pos]

        # Record for unlock tracking
        if self.unlock_manager and champion_id:
            player_id = getattr(player, "player_id", 0)
            self.unlock_manager.record_unit_sold(player_id, champion_id, star_level)

        return True

    def _execute_place(
        self, player: PlayerState, bench_idx: int, pos: Tuple[int, int]
    ) -> bool:
        """Execute place action."""
        units = getattr(player, "units", None)
        if units is None:
            return False

        if hasattr(units, "place_from_bench"):
            return units.place_from_bench(bench_idx, pos)

        # Manual place
        bench = getattr(units, "bench", [])
        board = getattr(units, "board", {})

        if bench_idx >= len(bench) or bench[bench_idx] is None:
            return False

        if pos in board:
            return False

        # Check max units
        level = getattr(player, "level", 1)
        if len(board) >= level:
            return False

        unit = bench[bench_idx]
        board[pos] = unit
        bench[bench_idx] = None
        return True

    def _simulate_round(self) -> Dict[str, Any]:
        """Simulate round based on StageManager round type."""
        agent = self.game.players[self.agent_player_idx]

        # Get round info from StageManager
        if self.stage_manager:
            round_info = self.stage_manager.get_current_round_info()

            # Handle Carousel round
            if round_info.is_carousel:
                result = self._simulate_carousel(agent, round_info.stage)
                self.stage_manager.advance_round()
                return result

            # Handle Augment selection - let RL agent choose
            if round_info.is_augment:
                augment_started = self._start_augment_selection(agent, round_info.stage)
                if augment_started:
                    # Return early - agent needs to select augment
                    # Don't advance round yet
                    return {"augment_selection": True, "won": None, "hp_lost": 0, "damage_dealt": 0}

            # Handle PVE round
            if round_info.is_pve:
                result = self._simulate_pve(agent, round_info.stage)
                self.stage_manager.advance_round()
                return result

            # Advance stage manager after PVP combat
            self.stage_manager.advance_round()

        # Select random alive opponent for PVP
        alive_opponents = [
            p
            for i, p in enumerate(self.game.players)
            if i != self.agent_player_idx and getattr(p, "is_alive", True)
        ]

        if not alive_opponents:
            return {"won": True, "hp_lost": 0, "damage_dealt": 0}

        opponent = np.random.choice(alive_opponents)

        # Get boards for combat
        agent_units = getattr(agent, "units", None)
        opponent_units = getattr(opponent, "units", None)

        agent_board = getattr(agent_units, "board", {}) if agent_units else {}
        opponent_board = getattr(opponent_units, "board", {}) if opponent_units else {}

        # Convert board positions to HexPositions
        def convert_board(board: Dict) -> Dict[HexPosition, Any]:
            """Convert (row, col) tuples to HexPosition objects."""
            converted = {}
            for pos, unit in board.items():
                if isinstance(pos, tuple) and len(pos) == 2:
                    hex_pos = HexPosition(row=pos[0], col=pos[1])
                    converted[hex_pos] = unit
                elif isinstance(pos, HexPosition):
                    converted[pos] = unit
            return converted

        blue_board = convert_board(agent_board)
        red_board = convert_board(opponent_board)

        # Handle empty boards
        if not blue_board and not red_board:
            # Both empty - draw, but still update bots
            self._post_combat_update(agent)
            return {"won": False, "hp_lost": 0, "damage_dealt": 0}
        elif not blue_board:
            # Agent has no units - auto-lose
            hp_lost = self._calculate_hp_loss_from_units(opponent_board, 0)
            agent_hp = getattr(agent, "hp", 100)
            agent.hp = max(0, agent_hp - hp_lost)
            if agent.hp <= 0:
                agent.is_alive = False
            self._post_combat_update(agent)
            return {"won": False, "hp_lost": hp_lost, "damage_dealt": 0}
        elif not red_board:
            # Opponent has no units - auto-win
            damage_dealt = self._calculate_hp_loss_from_units(agent_board, 0)
            self._post_combat_update(agent)
            return {"won": True, "hp_lost": 0, "damage_dealt": damage_dealt}

        # Calculate current stage from round number
        stage = max(2, (self.current_round // 7) + 2)

        # Run real combat
        try:
            engine = CombatEngine(stage=stage)
            engine.setup_combat_from_boards(
                blue_board=blue_board,
                red_board=red_board,
                blue_player=agent,
                red_player=opponent,
            )
            result = engine.run_combat()

            # Determine outcome
            won = result.winner == Team.BLUE
            hp_lost = 0
            damage_dealt = 0

            if won:
                # Agent won - opponent takes damage
                damage_dealt = int(result.total_damage_to_loser)
            else:
                # Agent lost - take damage
                hp_lost = int(result.total_damage_to_loser)
                agent_hp = getattr(agent, "hp", 100)
                agent.hp = max(0, agent_hp - hp_lost)
                if agent.hp <= 0:
                    agent.is_alive = False

            # Update streak
            streak = getattr(agent, "streak", 0)
            if won:
                agent.streak = max(1, streak + 1) if streak >= 0 else 1
            else:
                agent.streak = min(-1, streak - 1) if streak <= 0 else -1

        except Exception as e:
            # Fallback to simplified combat if engine fails
            won, hp_lost, damage_dealt = self._fallback_combat(agent, opponent)

        # Record events for unlock manager
        if self.unlock_manager:
            player_id = getattr(agent, "player_id", 0)

            # Record HP lost
            if hp_lost > 0:
                self.unlock_manager.record_hp_lost(player_id, hp_lost)

            # Record combat result with units on board
            units_on_board = []
            for unit in agent_board.values():
                champion = getattr(unit, "champion", unit)
                champion_id = getattr(champion, "id", None)
                if champion_id:
                    units_on_board.append(champion_id)
            self.unlock_manager.record_combat_result(player_id, won, units_on_board)

            # Record active traits for unlock conditions
            if self.synergy_calculator and agent_board:
                try:
                    board_units = list(agent_board.values())
                    synergies = self.synergy_calculator.calculate_synergies(board_units)
                    for trait_id, active_trait in synergies.items():
                        if active_trait.is_active:
                            self.unlock_manager.record_trait_combat(player_id, trait_id)
                except Exception:
                    pass  # Synergy calculation is optional

            # Update stage for unlock conditions
            stage_num = max(2, (self.current_round // 7) + 2)
            round_in_stage = (self.current_round % 7) + 1
            self.unlock_manager.update_stage(player_id, f"{stage_num}-{round_in_stage}")

            # Check for newly unlocked champions
            self.unlock_manager.check_unlocks(agent)

        # Post-combat updates
        self._post_combat_update(agent)

        return {"won": won, "hp_lost": hp_lost, "damage_dealt": damage_dealt}

    def _calculate_hp_loss_from_units(
        self, winner_board: Dict, loser_remaining: int
    ) -> int:
        """Calculate HP loss from remaining units (for empty board cases)."""
        # Base stage damage
        stage = max(2, (self.current_round // 7) + 2)
        from src.core.constants import BASE_STAGE_DAMAGE, UNIT_DAMAGE_BY_STAR_AND_COST

        base_damage = BASE_STAGE_DAMAGE.get(stage, BASE_STAGE_DAMAGE.get(7, 17))

        # Sum unit damage
        unit_damage = 0
        for unit in winner_board.values():
            star_level = getattr(unit, "star_level", 1)
            champion = getattr(unit, "champion", unit)
            cost = getattr(champion, "cost", 1)

            star_damage_table = UNIT_DAMAGE_BY_STAR_AND_COST.get(
                star_level, UNIT_DAMAGE_BY_STAR_AND_COST.get(1, {})
            )
            unit_damage += star_damage_table.get(cost, cost)

        return base_damage + unit_damage

    def _fallback_combat(
        self, agent: PlayerState, opponent: PlayerState
    ) -> Tuple[bool, int, int]:
        """Fallback simplified combat when engine fails."""
        agent_power = self._estimate_board_power(agent)
        opponent_power = self._estimate_board_power(opponent)

        won = agent_power > opponent_power + np.random.randn() * 10
        hp_lost = 0
        damage_dealt = 0

        if not won:
            agent_units = len(getattr(getattr(agent, "units", None), "board", {}))
            opponent_units = len(
                getattr(getattr(opponent, "units", None), "board", {})
            )
            units_remaining = max(1, opponent_units - agent_units // 2)
            hp_lost = units_remaining + self.current_round // 3

            agent_hp = getattr(agent, "hp", 100)
            agent.hp = max(0, agent_hp - hp_lost)
            if agent.hp <= 0:
                agent.is_alive = False
        else:
            agent_units = len(getattr(getattr(agent, "units", None), "board", {}))
            damage_dealt = agent_units + self.current_round // 3

        return won, hp_lost, damage_dealt

    def _post_combat_update(self, agent: PlayerState) -> None:
        """Common post-combat updates (bot players, income, shop refresh)."""
        # Update bot players
        self._update_bot_players()

        # Give income
        income = self._calculate_income(agent)
        agent.gold = getattr(agent, "gold", 0) + income

        # Add passive XP (2 per round, like real TFT)
        if hasattr(agent, "add_xp"):
            agent.add_xp(2)
        else:
            agent.xp = getattr(agent, "xp", 0) + 2
            # Manual level up check
            xp_thresholds = {1: 2, 2: 2, 3: 6, 4: 10, 5: 20, 6: 36, 7: 56, 8: 80, 9: 100}
            level = getattr(agent, "level", 1)
            while level < 9:
                required = xp_thresholds.get(level, 999)
                if agent.xp >= required:
                    agent.xp -= required
                    agent.level = level + 1
                    level = agent.level
                else:
                    break

        # Refresh shop
        player_id = getattr(agent, "player_id", 0)
        shop = self.shops.get(player_id)
        if shop:
            is_locked = getattr(shop, "is_locked", False)
            if not is_locked:
                shop.refresh()

    def _estimate_board_power(self, player: PlayerState) -> float:
        """Estimate board power (simple heuristic)."""
        power = 0.0

        units = getattr(player, "units", None)
        if units is None:
            return power

        board = getattr(units, "board", {})
        if isinstance(board, dict):
            for unit in board.values():
                champion = getattr(unit, "champion", unit)
                cost = getattr(champion, "cost", 1)
                star_level = getattr(unit, "star_level", 1)

                star_mult = {1: 1.0, 2: 1.8, 3: 3.0}.get(star_level, 1.0)
                power += cost * star_mult * 10

                # Item bonus
                items = getattr(unit, "items", [])
                power += len(items) * 5

        # Synergy bonus
        if hasattr(units, "get_active_synergies"):
            synergies = units.get_active_synergies()
            active_count = 0
            for s in synergies.values():
                if isinstance(s, dict):
                    if s.get("is_active"):
                        active_count += 1
                elif getattr(s, "is_active", False):
                    active_count += 1
            power += active_count * 15

        return power

    def _calculate_income(self, player: PlayerState) -> int:
        """Calculate round income."""
        base_income = 5

        # Interest (max 5)
        gold = getattr(player, "gold", 0)
        interest = min(gold // 10, 5)

        # Streak bonus (simplified)
        streak = getattr(player, "streak", 0)
        streak_bonus = min(abs(streak), 3)

        return base_income + interest + streak_bonus

    def _update_bot_players(self):
        """Update bot players (simple heuristic)."""
        for i, player in enumerate(self.game.players):
            if i == self.agent_player_idx:
                continue

            is_alive = getattr(player, "is_alive", True)
            if not is_alive:
                continue

            shop = self.shops.get(i)
            if not shop:
                continue

            gold = getattr(player, "gold", 0)

            # Random buy
            slots = getattr(shop, "slots", [])
            for slot_idx, slot in enumerate(slots):
                if slot is None:
                    continue

                is_purchased = getattr(slot, "is_purchased", False)
                if is_purchased:
                    continue

                champion = getattr(slot, "champion", slot)
                if champion is None:
                    continue

                cost = getattr(champion, "cost", 999)
                if gold >= cost and np.random.random() < 0.7:
                    # Purchase champion
                    if hasattr(shop, "purchase"):
                        purchased = shop.purchase(slot_idx)
                        if purchased:
                            player.gold = gold - cost
                            # Try to add to bench
                            units = getattr(player, "units", None)
                            if units and hasattr(units, "add_to_bench"):
                                units.add_to_bench(purchased)
                    gold = getattr(player, "gold", 0)

            # Random level up
            gold = getattr(player, "gold", 0)
            level = getattr(player, "level", 1)
            if gold >= 4 and level < 10 and np.random.random() < 0.3:
                player.gold = gold - 4
                if hasattr(player, "add_xp"):
                    player.add_xp(4)
                shop.player_level = getattr(player, "level", 1)

            # Give income
            income = self._calculate_income(player)
            player.gold = getattr(player, "gold", 0) + income

            # Place units from bench to board
            units = getattr(player, "units", None)
            if units:
                level = getattr(player, "level", 1)
                board_count = len(getattr(units, "board", {}))
                bench_list = getattr(units, "bench", [])

                # Place units up to level limit
                for bench_idx, unit in enumerate(bench_list):
                    if unit is None:
                        continue
                    if board_count >= level:
                        break

                    # Find empty board position (rows 0-3, cols 0-6)
                    placed = False
                    for y in range(4):
                        for x in range(7):
                            if hasattr(units, "place_on_board"):
                                if units.place_on_board(unit, x, y):
                                    board_count += 1
                                    placed = True
                                    break
                        if placed:
                            break

            # Refresh shop
            is_locked = getattr(shop, "is_locked", False)
            if not is_locked:
                shop.refresh()

    def _check_game_over(self) -> bool:
        """Check if game is over."""
        agent = self.game.players[self.agent_player_idx]

        # Agent dead
        if not getattr(agent, "is_alive", True):
            return True

        # Only 1 player alive
        alive_count = sum(
            1 for p in self.game.players if getattr(p, "is_alive", True)
        )
        return alive_count <= 1

    def _calculate_placement(self) -> int:
        """Calculate final placement."""
        agent = self.game.players[self.agent_player_idx]
        agent_id = getattr(agent, "player_id", 0)

        # Sort by (is_alive, hp)
        sorted_players = sorted(
            self.game.players,
            key=lambda p: (getattr(p, "is_alive", False), getattr(p, "hp", 0)),
            reverse=True,
        )

        for rank, player in enumerate(sorted_players, 1):
            player_id = getattr(player, "player_id", -1)
            if player_id == agent_id:
                return rank

        return self.num_players

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        player = self.game.players[self.agent_player_idx]
        # Pass augment choices if in augment selection phase
        augment_choices = self._pending_augment_choices if self._in_augment_selection else None
        return self.state_encoder.encode(player, self.game, self.agent_player_idx, augment_choices)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        player = self.game.players[self.agent_player_idx]

        # During augment selection, only augment actions are valid
        if self._in_augment_selection:
            num_choices = len(self._pending_augment_choices) if self._pending_augment_choices else 3
            valid_mask = self.action_space_handler.get_augment_action_mask(num_choices)
        else:
            valid_mask = self.action_space_handler.get_valid_actions(player, self.game)

        info = {
            "round": self.current_round,
            "hp": getattr(player, "hp", 100),
            "gold": getattr(player, "gold", 0),
            "level": getattr(player, "level", 1),
            "board_size": len(
                getattr(getattr(player, "units", None), "board", {})
            ),
            "valid_action_mask": valid_mask,
            "alive_players": sum(
                1 for p in self.game.players if getattr(p, "is_alive", True)
            ),
            "in_augment_selection": self._in_augment_selection,
        }

        # Add stage manager info
        if self.stage_manager:
            round_info = self.stage_manager.get_current_round_info()
            info["stage"] = round_info.stage
            info["round_type"] = round_info.round_type.value
            info["is_pve"] = round_info.is_pve
            info["is_carousel"] = round_info.is_carousel
            info["is_augment"] = round_info.is_augment

        # Add augment info
        augments = getattr(player, "augments", [])
        info["augments"] = len(augments)

        # Add augment choices info during selection
        if self._in_augment_selection and self._pending_augment_choices:
            info["augment_choices"] = [
                {
                    "id": getattr(aug, "id", "unknown"),
                    "name": getattr(aug, "name", "Unknown"),
                    "tier": getattr(aug, "tier", "silver"),
                    "category": getattr(aug, "category", "unknown"),
                }
                for aug in self._pending_augment_choices
            ]

        return info

    def render(self) -> Optional[str]:
        """Render environment."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
        return None

    def _render_ansi(self) -> str:
        """Text rendering."""
        player = self.game.players[self.agent_player_idx]
        units = getattr(player, "units", None)

        board_size = len(getattr(units, "board", {})) if units else 0
        bench_count = (
            sum(1 for b in getattr(units, "bench", []) if b) if units else 0
        )

        lines = [
            f"=== Round {self.current_round} ===",
            f"HP: {getattr(player, 'hp', 100)} | Gold: {getattr(player, 'gold', 0)} | Level: {getattr(player, 'level', 1)}",
            f"Board: {board_size} units",
            f"Bench: {bench_count} units",
            f"Alive players: {sum(1 for p in self.game.players if getattr(p, 'is_alive', True))}",
        ]

        return "\n".join(lines)

    def get_valid_action_mask(self) -> np.ndarray:
        """Get valid action mask (external access)."""
        player = self.game.players[self.agent_player_idx]
        return self.action_space_handler.get_valid_actions(player, self.game)

    def _simulate_carousel(self, agent: PlayerState, stage: str) -> Dict[str, Any]:
        """Simulate carousel round - auto-pick champion with item for agent."""
        if not self.carousel_system:
            return {"won": True, "hp_lost": 0, "damage_dealt": 0, "carousel": True}

        # Generate carousel
        carousel = self.carousel_system.generate_carousel(stage, self.game.pool)
        if not carousel:
            return {"won": True, "hp_lost": 0, "damage_dealt": 0, "carousel": True}

        # Calculate pick order (lower HP picks first)
        player_healths = {
            i: getattr(p, "hp", 100)
            for i, p in enumerate(self.game.players)
            if getattr(p, "is_alive", True)
        }
        pick_order = self.carousel_system.calculate_pick_order(player_healths, stage)

        # Sort by pick round
        pick_order.sort(key=lambda x: (x.pick_round, x.health))

        # Track which carousel champions are taken
        available = [True] * len(carousel)

        for pick in pick_order:
            # Find available champion
            choice_idx = None
            for i, avail in enumerate(available):
                if avail:
                    choice_idx = i
                    break

            if choice_idx is None:
                continue

            available[choice_idx] = False
            chosen = carousel[choice_idx]

            # Give champion and item to the player
            player = self.game.players[pick.player_id]
            units = getattr(player, "units", None)
            if units and hasattr(units, "add_to_bench"):
                # Create a simple champion instance (simplified)
                # In full implementation, would create proper ChampionInstance
                pass  # Champion adding is simplified for now

        # Post-carousel, give passive income and XP
        self._post_combat_update(agent)
        return {"won": True, "hp_lost": 0, "damage_dealt": 0, "carousel": True}

    def _simulate_pve(self, agent: PlayerState, stage: str) -> Dict[str, Any]:
        """Simulate PVE round against monsters."""
        if not self.pve_system:
            # Fallback - agent always wins PVE
            self._post_combat_update(agent)
            return {"won": True, "hp_lost": 0, "damage_dealt": 0, "pve": True}

        # Get monster configuration
        monster_type = self.pve_system.get_monster_type(stage)
        if not monster_type:
            self._post_combat_update(agent)
            return {"won": True, "hp_lost": 0, "damage_dealt": 0, "pve": True}

        # Simplified PVE combat - check if agent has units on board
        agent_units = getattr(agent, "units", None)
        agent_board = getattr(agent_units, "board", {}) if agent_units else {}

        if not agent_board:
            # No units - lose some HP based on monster strength
            monster_config = self.pve_system.get_monster_config(stage)
            damage = monster_config.get("damage", 5) if monster_config else 5
            agent_hp = getattr(agent, "hp", 100)
            agent.hp = max(0, agent_hp - damage)
            if agent.hp <= 0:
                agent.is_alive = False
            self._post_combat_update(agent)
            return {"won": False, "hp_lost": damage, "damage_dealt": 0, "pve": True}

        # Agent has units - they win PVE (simplified)
        # In full implementation, would run actual combat vs monsters

        # Give loot rewards (simplified - just gold)
        gold_reward = 1 if stage.startswith("1-") else 2
        agent.gold = getattr(agent, "gold", 0) + gold_reward

        self._post_combat_update(agent)
        return {"won": True, "hp_lost": 0, "damage_dealt": 0, "pve": True, "gold_gained": gold_reward}

    def _start_augment_selection(self, agent: PlayerState, stage: str) -> bool:
        """
        Start augment selection phase - generate choices for RL agent to select.

        Returns:
            True if augment selection started, False if no choices available.
        """
        if not self.augment_system:
            return False

        player_id = getattr(agent, "player_id", 0)

        # Generate augment choices
        existing_augments = getattr(agent, "augments", [])
        excluded = [a.id if hasattr(a, "id") else str(a) for a in existing_augments]

        try:
            choice = self.augment_system.generate_augment_choices(stage, player_id, excluded)
            if choice and choice.options and len(choice.options) > 0:
                # Store choices for agent to select
                self._pending_augment_choices = choice.options
                self._in_augment_selection = True
                return True
        except Exception:
            pass

        return False

    def _execute_augment_selection(self, agent: PlayerState, choice_idx: int) -> bool:
        """
        Execute augment selection action.

        Args:
            agent: Player state.
            choice_idx: Index of selected augment (0, 1, or 2).

        Returns:
            True if selection was valid.
        """
        if not self._pending_augment_choices:
            return False

        # Validate choice index
        if choice_idx < 0 or choice_idx >= len(self._pending_augment_choices):
            # Invalid choice - select first option as fallback
            choice_idx = 0

        selected = self._pending_augment_choices[choice_idx]

        # Store augment on player
        if not hasattr(agent, "augments"):
            agent.augments = []
        agent.augments.append(selected)

        # Apply immediate augment effects
        self._apply_augment_effects(agent, selected)

        return True

    def _simulate_augment_selection(self, agent: PlayerState, stage: str) -> None:
        """Simulate augment selection - auto-select for bot players."""
        if not self.augment_system:
            return

        player_id = getattr(agent, "player_id", 0)

        # Generate augment choices
        existing_augments = getattr(agent, "augments", [])
        excluded = [a.id if hasattr(a, "id") else str(a) for a in existing_augments]

        try:
            choice = self.augment_system.generate_augment_choices(stage, player_id, excluded)
            if choice and choice.options:
                # Auto-select first option for bots
                selected = choice.options[0]
                choice.selected = selected

                # Store augment on player
                if not hasattr(agent, "augments"):
                    agent.augments = []
                agent.augments.append(selected)

                # Apply immediate augment effects (simplified)
                self._apply_augment_effects(agent, selected)
        except Exception:
            pass  # Augment selection is optional

    def _apply_augment_effects(self, agent: PlayerState, augment) -> None:
        """Apply immediate augment effects."""
        effects = getattr(augment, "effects", {})
        if not effects:
            return

        # Apply gold bonuses
        if "instant_gold" in effects:
            agent.gold = getattr(agent, "gold", 0) + effects["instant_gold"]

        # Apply XP bonuses
        if "instant_xp" in effects:
            if hasattr(agent, "add_xp"):
                agent.add_xp(effects["instant_xp"])
            else:
                agent.xp = getattr(agent, "xp", 0) + effects["instant_xp"]

        # Apply reroll bonuses
        if "instant_rerolls" in effects:
            agent.free_rerolls = getattr(agent, "free_rerolls", 0) + effects["instant_rerolls"]


# Register environment
try:
    gym.register(
        id="TFT-v0",
        entry_point="src.rl.env.tft_env:TFTEnv",
        max_episode_steps=1000,
    )
except Exception:
    pass  # Already registered or gym not available
