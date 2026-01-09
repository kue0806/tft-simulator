"""Game State for TFT Set 16.

Complete game state for simulation including all players.
"""

from typing import Optional, Any

from src.data.loaders import load_champions
from src.core.champion_pool import ChampionPool
from src.core.shop import Shop
from src.core.player_units import PlayerUnits, ChampionInstance
from src.core.constants import (
    REROLL_COST,
    LEVEL_XP,
    BASE_INCOME,
    MAX_INTEREST,
    INTEREST_PER_10_GOLD,
    WIN_GOLD,
    get_streak_bonus,
    calculate_interest,
    get_round_passive_gold,
)
from src.core.economy import EconomyCalculator, EconomyState, IncomeBreakdown
from src.core.stage_manager import StageManager, RoundType, RoundPhase, RoundInfo
from src.core.economy_advisor import EconomyAdvisor, EconomyAdvice
from src.core.pve_system import PvESystem, PvEResult, get_pve_system
from src.core.carousel import CarouselSystem, CarouselResult, get_carousel_system
from src.core.augment import AugmentSystem, AugmentChoice, Augment, get_augment_system
from src.core.augment_effects import AugmentEffectSystem, get_augment_effect_system
from src.core.item_manager import ItemManager, ItemInstance
from src.core.unlock_manager import UnlockManager, get_unlock_manager, reset_unlock_manager


class PlayerState:
    """Complete state for one player."""

    XP_PER_BUY = 4
    XP_COST = 4

    def __init__(self, player_id: int, pool: ChampionPool):
        """
        Initialize a player's state.

        Args:
            player_id: Unique identifier for this player.
            pool: The shared champion pool.
        """
        self.player_id = player_id
        self.pool = pool

        # Economy
        self.level = 1
        self.xp = 0
        self.gold = 0

        # Health
        self.health = 100
        self.is_alive = True

        # Shop
        self.shop = Shop(pool, self.level)

        # Units
        self.units = PlayerUnits()

        # Streaks
        self.win_streak = 0
        self.loss_streak = 0

        # Stats tracking
        self.rounds_played = 0
        self.total_damage_dealt = 0
        self.total_damage_taken = 0

        # Economy system
        self.economy = EconomyCalculator()
        self._economy_advisor: Optional[EconomyAdvisor] = None

        # Augments
        self.augments: list[Augment] = []
        self._augment_effects: AugmentEffectSystem = get_augment_effect_system()

        # Item manager
        self.items = ItemManager()

    def can_afford_reroll(self) -> bool:
        """Check if player can afford to reroll."""
        return self.gold >= REROLL_COST

    def reroll(self, unlock_manager: Optional["UnlockManager"] = None) -> bool:
        """
        Spend gold to refresh shop.

        Args:
            unlock_manager: Optional unlock manager to track rerolls.

        Returns:
            True if reroll was successful, False if not enough gold.
        """
        if not self.can_afford_reroll():
            return False

        self.gold -= REROLL_COST
        self.shop.refresh()

        # Track reroll for unlock conditions
        if unlock_manager:
            unlock_manager.record_reroll(self.player_id)

        return True

    def can_afford_xp(self) -> bool:
        """Check if player can afford to buy XP (gold or health with Cruel Pact)."""
        aug_state = self._augment_effects.get_player_augment_state(self.player_id)

        # Cruel Pact: XP costs health instead of gold
        if aug_state.get("xp_costs_health"):
            return self.health > 4  # Need more than 4 health to survive
        return self.gold >= self.XP_COST

    def buy_xp(self) -> bool:
        """
        Spend 4 gold (or 4 health with Cruel Pact) for 4 XP.

        Returns:
            True if purchase was successful.
        """
        if not self.can_afford_xp():
            return False

        aug_state = self._augment_effects.get_player_augment_state(self.player_id)

        # Cruel Pact: XP costs health instead of gold
        if aug_state.get("xp_costs_health"):
            health_cost = 4
            self.health -= health_cost
        else:
            self.gold -= self.XP_COST

        self.add_xp(self.XP_PER_BUY)
        return True

    def add_xp(self, amount: int) -> bool:
        """
        Add XP and check for level up.

        Args:
            amount: XP to add.

        Returns:
            True if leveled up.
        """
        self.xp += amount
        return self._check_level_up()

    def _check_level_up(self) -> bool:
        """
        Check and process level up.

        Returns:
            True if leveled up.
        """
        if self.level >= 10:
            return False

        xp_needed = LEVEL_XP.get(self.level + 1, float('inf'))
        if self.xp >= xp_needed:
            self.xp -= xp_needed
            self.level += 1
            self.shop.set_level(self.level)

            # Apply augment level up effects (Golden Ticket, etc.)
            self.apply_augment_level_up()

            # Recursively check for more level ups
            self._check_level_up()
            return True

        return False

    def can_afford_champion(self, cost: int) -> bool:
        """Check if player can afford a champion of given cost."""
        return self.gold >= cost

    def buy_champion(self, slot_index: int) -> Optional[ChampionInstance]:
        """
        Purchase a champion from shop.

        Args:
            slot_index: Index of the shop slot (0-4).

        Returns:
            The ChampionInstance if successful, None otherwise.
        """
        champion = self.shop.get_slot(slot_index)
        if champion is None:
            return None

        if not self.can_afford_champion(champion.cost):
            return None

        if not self.units.has_bench_space():
            return None

        # Purchase from shop
        purchased = self.shop.purchase(slot_index)
        if purchased is None:
            return None

        # Deduct gold
        self.gold -= purchased.cost

        # Add to bench (this handles auto-upgrade)
        instance = self.units.add_to_bench(purchased)
        return instance

    def sell_unit(
        self, instance: ChampionInstance, unlock_manager: Optional["UnlockManager"] = None
    ) -> int:
        """
        Sell a unit.

        Args:
            instance: The unit to sell.
            unlock_manager: Optional unlock manager to track sold units.

        Returns:
            Gold received.
        """
        gold = self.units.sell(instance)
        self.gold += gold

        # Return copies to pool
        copies = 1 if instance.star_level == 1 else (3 if instance.star_level == 2 else 9)
        self.pool.return_champion(instance.champion.id, copies)

        # Track sold unit for unlock conditions
        if unlock_manager:
            unlock_manager.record_unit_sold(
                self.player_id, instance.champion.id, instance.star_level
            )

        return gold

    def sell_bench_slot(self, slot_index: int) -> int:
        """
        Sell unit from a specific bench slot.

        Args:
            slot_index: Index of the bench slot.

        Returns:
            Gold received.
        """
        instance, gold = self.units.sell_from_bench(slot_index)
        if instance:
            self.gold += gold
            # Return copies to pool
            copies = 1 if instance.star_level == 1 else (3 if instance.star_level == 2 else 9)
            self.pool.return_champion(instance.champion.id, copies)
        return gold

    def calculate_income(self) -> int:
        """
        Calculate total income for end of round.

        Returns:
            Total gold income.
        """
        income = BASE_INCOME

        # Interest
        income += calculate_interest(self.gold)

        # Streak bonus
        streak = max(self.win_streak, self.loss_streak)
        income += get_streak_bonus(streak)

        return income

    def end_round_income(self) -> int:
        """
        Process end of round income.

        Returns:
            Gold received.
        """
        income = self.calculate_income()
        self.gold += income
        return income

    def record_win(self) -> None:
        """Record a combat win."""
        self.win_streak += 1
        self.loss_streak = 0
        self.rounds_played += 1

    def record_loss(self, damage: int) -> None:
        """
        Record a combat loss.

        Args:
            damage: Damage taken to health.
        """
        self.loss_streak += 1
        self.win_streak = 0
        self.rounds_played += 1
        self.take_damage(damage)

    def take_damage(
        self, damage: int, unlock_manager: Optional["UnlockManager"] = None
    ) -> None:
        """
        Take damage to health.

        Args:
            damage: Amount of damage.
            unlock_manager: Optional unlock manager to track HP lost.
        """
        self.health -= damage
        self.total_damage_taken += damage

        # Track HP lost for unlock conditions
        if unlock_manager:
            unlock_manager.record_hp_lost(self.player_id, damage)

        if self.health <= 0:
            self.health = 0
            self.is_alive = False

    def get_board_size_limit(self) -> int:
        """Get maximum units allowed on board based on level."""
        return self.level

    def can_place_on_board(self) -> bool:
        """Check if player can place more units on board."""
        return self.units.get_board_count() < self.get_board_size_limit()

    def get_active_traits(self) -> dict[str, int]:
        """
        Get all active traits from board units.

        Returns:
            Dictionary mapping trait_id to count.
        """
        traits: dict[str, int] = {}
        for unit in self.units.get_board_units():
            for trait in unit.champion.traits:
                traits[trait] = traits.get(trait, 0) + 1
        return traits

    def get_economy_state(self, stage: str = "1-1") -> EconomyState:
        """
        Get current economy state.

        Args:
            stage: Current stage string.

        Returns:
            EconomyState representing current economy.
        """
        return EconomyState(
            gold=self.gold,
            level=self.level,
            xp=self.xp,
            win_streak=self.win_streak,
            loss_streak=self.loss_streak,
            round_number=self.rounds_played,
            stage=stage,
        )

    def end_round(
        self, won: bool, stage: str = "1-1", pve_gold: int = 0
    ) -> IncomeBreakdown:
        """
        Process end of round economy.

        Args:
            won: Whether player won this round.
            stage: Current stage string.
            pve_gold: Gold from PvE drops.

        Returns:
            IncomeBreakdown with all income components.
        """
        # Update streaks
        if won:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0

        # Get economy state for calculation
        state = self.get_economy_state(stage)

        # Calculate income
        is_pve = stage.startswith("1-") or stage.endswith("-7")
        income = self.economy.calculate_round_income(state, won, is_pve, pve_gold)

        # Add income
        self.gold += income.total

        self.rounds_played += 1
        return income

    def get_economy_advice(self, stage: str = "1-1") -> EconomyAdvice:
        """
        Get economy advice for current state.

        Args:
            stage: Current stage string.

        Returns:
            EconomyAdvice with recommended action.
        """
        if self._economy_advisor is None:
            self._economy_advisor = EconomyAdvisor()

        return self._economy_advisor.get_advice(
            gold=self.gold,
            level=self.level,
            xp=self.xp,
            health=self.health,
            stage=stage,
            win_streak=self.win_streak,
            loss_streak=self.loss_streak,
        )

    def calculate_gold_to_level(self, target_level: int) -> int:
        """
        Calculate gold needed to reach target level.

        Args:
            target_level: Target level to reach.

        Returns:
            Gold needed.
        """
        return self.economy.calculate_gold_to_level(self.level, self.xp, target_level)

    # =========================================================================
    # AUGMENT METHODS
    # =========================================================================

    def add_augment(self, augment: Augment) -> "AugmentEffectResult":
        """
        Add an augment and apply its immediate effects.

        Args:
            augment: The augment to add.

        Returns:
            AugmentEffectResult with details of applied effects.
        """
        from src.core.augment_effects import AugmentEffectResult
        self.augments.append(augment)
        return self._augment_effects.apply_immediate_effects(
            augment.id, augment.effects, self
        )

    def apply_augment_round_start(self) -> "AugmentEffectResult":
        """Apply augment effects at round start."""
        return self._augment_effects.apply_round_start_effects(self)

    def apply_augment_level_up(self) -> "AugmentEffectResult":
        """Apply augment effects when leveling up."""
        return self._augment_effects.apply_level_up_effects(self)

    def get_augment_interest_modifier(self, base_interest: int) -> int:
        """Get modified interest based on augments."""
        return self._augment_effects.modify_interest(self, base_interest)

    def has_augment(self, augment_id: str) -> bool:
        """Check if player has a specific augment."""
        return any(a.id == augment_id for a in self.augments)

    def get_augment_effects(self, effect_key: str) -> list:
        """Get all augment effects with a specific key."""
        results = []
        for aug in self.augments:
            if effect_key in aug.effects:
                results.append(aug.effects[effect_key])
        return results

    def __repr__(self) -> str:
        return (
            f"Player{self.player_id}(lvl={self.level}, hp={self.health}, "
            f"gold={self.gold}, units={self.units.get_total_units()})"
        )


class GameState:
    """
    Complete game state for simulation.
    """

    def __init__(self, num_players: int = 8, seed: Optional[int] = None):
        """
        Initialize a new game.

        Args:
            num_players: Number of players (default 8).
            seed: Random seed for reproducible games.
        """
        self.num_players = num_players
        self.seed = seed

        # Stage tracking using StageManager
        self.stage_manager = StageManager()

        # Legacy attributes for backward compatibility
        self.stage = 1
        self.round = 1
        self.total_rounds = 0

        # Initialize shared pool with all base champions
        all_champions = load_champions()
        self.pool = ChampionPool(all_champions, num_players)

        # Initialize unlock manager
        reset_unlock_manager()  # Reset for new game
        self.unlock_manager = get_unlock_manager(self.pool)

        # Initialize players
        self.players = [PlayerState(i, self.pool) for i in range(num_players)]
        self.eliminated: list[int] = []

        # Initialize subsystems
        self.pve_system = get_pve_system(seed)
        self.carousel_system = get_carousel_system(seed)
        self.augment_system = get_augment_system(seed)

        # Current round state
        self.current_carousel: list = []  # Current carousel champions
        self.current_augment_choices: dict[int, AugmentChoice] = {}  # Player ID -> choices

        # Starting gold is 0 - players earn gold from rounds
        for player in self.players:
            player.gold = 0

    def get_stage_string(self) -> str:
        """Get current stage as string (e.g., '2-3')."""
        return self.stage_manager.get_stage_string()

    def get_current_round_info(self):
        """Get information about the current round."""
        return self.stage_manager.get_current_round_info()

    def is_pve_round(self) -> bool:
        """Check if current round is PvE."""
        return self.stage_manager.is_pve_round()

    def is_rolldown_timing(self) -> bool:
        """Check if current round is a common rolldown timing."""
        return self.stage_manager.is_rolldown_timing()

    def is_level_timing(self) -> bool:
        """Check if current round is a standard leveling timing."""
        return self.stage_manager.is_level_timing()

    def get_player(self, player_id: int) -> Optional[PlayerState]:
        """
        Get a player by ID.

        Args:
            player_id: The player's ID.

        Returns:
            PlayerState if found and alive, None otherwise.
        """
        if 0 <= player_id < len(self.players):
            player = self.players[player_id]
            if player.is_alive:
                return player
        return None

    def get_alive_players(self) -> list[PlayerState]:
        """Get all players still alive."""
        return [p for p in self.players if p.is_alive]

    def get_eliminated_players(self) -> list[PlayerState]:
        """Get all eliminated players."""
        return [self.players[i] for i in self.eliminated]

    def eliminate_player(self, player_id: int) -> None:
        """
        Handle player elimination.
        Returns all their champions to pool.

        Args:
            player_id: ID of the player to eliminate.
        """
        if player_id in self.eliminated:
            return

        player = self.players[player_id]
        player.is_alive = False
        self.eliminated.append(player_id)

        # Return all units to pool
        instances = player.units.clear()
        for instance in instances:
            copies = 1 if instance.star_level == 1 else (3 if instance.star_level == 2 else 9)
            self.pool.return_champion(instance.champion.id, copies)

        # Return shop units to pool
        player.shop.clear()

    def advance_round(self) -> None:
        """
        Process end of round and advance to next round.
        """
        self.total_rounds += 1

        # Get current round info before advancing
        current_stage = self.get_stage_string()
        round_info = self.get_current_round_info()

        # Give income to all alive players (not for carousel rounds)
        # Carousel rounds don't give income, only PvE/PvP rounds do
        if not round_info.is_carousel:
            for player in self.get_alive_players():
                player.end_round_income()
                # Add passive XP (2 per round after stage 1)
                if round_info.passive_xp > 0:
                    player.add_xp(round_info.passive_xp)

        # Advance stage/round counter using StageManager
        self.stage_manager.advance_round()

        # Keep legacy attributes in sync
        self.stage = self.stage_manager.current_stage
        self.round = self.stage_manager.current_round

        # Update stage for all players' unlock tracking
        new_stage = self.get_stage_string()
        for player in self.get_alive_players():
            self.unlock_manager.update_stage(player.player_id, new_stage)

    def start_planning_phase(self) -> None:
        """
        Start the planning phase - refresh shops for all players.
        """
        for player in self.get_alive_players():
            if not player.shop.locked:
                player.shop.refresh()

    def is_game_over(self) -> bool:
        """Check if game is over (1 or 0 players remaining)."""
        return len(self.get_alive_players()) <= 1

    def get_winner(self) -> Optional[PlayerState]:
        """Get the winner if game is over."""
        alive = self.get_alive_players()
        if len(alive) == 1:
            return alive[0]
        return None

    def get_placements(self) -> list[PlayerState]:
        """
        Get players in placement order (1st to 8th).

        Returns:
            List of players, winner first.
        """
        # Alive players first (sorted by health), then eliminated in reverse order
        alive = sorted(self.get_alive_players(), key=lambda p: -p.health)
        dead = [self.players[i] for i in reversed(self.eliminated)]
        return alive + dead

    # =========================================================================
    # CAROUSEL METHODS
    # =========================================================================

    def is_carousel_round(self) -> bool:
        """Check if current round is a carousel round."""
        return self.stage_manager.is_carousel_round()

    def start_carousel(self) -> list:
        """
        Start a carousel round.

        Returns:
            List of CarouselChampion objects on the carousel.
        """
        stage = self.get_stage_string()
        self.current_carousel = self.carousel_system.generate_carousel(stage, self.pool)
        return self.current_carousel

    def get_carousel_pick_order(self) -> list:
        """
        Get the order in which players pick from carousel.

        Returns:
            List of CarouselPickOrder sorted by pick order.
        """
        stage = self.get_stage_string()
        player_healths = {
            p.player_id: p.health for p in self.get_alive_players()
        }
        return self.carousel_system.calculate_pick_order(player_healths, stage)

    def process_carousel_pick(
        self,
        player_id: int,
        pick_index: int,
    ) -> Optional[Any]:
        """
        Process a player's carousel pick.

        Args:
            player_id: Player making the pick.
            pick_index: Index of champion to pick.

        Returns:
            The picked CarouselChampion, or None if invalid.
        """
        picked = self.carousel_system.process_pick(
            self.current_carousel, player_id, pick_index
        )
        if picked:
            player = self.get_player(player_id)
            if player:
                # Add item to player's inventory
                item = player.items.get_item(picked.item_id)
                if item:
                    player.items.add_to_inventory(item)

                # Add champion to bench (add_to_bench takes Champion, not ChampionInstance)
                champion = self.pool.get_champion(picked.champion_id)
                if champion:
                    player.units.add_to_bench(champion)
        return picked

    # =========================================================================
    # AUGMENT METHODS
    # =========================================================================

    def is_augment_round(self) -> bool:
        """Check if current round has augment selection."""
        return self.stage_manager.is_augment_round()

    def start_augment_selection(self) -> dict[int, AugmentChoice]:
        """
        Start augment selection for all players.

        Returns:
            Dictionary mapping player_id to their AugmentChoice.
        """
        stage = self.get_stage_string()
        self.current_augment_choices.clear()

        for player in self.get_alive_players():
            excluded = [a.id for a in player.augments]
            choice = self.augment_system.generate_augment_choices(
                stage, player.player_id, excluded
            )
            self.current_augment_choices[player.player_id] = choice

        return self.current_augment_choices

    def select_augment(
        self,
        player_id: int,
        augment_index: int,
    ) -> Optional[Augment]:
        """
        Process a player's augment selection.

        Args:
            player_id: Player making the selection.
            augment_index: Index of augment to select (0-2).

        Returns:
            The selected Augment, or None if invalid.
        """
        if player_id not in self.current_augment_choices:
            return None

        choice = self.current_augment_choices[player_id]
        selected = self.augment_system.select_augment(choice, player_id, augment_index)

        if selected:
            player = self.get_player(player_id)
            if player:
                player.augments.append(selected)
                # Apply instant effects
                self._apply_augment_effects(player, selected)

        return selected

    def _apply_augment_effects(self, player: PlayerState, augment: Augment) -> None:
        """Apply an augment's immediate effects to a player."""
        effects = augment.effects

        if "instant_gold" in effects:
            player.gold += effects["instant_gold"]

        if "instant_xp" in effects:
            player.add_xp(effects["instant_xp"])

        if "random_components" in effects:
            # Simplified: would generate random components
            pass

    # =========================================================================
    # PVE METHODS
    # =========================================================================

    def process_pve_round(self, player_id: int) -> PvEResult:
        """
        Process a PvE combat for a player.

        Args:
            player_id: Player fighting PvE.

        Returns:
            PvEResult with combat outcome and loot.
        """
        stage = self.get_stage_string()
        player = self.get_player(player_id)

        if not player:
            raise ValueError(f"Player {player_id} not found or eliminated")

        # Simplified power calculation
        power = len(player.units.get_board_units()) * 20 + player.level * 10

        result = self.pve_system.simulate_pve_combat(stage, player_id, power)

        # Apply results
        if result.won:
            player.gold += result.gold_gained
            for item_id in result.items_gained:
                item = player.items.get_item(item_id)
                if item:
                    player.items.add_to_inventory(item)
        else:
            player.take_damage(result.damage_taken)

        return result

    # =========================================================================
    # ROUND FLOW METHODS
    # =========================================================================

    def process_round_income(self) -> dict[int, int]:
        """
        Process passive gold income for the round.

        Returns:
            Dictionary mapping player_id to gold received.
        """
        round_info = self.get_current_round_info()
        income_map = {}

        for player in self.get_alive_players():
            # Passive gold based on round
            income = round_info.passive_gold

            # Interest
            income += calculate_interest(player.gold)

            # Streak bonus
            streak = max(player.win_streak, player.loss_streak)
            income += get_streak_bonus(streak)

            player.gold += income
            income_map[player.player_id] = income

        return income_map

    def process_round_xp(self) -> dict[int, int]:
        """
        Process passive XP for the round.

        Returns:
            Dictionary mapping player_id to XP received.
        """
        round_info = self.get_current_round_info()
        xp_map = {}

        for player in self.get_alive_players():
            if round_info.passive_xp > 0:
                player.add_xp(round_info.passive_xp)
                xp_map[player.player_id] = round_info.passive_xp

        return xp_map

    def get_current_phase(self) -> Optional[RoundPhase]:
        """Get the current phase of the round."""
        return self.stage_manager.get_current_phase()

    def advance_phase(self) -> Optional[RoundPhase]:
        """Advance to the next phase within the round."""
        return self.stage_manager.advance_phase()

    def is_round_complete(self) -> bool:
        """Check if all phases in the current round are complete."""
        return self.stage_manager.is_round_complete()

    # =========================================================================
    # UNLOCK SYSTEM METHODS
    # =========================================================================

    def process_combat_result(
        self,
        player_id: int,
        won: bool,
        active_traits: Optional[list[str]] = None,
    ) -> list:
        """
        Process combat result and check for unlocks.

        Args:
            player_id: The player who fought.
            won: Whether the player won.
            active_traits: List of active trait IDs during combat.

        Returns:
            List of newly unlocked Champion objects.
        """
        player = self.get_player(player_id)
        if not player:
            return []

        # Get units on board for streak tracking
        board_units = [u.champion.id for u in player.units.get_board_units()]

        # Record combat result for streak tracking
        self.unlock_manager.record_combat_result(player_id, won, board_units)

        # Record active traits
        if active_traits:
            for trait_id in active_traits:
                self.unlock_manager.record_trait_combat(player_id, trait_id)

        # Check for newly unlocked champions
        newly_unlocked = self.unlock_manager.check_unlocks(player)

        # Set pending unlock for shop guarantee
        for champion in newly_unlocked:
            player.shop.set_pending_unlock(champion.id)

        return newly_unlocked

    def check_unlocks(self, player_id: int) -> list:
        """
        Check unlock conditions for a player.

        Args:
            player_id: The player to check.

        Returns:
            List of newly unlocked Champion objects.
        """
        player = self.get_player(player_id)
        if not player:
            return []

        return self.unlock_manager.check_unlocks(player)

    def get_unlock_progress(self, player_id: int, champion_id: str) -> dict:
        """
        Get progress toward unlocking a champion.

        Args:
            player_id: The player to check.
            champion_id: The champion to check progress for.

        Returns:
            Dictionary with unlock progress info.
        """
        player = self.get_player(player_id)
        if not player:
            return {"error": "Player not found"}

        return self.unlock_manager.get_unlock_progress(player, champion_id)

    def get_all_unlockables(self) -> list:
        """Get all unlockable champions."""
        return self.unlock_manager.get_all_unlockables()

    def is_champion_unlocked(self, player_id: int, champion_id: str) -> bool:
        """Check if a champion is unlocked for a player."""
        return self.unlock_manager.is_unlocked(player_id, champion_id)

    def __repr__(self) -> str:
        alive = len(self.get_alive_players())
        return f"GameState(stage={self.get_stage_string()}, alive={alive}/{self.num_players})"
