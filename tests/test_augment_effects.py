"""Tests for Augment Effects System."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.core.augment_effects import (
    AugmentEffectSystem,
    AugmentEffectResult,
    AugmentEffectTrigger,
)


@dataclass
class MockAugment:
    """Mock augment for testing."""
    id: str
    name: str
    tier: int
    effects: dict


def create_mock_player(player_id: int = 1, gold: int = 50, health: int = 100, level: int = 5):
    """Create a mock player state."""
    player = MagicMock()
    player.player_id = player_id
    player.gold = gold
    player.health = health
    player.level = level
    player.augments = []

    # Mock XP methods
    player.add_xp = MagicMock()

    # Mock units
    player.units = MagicMock()
    player.units.get_active_synergies = MagicMock(return_value={})
    player.units.add_to_bench = MagicMock(return_value=MagicMock())

    # Mock items
    player.items = MagicMock()
    player.items.add_to_inventory = MagicMock()

    return player


class TestAugmentEffectSystem:
    """Test augment effect system initialization."""

    def test_init(self):
        """System should initialize correctly."""
        system = AugmentEffectSystem()
        assert system is not None
        assert system._player_states == {}

    def test_init_with_seed(self):
        """System should accept random seed."""
        system = AugmentEffectSystem(seed=42)
        assert system is not None


class TestPlayerAugmentState:
    """Test player augment state tracking."""

    def test_get_player_augment_state_creates_new(self):
        """Should create new state for new player."""
        system = AugmentEffectSystem()
        state = system.get_player_augment_state(1)

        assert state is not None
        assert 1 in system._player_states

    def test_get_player_augment_state_returns_existing(self):
        """Should return existing state for known player."""
        system = AugmentEffectSystem()
        state1 = system.get_player_augment_state(1)
        state1["test_value"] = 42
        state2 = system.get_player_augment_state(1)

        assert state2["test_value"] == 42


class TestImmediateEffects:
    """Test immediate augment effects (on selection)."""

    def test_gold_grant(self):
        """Should grant gold immediately."""
        system = AugmentEffectSystem()
        player = create_mock_player(gold=50)

        result = system.apply_immediate_effects(
            augment_id="test_gold",
            effects={"instant_gold": 10},
            player=player
        )

        assert result.success is True
        assert result.gold_gained == 10
        assert player.gold == 60

    def test_xp_grant(self):
        """Should grant XP immediately."""
        system = AugmentEffectSystem()
        player = create_mock_player()

        result = system.apply_immediate_effects(
            augment_id="test_xp",
            effects={"instant_xp": 20},
            player=player
        )

        assert result.success is True
        assert result.xp_gained == 20
        player.add_xp.assert_called_once_with(20)

    def test_interest_cap_increase(self):
        """Should increase interest cap."""
        system = AugmentEffectSystem()
        player = create_mock_player()

        result = system.apply_immediate_effects(
            augment_id="test_cap",
            effects={"interest_cap_increase": 2},
            player=player
        )

        assert result.success is True
        state = system.get_player_augment_state(player.player_id)
        assert state["interest_cap_increase"] == 2


class TestRoundStartEffects:
    """Test round start augment effects."""

    def test_gold_per_round(self):
        """Should grant gold at round start."""
        system = AugmentEffectSystem()
        player = create_mock_player(gold=50)

        # Set gold_per_round in state
        state = system.get_player_augment_state(player.player_id)
        state["gold_per_round"] = 3

        player.shop = MagicMock()
        player.augments = []

        result = system.apply_round_start_effects(player)

        assert result.success is True
        assert result.gold_gained == 3
        assert player.gold == 53

    def test_heal_per_round(self):
        """Should heal at round start."""
        system = AugmentEffectSystem()
        player = create_mock_player(health=80)
        augment = MockAugment(
            id="tiny_titans",
            name="Tiny Titans",
            tier=1,
            effects={"heal_per_round": 5}
        )
        player.augments = [augment]
        player.shop = MagicMock()

        result = system.apply_round_start_effects(player)

        assert result.success is True
        assert player.health == 85


class TestRoundEndEffects:
    """Test round end augment effects."""

    def test_interest_modifier(self):
        """Should modify interest correctly."""
        system = AugmentEffectSystem()
        player = create_mock_player()

        # Normal interest
        mult, cap = system.get_interest_modifier(player)
        assert mult == 1.0
        assert cap == 0

    def test_no_interest(self):
        """Should disable interest when augment says so."""
        system = AugmentEffectSystem()
        player = create_mock_player()

        # Set no interest state
        state = system.get_player_augment_state(player.player_id)
        state["no_interest"] = True

        mult, cap = system.get_interest_modifier(player)
        assert mult == 0.0

    def test_interest_cap_increase(self):
        """Should increase interest cap."""
        system = AugmentEffectSystem()
        player = create_mock_player()

        state = system.get_player_augment_state(player.player_id)
        state["interest_cap_increase"] = 2

        mult, cap = system.get_interest_modifier(player)
        assert cap == 2


class TestRerollEffects:
    """Test reroll-related augment effects."""

    def test_free_reroll(self):
        """Should allow free reroll when pending."""
        system = AugmentEffectSystem()
        player = create_mock_player()

        state = system.get_player_augment_state(player.player_id)
        state["pending_rerolls"] = 1

        cost = system.get_reroll_cost(player)
        assert cost == 0

    def test_free_reroll_chance(self):
        """Should have chance for free reroll."""
        system = AugmentEffectSystem(seed=42)  # Fixed seed for reproducibility
        player = create_mock_player()

        state = system.get_player_augment_state(player.player_id)
        state["free_reroll_chance"] = 1.0  # 100% chance

        cost = system.get_reroll_cost(player)
        assert cost == 0

    def test_base_reroll_cost(self):
        """Should return base cost normally."""
        system = AugmentEffectSystem()
        player = create_mock_player()

        cost = system.get_reroll_cost(player)
        assert cost == 2  # Base cost


class TestPurchaseEffects:
    """Test champion purchase augment effects."""

    def test_reroll_on_2_cost_purchase(self):
        """Should grant reroll when buying 2-cost."""
        system = AugmentEffectSystem()
        player = create_mock_player()
        augment = MockAugment(
            id="two_much_value",
            name="Two Much Value",
            tier=1,
            effects={"rerolls_per_2cost": 1}
        )
        player.augments = [augment]

        result = system.apply_purchase_effects(player, champion_cost=2)

        state = system.get_player_augment_state(player.player_id)
        assert state.get("pending_rerolls", 0) == 1

    def test_no_reroll_on_other_cost(self):
        """Should not grant reroll for non-2-cost."""
        system = AugmentEffectSystem()
        player = create_mock_player()
        augment = MockAugment(
            id="two_much_value",
            name="Two Much Value",
            tier=1,
            effects={"rerolls_per_2cost": 1}
        )
        player.augments = [augment]

        system.apply_purchase_effects(player, champion_cost=3)

        state = system.get_player_augment_state(player.player_id)
        assert state.get("pending_rerolls", 0) == 0


class TestCombatStatModifiers:
    """Test combat stat modifiers from augments."""

    def test_item_holder_bonus(self):
        """Should give bonus to units with items."""
        system = AugmentEffectSystem()
        player = create_mock_player()
        augment = MockAugment(
            id="item_holder",
            name="Item Holder",
            tier=1,
            effects={"item_holder_ad": 15, "item_holder_ap": 15}
        )
        player.augments = [augment]

        # Mock unit with items
        unit = MagicMock()
        unit.items = [MagicMock()]  # Has 1 item
        unit.stats = MagicMock()
        unit.stats.crit_chance = 0.25

        modifiers = system.get_combat_stat_modifiers(player, unit)

        assert modifiers["bonus_ad"] == 15
        assert modifiers["bonus_ap"] == 15

    def test_three_item_holder_bonus(self):
        """Should give bonus to units with 3 items."""
        system = AugmentEffectSystem()
        player = create_mock_player()
        augment = MockAugment(
            id="cybernetic",
            name="Cybernetic Implants",
            tier=2,
            effects={"three_item_holder_health": 300, "three_item_holder_ad": 30}
        )
        player.augments = [augment]

        # Mock unit with 3 items
        unit = MagicMock()
        unit.items = [MagicMock(), MagicMock(), MagicMock()]
        unit.stats = MagicMock()
        unit.stats.crit_chance = 0.25

        modifiers = system.get_combat_stat_modifiers(player, unit)

        assert modifiers["bonus_health"] == 300
        assert modifiers["bonus_ad"] == 30

    def test_damage_amp(self):
        """Should apply damage amplification."""
        system = AugmentEffectSystem()
        player = create_mock_player()
        augment = MockAugment(
            id="damage_amp",
            name="Damage Amp",
            tier=1,
            effects={"damage_amp_all": 0.10}
        )
        player.augments = [augment]

        unit = MagicMock()
        unit.items = []
        unit.stats = MagicMock()
        unit.stats.crit_chance = 0.25

        modifiers = system.get_combat_stat_modifiers(player, unit)

        assert modifiers["damage_amp"] == 0.10


class TestLevelUpEffects:
    """Test level up augment effects."""

    def test_gold_on_level_up(self):
        """Should grant gold on level up."""
        system = AugmentEffectSystem()
        player = create_mock_player(gold=50)
        augment = MockAugment(
            id="level_gold",
            name="Level Gold",
            tier=1,
            effects={"gold_per_level_up": 5}  # Correct key
        )
        player.augments = [augment]

        result = system.apply_level_up_effects(player)  # No new_level param

        assert result.success is True
        assert result.gold_gained == 5
        assert player.gold == 55


class TestCombatEffects:
    """Test combat-related augment effects."""

    def test_on_attack_effects_returns_modifiers(self):
        """Should return attack modifiers dict."""
        system = AugmentEffectSystem()
        player = create_mock_player()
        augment = MockAugment(
            id="fire_axiom",
            name="Fire Axiom",
            tier=1,
            effects={"apply_burn": True, "apply_wound": True}
        )
        player.augments = [augment]

        attacker = MagicMock()
        target = MagicMock()
        engine = MagicMock()

        mods = system.apply_on_attack_effects(player, attacker, target, engine)

        assert mods["apply_burn"] is True
        assert mods["apply_wound"] is True


class TestTraitAugmentSynergy:
    """Test synergy between traits and augments."""

    def test_bronze_trait_bonus(self):
        """Should give bonus per bronze trait."""
        system = AugmentEffectSystem()
        player = create_mock_player()

        # Mock 3 bronze traits
        mock_trait = MagicMock()
        mock_trait.style = "bronze"
        mock_trait.is_active = True
        player.units.get_active_synergies.return_value = {
            "trait1": mock_trait,
            "trait2": mock_trait,
            "trait3": mock_trait,
        }

        augment = MockAugment(
            id="bronze_for_life",
            name="Bronze For Life",
            tier=2,
            effects={"damage_amp_per_bronze": 0.05}
        )
        player.augments = [augment]

        bonuses = system.apply_trait_augment_synergy(player, "any")

        assert bonuses.get("damage_amp", 0) == pytest.approx(0.15)  # 3 * 0.05


class TestClearState:
    """Test state clearing."""

    def test_reset_player(self):
        """Should reset a player's state."""
        system = AugmentEffectSystem()
        state = system.get_player_augment_state(1)
        state["test"] = 42

        system.reset_player(1)

        assert 1 not in system._player_states

    def test_reset_all_states(self):
        """Should reset all player states."""
        system = AugmentEffectSystem()
        system.get_player_augment_state(1)
        system.get_player_augment_state(2)

        system.reset()

        assert system._player_states == {}
