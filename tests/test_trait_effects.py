"""Tests for Trait Effects System."""

import pytest
from unittest.mock import MagicMock, patch

from src.combat.hex_grid import Team, HexGrid, HexPosition
from src.combat.combat_unit import CombatUnit, CombatStats
from src.combat.trait_effects import TraitEffectSystem, TraitEffectContext
from src.combat.status_effects import StatusEffectSystem, StatusEffectType


def create_test_unit(
    unit_id: str,
    team: Team = Team.BLUE,
    hp: float = 1000,
    traits: list = None,
) -> CombatUnit:
    """Create a test combat unit with specified traits."""
    stats = CombatStats(
        max_hp=hp,
        current_hp=hp,
        attack_damage=100,
        ability_power=100,
        armor=50,
        magic_resist=50,
        attack_speed=1.0,
        crit_chance=0.25,
        crit_damage=1.4,
        max_mana=100,
        current_mana=50,
        starting_mana=50,
        attack_range=1,
        dodge_chance=0.0,
        omnivamp=0.0,
        damage_amp=1.0,
        damage_reduction=0.0,
    )
    unit = CombatUnit(
        id=unit_id,
        name=f"Unit_{unit_id}",
        champion_id=f"TFT_Test_{unit_id}",
        star_level=1,
        team=team,
        stats=stats,
    )

    # Mock source_instance with traits
    if traits:
        mock_champion = MagicMock()
        mock_champion.traits = traits
        mock_instance = MagicMock()
        mock_instance.champion = mock_champion
        unit.source_instance = mock_instance

    return unit


class TestTraitEffectSystem:
    """TraitEffectSystem tests."""

    @pytest.fixture
    def system(self):
        """Create trait effect system."""
        return TraitEffectSystem()

    @pytest.fixture
    def grid(self):
        """Create hex grid."""
        return HexGrid()

    @pytest.fixture
    def status_effects(self):
        """Create status effect system."""
        return StatusEffectSystem()

    @pytest.fixture
    def context(self, status_effects, grid):
        """Create trait effect context."""
        return TraitEffectContext(
            status_effects=status_effects,
            all_units={},
            grid=grid,
        )

    def test_system_init(self, system):
        """Test system initialization."""
        assert system is not None
        assert "blue" in system.team_handlers
        assert "red" in system.team_handlers


class TestLongshotTrait:
    """Tests for Longshot trait distance-based damage."""

    @pytest.fixture
    def system(self):
        """Create trait effect system."""
        return TraitEffectSystem()

    @pytest.fixture
    def grid(self):
        """Create hex grid."""
        return HexGrid()

    @pytest.fixture
    def status_effects(self):
        """Create status effect system."""
        return StatusEffectSystem()

    @pytest.fixture
    def context(self, status_effects, grid):
        """Create trait effect context."""
        return TraitEffectContext(
            status_effects=status_effects,
            all_units={},
            grid=grid,
        )

    def test_longshot_damage_increases_with_distance(self, system, grid, context):
        """Longshot should deal more damage at longer distances."""
        attacker = create_test_unit("attacker", Team.BLUE, traits=["longshot"])
        target = create_test_unit("target", Team.RED)

        # Place units at different distances
        grid.place_unit("attacker", HexPosition(0, 0))
        grid.place_unit("target", HexPosition(3, 0))  # 3 hexes away

        # Setup longshot handler
        from src.core.unique_traits import LongshotHandler
        system.team_handlers["blue"]["longshot"] = LongshotHandler()

        # Get damage modifier
        modifier = system.get_damage_modifier(attacker, target, context)

        # Should have bonus damage (base + distance bonus)
        # At 2 Longshot: 10% base + 3 hexes * 8% = 34% bonus
        assert modifier > 1.0

    def test_longshot_damage_caps_at_5_hexes(self, system, grid, context):
        """Longshot distance bonus should cap at 5 hexes."""
        attacker = create_test_unit("attacker", Team.BLUE, traits=["longshot"])
        target = create_test_unit("target", Team.RED)

        # Place units far apart
        grid.place_unit("attacker", HexPosition(0, 0))
        grid.place_unit("target", HexPosition(6, 0))  # 6 hexes away

        from src.core.unique_traits import LongshotHandler
        system.team_handlers["blue"]["longshot"] = LongshotHandler()

        modifier_far = system.get_damage_modifier(attacker, target, context)

        # Move target to exactly 5 hexes
        grid.remove_unit("target")
        grid.place_unit("target", HexPosition(5, 0))

        modifier_5hex = system.get_damage_modifier(attacker, target, context)

        # Should be the same (capped at 5)
        assert abs(modifier_far - modifier_5hex) < 0.01


class TestDisruptorTrait:
    """Tests for Disruptor Dazzle effect."""

    @pytest.fixture
    def system(self):
        """Create trait effect system."""
        return TraitEffectSystem()

    @pytest.fixture
    def status_effects(self):
        """Create status effect system."""
        return StatusEffectSystem()

    @pytest.fixture
    def context(self, status_effects):
        """Create trait effect context."""
        return TraitEffectContext(
            status_effects=status_effects,
            all_units={},
        )

    def test_disruptor_applies_dazzle_on_ability(self, system, status_effects, context):
        """Disruptor should apply Dazzle (DISARM) when casting abilities."""
        caster = create_test_unit("caster", Team.BLUE, traits=["disruptor"])
        target = create_test_unit("target", Team.RED)

        from src.core.unique_traits import DisruptorHandler
        system.team_handlers["blue"]["disruptor"] = DisruptorHandler()

        # Apply ability cast effects
        events = system.apply_on_ability_cast(caster, [target], context)

        # Target should be dazzled (DISARMed)
        assert status_effects.has_effect(target, StatusEffectType.DISARM)
        assert len(events) > 0
        assert events[0]["type"] == "trait_disruptor_dazzle"

    def test_disruptor_damage_bonus_to_dazzled(self, system, status_effects, context):
        """Disruptor should deal bonus damage to dazzled targets."""
        # Attacker needs disruptor trait for bonus damage
        attacker = create_test_unit("attacker", Team.BLUE, traits=["disruptor"])
        target = create_test_unit("target", Team.RED)

        from src.core.unique_traits import DisruptorHandler
        handler = DisruptorHandler()
        # Mock get_bonus to return values (simulating 2+ disruptors)
        handler.get_bonus = MagicMock(return_value={"dazzle_duration": 1.5, "dazzle_damage": 25})
        system.team_handlers["blue"]["disruptor"] = handler

        # Apply dazzle to target
        from src.combat.status_effects import StatusEffect
        dazzle = StatusEffect(
            effect_type=StatusEffectType.DISARM,
            source_id="test",
            duration=2.0,
        )
        status_effects.apply_effect(target, dazzle)

        # Get damage modifier
        modifier = system.get_damage_modifier(attacker, target, context)

        # Should have bonus damage (25% at 2 Disruptor)
        assert modifier > 1.0


class TestHuntressTrait:
    """Tests for Huntress leap mechanics."""

    @pytest.fixture
    def system(self):
        """Create trait effect system."""
        return TraitEffectSystem()

    @pytest.fixture
    def context(self):
        """Create trait effect context."""
        return TraitEffectContext(
            status_effects=MagicMock(),
            all_units={},
        )

    def test_huntress_leap_targets_lowest_hp(self, system, context):
        """Huntress should leap to lowest HP enemy."""
        huntress = create_test_unit("huntress", Team.BLUE, traits=["huntress"])

        enemy1 = create_test_unit("enemy1", Team.RED, hp=1000)
        enemy2 = create_test_unit("enemy2", Team.RED, hp=500)  # Lowest HP
        enemy3 = create_test_unit("enemy3", Team.RED, hp=800)

        from src.core.unique_traits import HuntressHandler
        system.team_handlers["blue"]["huntress"] = HuntressHandler()

        target = system.get_huntress_leap_target(
            huntress,
            [enemy1, enemy2, enemy3],
            context
        )

        assert target is not None
        assert target.id == "enemy2"

    def test_should_huntress_leap(self, system):
        """Test huntress leap check."""
        huntress = create_test_unit("huntress", Team.BLUE, traits=["huntress"])
        non_huntress = create_test_unit("non_huntress", Team.BLUE, traits=["bruiser"])

        from src.core.unique_traits import HuntressHandler
        system.team_handlers["blue"]["huntress"] = HuntressHandler()

        assert system.should_huntress_leap(huntress) is True
        assert system.should_huntress_leap(non_huntress) is False


class TestSlayerTrait:
    """Tests for Slayer missing HP damage bonus."""

    @pytest.fixture
    def system(self):
        """Create trait effect system."""
        return TraitEffectSystem()

    def test_slayer_damage_scales_with_missing_hp(self, system):
        """Slayer should deal more damage when low on health."""
        attacker = create_test_unit("attacker", Team.BLUE, hp=1000, traits=["slayer"])
        target = create_test_unit("target", Team.RED)

        from src.core.unique_traits import SlayerHandler
        system.team_handlers["blue"]["slayer"] = SlayerHandler()

        # Full HP - no bonus
        modifier_full = system.get_damage_modifier(attacker, target)
        assert modifier_full == 1.0

        # Half HP - 25% bonus (50% missing * 0.5)
        attacker.stats.current_hp = 500
        modifier_half = system.get_damage_modifier(attacker, target)
        assert modifier_half > modifier_full

        # Very low HP - more bonus
        attacker.stats.current_hp = 100
        modifier_low = system.get_damage_modifier(attacker, target)
        assert modifier_low > modifier_half


class TestCombatStartEffects:
    """Tests for trait effects at combat start."""

    @pytest.fixture
    def system(self):
        """Create trait effect system."""
        return TraitEffectSystem()

    @pytest.fixture
    def status_effects(self):
        """Create status effect system."""
        return StatusEffectSystem()

    @pytest.fixture
    def context(self, status_effects):
        """Create trait effect context."""
        return TraitEffectContext(
            status_effects=status_effects,
            all_units={},
        )

    def test_quickstriker_as_buff_at_start(self, system, status_effects, context):
        """Quickstriker should get AS buff at combat start."""
        unit = create_test_unit("quickstriker", Team.BLUE, traits=["quickstriker"])

        from src.core.unique_traits import QuickstrikerHandler
        handler = QuickstrikerHandler()
        # Mock get_bonus to return values (simulating 2+ quickstrikers)
        handler.get_bonus = MagicMock(return_value={"attack_speed": 30, "duration": 5})
        system.team_handlers["blue"]["quickstriker"] = handler

        events = system.apply_combat_start_effects("blue", [unit], context)

        # Should have AS buff applied
        assert status_effects.has_effect(unit, StatusEffectType.ATTACK_SPEED_BUFF)

    def test_bruiser_hp_bonus_at_start(self, system, context):
        """Bruiser should get HP bonus at combat start."""
        unit = create_test_unit("bruiser", Team.BLUE, hp=1000, traits=["bruiser"])
        initial_hp = unit.stats.max_hp

        from src.core.unique_traits import BruiserHandler
        handler = BruiserHandler()
        # Mock get_bonus to return values (simulating 2+ bruisers)
        handler.get_bonus = MagicMock(return_value={"hp_percent": 15})
        system.team_handlers["blue"]["bruiser"] = handler

        events = system.apply_combat_start_effects("blue", [unit], context)

        # HP should have increased
        assert unit.stats.max_hp > initial_hp

    def test_arcanist_ap_bonus_at_start(self, system, context):
        """Arcanist should get AP bonus at combat start."""
        unit = create_test_unit("arcanist", Team.BLUE, traits=["arcanist"])
        initial_ap = unit.stats.ability_power

        from src.core.unique_traits import ArcanistHandler
        handler = ArcanistHandler()
        # Mock get_bonus to return values (simulating 2+ arcanists)
        handler.get_bonus = MagicMock(return_value={"ability_power": 20})
        system.team_handlers["blue"]["arcanist"] = handler

        events = system.apply_combat_start_effects("blue", [unit], context)

        # AP should have increased
        assert unit.stats.ability_power > initial_ap

    def test_defender_defense_bonus_at_start(self, system, context):
        """Defender should get armor/MR bonus at combat start."""
        unit = create_test_unit("defender", Team.BLUE, traits=["defender"])
        initial_armor = unit.stats.armor
        initial_mr = unit.stats.magic_resist

        from src.core.unique_traits import DefenderHandler
        handler = DefenderHandler()
        # Mock get_bonus to return values (simulating 2+ defenders)
        handler.get_bonus = MagicMock(return_value={"armor": 25, "magic_resist": 25})
        system.team_handlers["blue"]["defender"] = handler

        events = system.apply_combat_start_effects("blue", [unit], context)

        # Defense should have increased
        assert unit.stats.armor > initial_armor
        assert unit.stats.magic_resist > initial_mr


class TestVanquisherTrait:
    """Tests for Vanquisher ability crit."""

    @pytest.fixture
    def system(self):
        """Create trait effect system."""
        return TraitEffectSystem()

    @pytest.fixture
    def context(self):
        """Create trait effect context."""
        return TraitEffectContext(
            status_effects=MagicMock(),
            all_units={},
        )

    def test_vanquisher_crit_event_on_ability(self, system, context):
        """Vanquisher should emit crit chance event on ability cast."""
        caster = create_test_unit("caster", Team.BLUE, traits=["vanquisher"])
        target = create_test_unit("target", Team.RED)

        from src.core.unique_traits import VanquisherHandler
        handler = VanquisherHandler()
        # Mock get_bonus to return values (simulating 2+ vanquishers)
        handler.get_bonus = MagicMock(return_value={"crit_chance": 20, "crit_damage": 10})
        system.team_handlers["blue"]["vanquisher"] = handler

        events = system.apply_on_ability_cast(caster, [target], context)

        # Should have crit chance event
        crit_events = [e for e in events if e["type"] == "trait_vanquisher_crit_chance"]
        assert len(crit_events) > 0
        assert crit_events[0]["crit_chance"] > 0
