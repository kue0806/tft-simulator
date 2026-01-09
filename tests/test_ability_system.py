"""Tests for Ability System."""

import pytest
from unittest.mock import MagicMock, patch

from src.combat.hex_grid import Team, HexGrid, HexPosition
from src.combat.combat_unit import CombatUnit, CombatStats
from src.combat.ability import AbilitySystem, AbilityData
from src.combat.ability_effects import AbilityEffectHandlers
from src.combat.status_effects import StatusEffectSystem, StatusEffectType


def create_test_unit(
    unit_id: str,
    team: Team = Team.BLUE,
    hp: float = 1000,
    attack_damage: float = 100,
    ability_power: float = 100,
) -> CombatUnit:
    """Create a test combat unit."""
    stats = CombatStats(
        max_hp=hp,
        current_hp=hp,
        attack_damage=attack_damage,
        ability_power=ability_power,
        armor=50,
        magic_resist=50,
        attack_speed=1.0,
        crit_chance=0.25,
        crit_damage=1.4,
        max_mana=100,
        current_mana=100,  # Full mana to cast
        starting_mana=50,
        attack_range=1,
        dodge_chance=0.0,
        omnivamp=0.0,
        damage_amp=1.0,
        damage_reduction=0.0,
    )
    return CombatUnit(
        id=unit_id,
        name=f"Unit_{unit_id}",
        champion_id=f"TFT_Test_{unit_id}",
        star_level=1,
        team=team,
        stats=stats,
    )


class TestAbilitySystem:
    """AbilitySystem tests."""

    @pytest.fixture
    def grid(self):
        """Create hex grid."""
        return HexGrid()

    @pytest.fixture
    def status_effects(self):
        """Create status effect system."""
        return StatusEffectSystem()

    @pytest.fixture
    def ability_system(self, grid, status_effects):
        """Create ability system."""
        target_selector = MagicMock()
        damage_calculator = MagicMock()
        return AbilitySystem(
            grid=grid,
            target_selector=target_selector,
            damage_calculator=damage_calculator,
            status_effect_system=status_effects,
        )

    @pytest.fixture
    def effect_handlers(self, ability_system):
        """Create ability effect handlers."""
        handlers = AbilityEffectHandlers(ability_system)
        handlers.register_all_handlers()
        return handlers

    def test_ability_system_init(self, ability_system):
        """Test ability system initialization."""
        assert ability_system is not None
        assert ability_system.grid is not None

    def test_register_handlers(self, ability_system, effect_handlers):
        """Test that handlers are registered."""
        # Check some key handlers are registered in the ability system
        assert "dr_mundo_goes_where_he_pleases" in ability_system._custom_ability_handlers
        assert "mel_golden_barrier" in ability_system._custom_ability_handlers
        assert "milio_cozy_campfire" in ability_system._custom_ability_handlers


class TestAbilityEffectHandlers:
    """Tests for specific ability effect handlers."""

    @pytest.fixture
    def grid(self):
        """Create hex grid."""
        return HexGrid()

    @pytest.fixture
    def status_effects(self):
        """Create status effect system."""
        return StatusEffectSystem()

    @pytest.fixture
    def ability_system(self, grid, status_effects):
        """Create ability system."""
        target_selector = MagicMock()
        damage_calculator = MagicMock()
        return AbilitySystem(
            grid=grid,
            target_selector=target_selector,
            damage_calculator=damage_calculator,
            status_effect_system=status_effects,
        )

    @pytest.fixture
    def handlers(self, ability_system):
        """Get effect handlers."""
        handlers = AbilityEffectHandlers(ability_system)
        handlers.register_all_handlers()
        return handlers


class TestDrMundoHandler:
    """Tests for Dr. Mundo ability handler."""

    @pytest.fixture
    def grid(self):
        """Create hex grid."""
        return HexGrid()

    @pytest.fixture
    def status_effects(self):
        """Create status effect system."""
        return StatusEffectSystem()

    @pytest.fixture
    def ability_system(self, grid, status_effects):
        """Create ability system."""
        target_selector = MagicMock()
        damage_calculator = MagicMock()
        return AbilitySystem(
            grid=grid,
            target_selector=target_selector,
            damage_calculator=damage_calculator,
            status_effect_system=status_effects,
        )

    @pytest.fixture
    def handlers(self, ability_system):
        """Get effect handlers."""
        handlers = AbilityEffectHandlers(ability_system)
        handlers.register_all_handlers()
        return handlers

    def test_dr_mundo_heals_self(self, handlers, grid, status_effects):
        """Dr. Mundo's ability should heal himself over time."""
        caster = create_test_unit("mundo", Team.BLUE, hp=1000)
        caster.stats.current_hp = 500  # Damaged
        grid.place_unit("mundo", HexPosition(0, 0))

        ability = AbilityData(
            ability_id="dr_mundo_goes_where_he_pleases",
            name="Maximum Dosage",
            description="Dr. Mundo heals and gains AD.",
            base_damage=[0, 0, 0],
            damage_type="none",
            custom_data={
                "hp_percent_heal": [0.20, 0.25, 0.80],
                "duration": 4.0,
                "ad_bonus_percent": [0.15, 0.20, 0.50],
            },
        )

        all_units = {"mundo": caster}

        result = handlers.handle_dr_mundo_maximum_dosage(caster, ability, all_units)

        assert result.success is True
        assert caster.id in result.targets_hit
        # Check that HoT effect was applied
        assert status_effects.has_effect(caster, StatusEffectType.HEAL_OVER_TIME)
        # Check that AD buff was applied
        assert status_effects.has_effect(caster, StatusEffectType.ATTACK_DAMAGE_BUFF)

    def test_dr_mundo_ad_buff_scales_with_ad(self, handlers, grid, status_effects):
        """Dr. Mundo's AD buff should scale with his attack damage."""
        caster = create_test_unit("mundo", Team.BLUE, hp=1000, attack_damage=200)
        grid.place_unit("mundo", HexPosition(0, 0))

        ability = AbilityData(
            ability_id="dr_mundo_goes_where_he_pleases",
            name="Maximum Dosage",
            description="Dr. Mundo heals and gains AD.",
            base_damage=[0, 0, 0],
            damage_type="none",
            custom_data={
                "hp_percent_heal": [0.20, 0.25, 0.80],
                "duration": 4.0,
                "ad_bonus_percent": [0.15, 0.20, 0.50],
            },
        )

        all_units = {"mundo": caster}
        result = handlers.handle_dr_mundo_maximum_dosage(caster, ability, all_units)

        assert result.success is True
        # AD buff should be 15% of 200 = 30
        ad_buffs = [e for e in status_effects.get_effects(caster)
                   if e.effect_type == StatusEffectType.ATTACK_DAMAGE_BUFF]
        assert len(ad_buffs) > 0
        assert ad_buffs[0].value == 30  # 200 * 0.15


class TestMelHandler:
    """Tests for Mel ability handler."""

    @pytest.fixture
    def grid(self):
        """Create hex grid."""
        return HexGrid()

    @pytest.fixture
    def status_effects(self):
        """Create status effect system."""
        return StatusEffectSystem()

    @pytest.fixture
    def ability_system(self, grid, status_effects):
        """Create ability system."""
        target_selector = MagicMock()
        damage_calculator = MagicMock()
        return AbilitySystem(
            grid=grid,
            target_selector=target_selector,
            damage_calculator=damage_calculator,
            status_effect_system=status_effects,
        )

    @pytest.fixture
    def handlers(self, ability_system):
        """Get effect handlers."""
        handlers = AbilityEffectHandlers(ability_system)
        handlers.register_all_handlers()
        return handlers

    def test_mel_shields_all_allies(self, handlers, grid, status_effects):
        """Mel's ability should shield all allies."""
        caster = create_test_unit("mel", Team.BLUE, ability_power=150)
        ally1 = create_test_unit("ally1", Team.BLUE)
        ally2 = create_test_unit("ally2", Team.BLUE)
        enemy = create_test_unit("enemy", Team.RED)

        grid.place_unit("mel", HexPosition(0, 0))
        grid.place_unit("ally1", HexPosition(1, 0))
        grid.place_unit("ally2", HexPosition(2, 0))
        grid.place_unit("enemy", HexPosition(3, 0))

        ability = AbilityData(
            ability_id="mel_golden_barrier",
            name="Council's Blessing",
            description="Mel shields all allies.",
            base_damage=[0, 0, 0],
            damage_type="none",
            custom_data={
                "base_shield": [200, 300, 1000],
                "ap_scaling": 0.4,
                "ap_buff": [10, 15, 40],
                "duration": 4.0,
            },
        )

        all_units = {"mel": caster, "ally1": ally1, "ally2": ally2, "enemy": enemy}
        result = handlers.handle_mel_councils_blessing(caster, ability, all_units)

        assert result.success is True
        # All 3 allies (including caster) should be shielded
        assert len(result.targets_hit) == 3
        assert "mel" in result.targets_hit
        assert "ally1" in result.targets_hit
        assert "ally2" in result.targets_hit
        assert "enemy" not in result.targets_hit

        # Check shield effects were applied
        assert status_effects.has_effect(caster, StatusEffectType.SHIELD)
        assert status_effects.has_effect(ally1, StatusEffectType.SHIELD)
        assert status_effects.has_effect(ally2, StatusEffectType.SHIELD)
        assert not status_effects.has_effect(enemy, StatusEffectType.SHIELD)

    def test_mel_grants_ap_buff(self, handlers, grid, status_effects):
        """Mel's ability should grant AP buff to allies."""
        caster = create_test_unit("mel", Team.BLUE)
        ally1 = create_test_unit("ally1", Team.BLUE)

        grid.place_unit("mel", HexPosition(0, 0))
        grid.place_unit("ally1", HexPosition(1, 0))

        ability = AbilityData(
            ability_id="mel_golden_barrier",
            name="Council's Blessing",
            description="Mel shields all allies.",
            base_damage=[0, 0, 0],
            damage_type="none",
            custom_data={
                "base_shield": [200, 300, 1000],
                "ap_scaling": 0.4,
                "ap_buff": [10, 15, 40],
                "duration": 4.0,
            },
        )

        all_units = {"mel": caster, "ally1": ally1}
        handlers.handle_mel_councils_blessing(caster, ability, all_units)

        # Check AP buff effects were applied
        assert status_effects.has_effect(caster, StatusEffectType.ABILITY_POWER_BUFF)
        assert status_effects.has_effect(ally1, StatusEffectType.ABILITY_POWER_BUFF)


class TestMilioHandler:
    """Tests for Milio ability handler."""

    @pytest.fixture
    def grid(self):
        """Create hex grid."""
        return HexGrid()

    @pytest.fixture
    def status_effects(self):
        """Create status effect system."""
        return StatusEffectSystem()

    @pytest.fixture
    def ability_system(self, grid, status_effects):
        """Create ability system."""
        target_selector = MagicMock()
        damage_calculator = MagicMock()
        return AbilitySystem(
            grid=grid,
            target_selector=target_selector,
            damage_calculator=damage_calculator,
            status_effect_system=status_effects,
        )

    @pytest.fixture
    def handlers(self, ability_system):
        """Get effect handlers."""
        handlers = AbilityEffectHandlers(ability_system)
        handlers.register_all_handlers()
        return handlers

    def test_milio_heals_all_allies(self, handlers, grid, status_effects):
        """Milio's ability should heal all allies."""
        caster = create_test_unit("milio", Team.BLUE, ability_power=100)
        ally1 = create_test_unit("ally1", Team.BLUE)
        ally2 = create_test_unit("ally2", Team.BLUE)

        # Damage allies
        caster.stats.current_hp = 500
        ally1.stats.current_hp = 300
        ally2.stats.current_hp = 400

        grid.place_unit("milio", HexPosition(0, 0))
        grid.place_unit("ally1", HexPosition(1, 0))
        grid.place_unit("ally2", HexPosition(2, 0))

        ability = AbilityData(
            ability_id="milio_cozy_campfire",
            name="Breath of Life",
            description="Milio heals all allies.",
            base_damage=[0, 0, 0],
            damage_type="none",
            custom_data={
                "base_heal": [150, 225, 800],
                "ap_scaling": 0.35,
            },
        )

        all_units = {"milio": caster, "ally1": ally1, "ally2": ally2}
        result = handlers.handle_milio_breath_of_life(caster, ability, all_units)

        assert result.success is True
        assert len(result.targets_hit) == 3
        assert result.total_healing > 0

        # All allies should have more HP
        # Heal amount = 150 + (100 * 0.35) = 185
        assert caster.stats.current_hp > 500
        assert ally1.stats.current_hp > 300
        assert ally2.stats.current_hp > 400

    def test_milio_cleanses_cc(self, handlers, grid, status_effects):
        """Milio's ability should cleanse CC effects."""
        caster = create_test_unit("milio", Team.BLUE)
        ally1 = create_test_unit("ally1", Team.BLUE)

        grid.place_unit("milio", HexPosition(0, 0))
        grid.place_unit("ally1", HexPosition(1, 0))

        # Apply CC to ally
        from src.combat.status_effects import StatusEffect
        stun = StatusEffect(
            effect_type=StatusEffectType.STUN,
            source_id="enemy",
            duration=3.0,
        )
        status_effects.apply_effect(ally1, stun)
        assert status_effects.has_effect(ally1, StatusEffectType.STUN)

        ability = AbilityData(
            ability_id="milio_cozy_campfire",
            name="Breath of Life",
            description="Milio heals all allies.",
            base_damage=[0, 0, 0],
            damage_type="none",
            custom_data={
                "base_heal": [150, 225, 800],
                "ap_scaling": 0.35,
            },
        )

        all_units = {"milio": caster, "ally1": ally1}
        handlers.handle_milio_breath_of_life(caster, ability, all_units)

        # Stun should be cleansed
        assert not status_effects.has_effect(ally1, StatusEffectType.STUN)


class TestAbilityDataCustomData:
    """Tests for ability custom data handling."""

    def test_custom_data_default_values(self):
        """Test that abilities use default values when custom_data is missing."""
        ability = AbilityData(
            ability_id="test_ability",
            name="Test",
            description="Test ability",
        )

        assert ability.custom_data == {}
        assert ability.custom_data.get("missing_key", 10) == 10

    def test_custom_data_star_level_scaling(self):
        """Test that abilities properly scale with star level."""
        ability = AbilityData(
            ability_id="test_ability",
            name="Test",
            description="Test ability",
            custom_data={
                "bonus": [10, 20, 30],
            },
        )

        for star in range(1, 4):
            idx = min(star - 1, 2)
            assert ability.base_damage[idx] == [0, 0, 0][idx]  # default
            assert ability.custom_data["bonus"][idx] == [10, 20, 30][idx]
