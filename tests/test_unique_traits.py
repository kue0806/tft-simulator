"""Tests for Unique Trait Handlers."""

import pytest
from unittest.mock import MagicMock

from src.core.unique_traits import (
    get_handler,
    UNIQUE_HANDLERS,
    DemaciaHandler,
    IoniaHandler,
    NoxusHandler,
    VoidHandler,
    YordleHandler,
    BilgewaterHandler,
    FreljordHandler,
    ShurimaHandler,
    ArcanistHandler,
    BruiserHandler,
    DefenderHandler,
    SlayerHandler,
    LongshotHandler,
    VanquisherHandler,
    QuickstrikerHandler,
    DisruptorHandler,
    HuntressHandler,
)
from src.core.player_units import ChampionInstance


def create_mock_champion(star_level: int = 1, traits: list = None):
    """Create a mock ChampionInstance."""
    mock_champion = MagicMock()
    mock_champion.traits = traits or []

    mock_instance = MagicMock(spec=ChampionInstance)
    mock_instance.champion = mock_champion
    mock_instance.star_level = star_level
    return mock_instance


class TestGetHandler:
    """Test handler retrieval."""

    def test_get_existing_handler(self):
        """Should return handler for existing trait."""
        handler = get_handler("demacia")
        assert handler is not None
        assert isinstance(handler, DemaciaHandler)

    def test_get_nonexistent_handler(self):
        """Should return None for nonexistent trait."""
        handler = get_handler("nonexistent_trait")
        assert handler is None

    def test_all_handlers_registered(self):
        """All handlers should be registered."""
        expected_traits = [
            "demacia", "ionia", "noxus", "void", "yordle", "bilgewater",
            "freljord", "shurima", "shadow_isles", "zaun", "darkin", "targon",
            "ixtal", "piltover", "arcanist", "bruiser", "defender", "juggernaut",
            "slayer", "gunslinger", "invoker", "longshot", "vanquisher",
            "warden", "quickstriker", "disruptor", "huntress",
        ]
        for trait in expected_traits:
            assert trait in UNIQUE_HANDLERS, f"Missing handler for {trait}"


class TestDemaciaHandler:
    """Test Demacia trait handler."""

    def test_breakpoint_3(self):
        """3 Demacia should give 12 armor/MR."""
        handler = DemaciaHandler()
        champions = [create_mock_champion() for _ in range(3)]

        bonus = handler.get_bonus(None, champions)

        assert bonus["armor"] == 12
        assert bonus["magic_resist"] == 12

    def test_breakpoint_5(self):
        """5 Demacia should give 35 armor/MR (Set 16 values)."""
        handler = DemaciaHandler()
        champions = [create_mock_champion() for _ in range(5)]

        bonus = handler.get_bonus(None, champions)

        assert bonus["armor"] == 35
        assert bonus["magic_resist"] == 35

    def test_rally_triggers_on_health_loss(self):
        """Rally should trigger when crossing 25% threshold."""
        handler = DemaciaHandler()

        # Start at 100%
        assert handler.on_health_lost(100.0) is False

        # Drop to 70% (crossed 75% threshold)
        assert handler.on_health_lost(70.0) is True
        assert handler.rally_count == 1

        # Drop to 40% (crossed 50% threshold)
        assert handler.on_health_lost(40.0) is True
        assert handler.rally_count == 2

    def test_rally_bonus_stacks(self):
        """Rally bonuses should stack with base."""
        handler = DemaciaHandler()
        champions = [create_mock_champion() for _ in range(3)]

        # Trigger 2 rallies
        handler.on_health_lost(70.0)
        handler.on_health_lost(40.0)

        bonus = handler.get_bonus(None, champions)

        # Base 12 + (2 rallies * 5)
        assert bonus["armor"] == 22
        assert bonus["magic_resist"] == 22

    def test_reset_clears_rally(self):
        """Reset should clear rally count."""
        handler = DemaciaHandler()
        handler.on_health_lost(40.0)

        handler.reset()

        assert handler.rally_count == 0
        assert handler.last_health_percent == 100.0


class TestIoniaHandler:
    """Test Ionia trait handler."""

    def test_roll_path_selects_valid_path(self):
        """Roll path should select one of the valid paths."""
        handler = IoniaHandler()

        path = handler.roll_path()

        assert path in handler.PATHS
        assert handler.current_path == path

    def test_no_bonus_without_path(self):
        """No bonus should be given if path not rolled."""
        handler = IoniaHandler()
        champions = [create_mock_champion()]

        bonus = handler.get_bonus(None, champions)

        assert bonus == {}

    def test_spirit_path_gives_health(self):
        """Spirit path should give health bonus."""
        handler = IoniaHandler()
        handler.current_path = "spirit"
        champions = [create_mock_champion()]

        bonus = handler.get_bonus(None, champions)

        assert bonus.get("health") == 200


class TestNoxusHandler:
    """Test Noxus trait handler."""

    def test_atakhan_summon_at_15_percent(self):
        """Atakhan should summon when 15% damage dealt."""
        handler = NoxusHandler()

        # Deal 10% damage
        assert handler.on_enemy_damage(10.0) is False

        # Deal 5% more (total 15%)
        assert handler.on_enemy_damage(5.0) is True
        assert handler.atakhan_summoned is True

    def test_atakhan_only_summons_once(self):
        """Atakhan should only summon once per combat."""
        handler = NoxusHandler()

        handler.on_enemy_damage(15.0)  # Summon
        assert handler.on_enemy_damage(10.0) is False  # Should not summon again

    def test_atakhan_power_scales_with_stars(self):
        """Atakhan stats should scale with total stars."""
        handler = NoxusHandler()

        # 3 one-stars = 3 total stars
        champions = [create_mock_champion(star_level=1) for _ in range(3)]
        stats = handler.calculate_atakhan_power(champions)

        assert stats["health"] == 1000 + (3 * 200)  # 1600
        assert stats["attack_damage"] == 50 + (3 * 15)  # 95


class TestVoidHandler:
    """Test Void trait handler."""

    def test_attack_speed_breakpoints(self):
        """Void should give AS based on count."""
        handler = VoidHandler()

        assert handler.get_bonus(None, [create_mock_champion()] * 2)["attack_speed"] == 8
        assert handler.get_bonus(None, [create_mock_champion()] * 4)["attack_speed"] == 18
        assert handler.get_bonus(None, [create_mock_champion()] * 6)["attack_speed"] == 28
        assert handler.get_bonus(None, [create_mock_champion()] * 9)["attack_speed"] == 35

    def test_mutation_assignment(self):
        """Mutations should be assignable to champions."""
        handler = VoidHandler()
        champ = create_mock_champion()

        assert handler.assign_mutation(champ, "vampiric") is True
        assert handler.get_champion_mutation(champ) == "vampiric"

    def test_invalid_mutation_rejected(self):
        """Invalid mutations should be rejected."""
        handler = VoidHandler()
        champ = create_mock_champion()

        assert handler.assign_mutation(champ, "invalid") is False


class TestYordleHandler:
    """Test Yordle trait handler."""

    def test_bonus_per_yordle(self):
        """Bonus should scale with yordle count."""
        handler = YordleHandler()
        champions = [create_mock_champion(star_level=1) for _ in range(3)]

        bonus = handler.get_bonus(None, champions)

        assert bonus["health"] == 50 * 3  # 150
        assert bonus["attack_speed"] == 0.07 * 3  # 0.21

    def test_three_star_bonus(self):
        """3-star yordles should give 50% more."""
        handler = YordleHandler()
        champions = [create_mock_champion(star_level=3)]

        bonus = handler.get_bonus(None, champions)

        assert bonus["health"] == 50 * 1.5  # 75
        assert bonus["attack_speed"] == 0.07 * 1.5  # 0.105


class TestBilgewaterHandler:
    """Test Bilgewater trait handler."""

    def test_serpents_per_round(self):
        """Should earn serpents at round end (Set 16 values)."""
        handler = BilgewaterHandler()
        champions = [create_mock_champion() for _ in range(5)]

        handler.apply(None, champions)  # Set serpents rate
        earned = handler.on_round_end()

        assert earned == 35  # 5 bilgewater = 35 per round
        assert handler.silver_serpents == 35

    def test_serpents_breakpoints(self):
        """Serpent rate should increase with count (Set 16 values)."""
        handler = BilgewaterHandler()

        handler.apply(None, [create_mock_champion() for _ in range(3)])
        assert handler.serpents_per_round == 15

        handler.apply(None, [create_mock_champion() for _ in range(7)])
        assert handler.serpents_per_round == 65


class TestArcanistHandler:
    """Test Arcanist class handler."""

    def test_ap_breakpoints(self):
        """Arcanist should give AP based on count."""
        handler = ArcanistHandler()

        assert handler.get_bonus(None, [create_mock_champion()] * 2)["ability_power"] == 18
        assert handler.get_bonus(None, [create_mock_champion()] * 4)["ability_power"] == 25
        assert handler.get_bonus(None, [create_mock_champion()] * 6)["ability_power"] == 40


class TestBruiserHandler:
    """Test Bruiser class handler."""

    def test_health_percent_breakpoints(self):
        """Bruiser should give health% based on count."""
        handler = BruiserHandler()

        assert handler.get_bonus(None, [create_mock_champion()] * 2)["health_percent"] == 25
        assert handler.get_bonus(None, [create_mock_champion()] * 4)["health_percent"] == 45
        assert handler.get_bonus(None, [create_mock_champion()] * 6)["health_percent"] == 65


class TestDefenderHandler:
    """Test Defender class handler."""

    def test_defense_breakpoints(self):
        """Defender should give armor/MR based on count (includes base 12)."""
        handler = DefenderHandler()

        bonus = handler.get_bonus(None, [create_mock_champion()] * 4)

        # Base 12 + 55 = 67
        assert bonus["armor"] == 67
        assert bonus["magic_resist"] == 67


class TestSlayerHandler:
    """Test Slayer class handler."""

    def test_ad_and_omnivamp_breakpoints(self):
        """Slayer should give AD% and omnivamp (Set 16 values)."""
        handler = SlayerHandler()

        bonus = handler.get_bonus(None, [create_mock_champion()] * 4)

        assert bonus["ad_percent"] == 33
        assert bonus["omnivamp"] == 16  # Fixed from 15 to 16


class TestLongshotHandler:
    """Test Longshot class handler."""

    def test_damage_amp_breakpoints(self):
        """Longshot should give damage amp based on count (Set 16 values)."""
        handler = LongshotHandler()

        # Set 16 values: base damage_amp + damage_per_hex
        assert handler.get_bonus(None, [create_mock_champion()] * 2)["damage_amp"] == 18
        assert handler.get_bonus(None, [create_mock_champion()] * 3)["damage_amp"] == 24
        assert handler.get_bonus(None, [create_mock_champion()] * 4)["damage_amp"] == 28
        assert handler.get_bonus(None, [create_mock_champion()] * 5)["damage_amp"] == 32


class TestVanquisherHandler:
    """Test Vanquisher class handler."""

    def test_crit_chance_breakpoints(self):
        """Vanquisher (Conqueror) should give crit chance and damage (Set 16 values)."""
        handler = VanquisherHandler()

        # Set 16 values: crit_chance and crit_damage are equal
        assert handler.get_bonus(None, [create_mock_champion()] * 2)["crit_chance"] == 15
        assert handler.get_bonus(None, [create_mock_champion()] * 2)["crit_damage"] == 15
        assert handler.get_bonus(None, [create_mock_champion()] * 4)["crit_chance"] == 25
        assert handler.get_bonus(None, [create_mock_champion()] * 4)["crit_damage"] == 25


class TestQuickstrikerHandler:
    """Test Quickstriker class handler."""

    def test_attack_speed_breakpoints(self):
        """Quickstriker should give AS range based on count (Set 16 values)."""
        handler = QuickstrikerHandler()

        # Set 16 values: team_attack_speed + range (attack_speed_min to attack_speed_max)
        bonus2 = handler.get_bonus(None, [create_mock_champion()] * 2)
        assert bonus2["team_attack_speed"] == 15
        assert bonus2["attack_speed_min"] == 10
        assert bonus2["attack_speed_max"] == 30

        bonus4 = handler.get_bonus(None, [create_mock_champion()] * 4)
        assert bonus4["attack_speed_min"] == 30
        assert bonus4["attack_speed_max"] == 60


class TestDisruptorHandler:
    """Test Disruptor class handler."""

    def test_dazzle_damage_breakpoints(self):
        """Disruptor should give dazzle damage based on count."""
        handler = DisruptorHandler()

        assert handler.get_bonus(None, [create_mock_champion()] * 2)["dazzle_damage"] == 25
        assert handler.get_bonus(None, [create_mock_champion()] * 4)["dazzle_damage"] == 45


class TestHuntressHandler:
    """Test Huntress class handler."""

    def test_damage_amp_breakpoints(self):
        """Huntress should give damage amp based on count."""
        handler = HuntressHandler()

        assert handler.get_bonus(None, [create_mock_champion()] * 2)["damage_amp"] == 15
        assert handler.get_bonus(None, [create_mock_champion()] * 3)["damage_amp"] == 25
