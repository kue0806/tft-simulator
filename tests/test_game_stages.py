"""Tests for TFT game stages and round mechanics.

Tests the new game stage system including:
- Round types and phases
- Carousel mechanics
- Augment selection
- PvE encounters
- Gold/XP progression
"""

import pytest
from src.core.stage_manager import StageManager, RoundType, RoundPhase
from src.core.constants import (
    CAROUSEL_ROUNDS,
    AUGMENT_ROUNDS,
    PVE_ROUNDS,
    get_round_passive_gold,
    PASSIVE_XP_PER_ROUND,
)
from src.core.pve_system import PvESystem
from src.core.carousel import CarouselSystem
from src.core.augment import AugmentSystem, AugmentTier


class TestStageManager:
    """Test StageManager functionality."""

    def test_initial_state(self):
        """Test that game starts at 1-1."""
        sm = StageManager()
        assert sm.current_stage == 1
        assert sm.current_round == 1
        assert sm.get_stage_string() == "1-1"

    def test_round_1_1_is_carousel(self):
        """Test that 1-1 is a carousel round."""
        sm = StageManager()
        info = sm.get_current_round_info()
        assert info.is_carousel is True
        assert info.round_type == RoundType.CAROUSEL

    def test_round_progression_stage_1(self):
        """Test advancing through stage 1."""
        sm = StageManager()
        stages = ["1-1", "1-2", "1-3", "1-4", "2-1"]

        for i, expected in enumerate(stages):
            assert sm.get_stage_string() == expected, f"Expected {expected} at step {i}"
            if i < len(stages) - 1:
                sm.advance_round()

    def test_round_types_stage_1(self):
        """Test round types in stage 1."""
        sm = StageManager()

        # 1-1 is carousel
        assert sm.get_current_round_info().is_carousel is True

        # 1-2, 1-3, 1-4 are PvE
        for _ in range(3):
            sm.advance_round()
            info = sm.get_current_round_info()
            assert info.is_pve is True
            assert info.round_type == RoundType.PVE

    def test_augment_rounds(self):
        """Test that augment rounds are correctly identified."""
        sm = StageManager()

        # Advance to 2-1
        for _ in range(4):
            sm.advance_round()

        assert sm.get_stage_string() == "2-1"
        info = sm.get_current_round_info()
        assert info.is_augment is True

        # Advance to 3-2
        for _ in range(8):
            sm.advance_round()

        assert sm.get_stage_string() == "3-2"
        info = sm.get_current_round_info()
        assert info.is_augment is True

    def test_carousel_rounds_stage_2_plus(self):
        """Test that X-4 rounds are carousels."""
        sm = StageManager()

        # Advance to 2-4
        for _ in range(7):
            sm.advance_round()

        assert sm.get_stage_string() == "2-4"
        info = sm.get_current_round_info()
        assert info.is_carousel is True

    def test_pve_rounds_end_of_stage(self):
        """Test that X-7 rounds are PvE."""
        sm = StageManager()

        # Advance to 2-7
        for _ in range(10):
            sm.advance_round()

        assert sm.get_stage_string() == "2-7"
        info = sm.get_current_round_info()
        assert info.is_pve is True
        assert info.monster_type == "krugs"

    def test_passive_gold_early_rounds(self):
        """Test passive gold for early rounds."""
        assert get_round_passive_gold("1-1") == 0
        assert get_round_passive_gold("1-2") == 2
        assert get_round_passive_gold("1-3") == 2
        assert get_round_passive_gold("1-4") == 3
        assert get_round_passive_gold("2-1") == 4
        assert get_round_passive_gold("2-2") == 5  # Default 5

    def test_passive_xp(self):
        """Test passive XP per round."""
        sm = StageManager()

        # 1-1 has no XP
        info = sm.get_current_round_info()
        assert info.passive_xp == 0

        # Other rounds have 2 XP
        sm.advance_round()
        info = sm.get_current_round_info()
        assert info.passive_xp == PASSIVE_XP_PER_ROUND

    def test_round_phases(self):
        """Test that rounds have correct phases."""
        sm = StageManager()

        # 1-1 carousel: CAROUSEL phase only
        info = sm.get_current_round_info()
        assert RoundPhase.CAROUSEL in info.phases

        # 1-2 PvE: PLANNING, COMBAT, LOOT
        sm.advance_round()
        info = sm.get_current_round_info()
        assert RoundPhase.PLANNING in info.phases
        assert RoundPhase.COMBAT in info.phases
        assert RoundPhase.LOOT in info.phases

        # Advance to 2-1 (augment round): AUGMENT_SELECTION, PLANNING, COMBAT
        for _ in range(3):
            sm.advance_round()
        info = sm.get_current_round_info()
        assert RoundPhase.AUGMENT_SELECTION in info.phases
        assert RoundPhase.PLANNING in info.phases


class TestPvESystem:
    """Test PvE system functionality."""

    def test_is_pve_round(self):
        """Test PvE round detection."""
        pve = PvESystem(seed=42)
        assert pve.is_pve_round("1-2") is True
        assert pve.is_pve_round("1-3") is True
        assert pve.is_pve_round("2-1") is False
        assert pve.is_pve_round("2-7") is True

    def test_monster_types(self):
        """Test monster type identification."""
        pve = PvESystem(seed=42)
        assert pve.get_monster_type("1-2") == "minions"
        assert pve.get_monster_type("2-7") == "krugs"
        assert pve.get_monster_type("3-7") == "wolves"
        assert pve.get_monster_type("4-7") == "raptors"
        assert pve.get_monster_type("5-7") == "rift_herald"

    def test_pve_combat_simulation(self):
        """Test PvE combat returns results."""
        pve = PvESystem(seed=42)
        result = pve.simulate_pve_combat("1-2", player_id=0, player_power=100)

        assert result.round_stage == "1-2"
        assert result.monster_type == "minions"
        # With seed and reasonable power, should usually win
        # At least verify we get valid results
        assert isinstance(result.won, bool)
        assert isinstance(result.gold_gained, int)


class TestCarouselSystem:
    """Test carousel system functionality."""

    def test_is_carousel_round(self):
        """Test carousel round detection."""
        carousel = CarouselSystem(seed=42)
        assert carousel.is_carousel_round("1-1") is True
        assert carousel.is_carousel_round("1-2") is False
        assert carousel.is_carousel_round("2-4") is True

    def test_generate_carousel(self):
        """Test carousel generation."""
        carousel = CarouselSystem(seed=42)
        champions = carousel.generate_carousel("1-1")

        # Carousel should have up to 9 champions (limited by pool size)
        assert len(champions) >= 1 and len(champions) <= 9
        for champ in champions:
            assert champ.cost == 1  # 1-1 only has 1-cost

    def test_pick_order_first_carousel(self):
        """Test that first carousel has all players pick simultaneously."""
        carousel = CarouselSystem(seed=42)
        player_healths = {i: 100 for i in range(8)}

        pick_order = carousel.calculate_pick_order(player_healths, "1-1")

        # All players should have pick_round = 0 for first carousel
        assert all(po.pick_round == 0 for po in pick_order)

    def test_pick_order_by_health(self):
        """Test that later carousels order by health."""
        carousel = CarouselSystem(seed=42)
        player_healths = {
            0: 100,
            1: 50,
            2: 75,
            3: 25,
        }

        pick_order = carousel.calculate_pick_order(player_healths, "2-4")

        # Player 3 (25 HP) and player 1 (50 HP) should pick first
        first_pickers = [po.player_id for po in pick_order if po.pick_round == 0]
        assert 3 in first_pickers
        assert 1 in first_pickers


class TestAugmentSystem:
    """Test augment system functionality."""

    def test_is_augment_round(self):
        """Test augment round detection."""
        augment = AugmentSystem(seed=42)
        assert augment.is_augment_round("2-1") is True
        assert augment.is_augment_round("3-2") is True
        assert augment.is_augment_round("4-2") is True
        assert augment.is_augment_round("2-2") is False

    def test_generate_augment_choices(self):
        """Test augment choice generation."""
        augment = AugmentSystem(seed=42)
        choice = augment.generate_augment_choices("2-1", player_id=0)

        assert len(choice.options) <= 3
        assert choice.timer_seconds == 43  # First augment has 43s timer
        assert choice.selected is None

    def test_augment_selection(self):
        """Test selecting an augment."""
        augment = AugmentSystem(seed=42)
        choice = augment.generate_augment_choices("2-1", player_id=0)

        if choice.options:
            selected = augment.select_augment(choice, player_id=0, augment_index=0)
            assert selected is not None
            assert choice.selected == selected
            assert selected in augment.get_player_augments(0)

    def test_no_triple_economy_augments(self):
        """Test that no round offers 3 economy augments."""
        augment = AugmentSystem(seed=42)

        # Generate many choices to verify rule
        for _ in range(10):
            choice = augment.generate_augment_choices("2-1", player_id=0)
            from src.core.augment import AugmentCategory
            economy_count = sum(
                1 for opt in choice.options
                if opt.category == AugmentCategory.ECONOMY
            )
            assert economy_count <= 2
            augment.reset_game()


class TestIntegration:
    """Integration tests for game flow."""

    def test_full_stage_1_flow(self):
        """Test complete stage 1 game flow."""
        sm = StageManager()

        # 1-1: Carousel
        info = sm.get_current_round_info()
        assert info.is_carousel
        assert info.passive_gold == 0
        assert info.passive_xp == 0

        # 1-2: First PvE
        sm.advance_round()
        info = sm.get_current_round_info()
        assert info.is_pve
        assert info.passive_gold == 2
        assert info.passive_xp == 2

        # 1-3: PvE
        sm.advance_round()
        info = sm.get_current_round_info()
        assert info.is_pve
        assert info.passive_gold == 2

        # 1-4: Last PvE of stage 1
        sm.advance_round()
        info = sm.get_current_round_info()
        assert info.is_pve
        assert info.passive_gold == 3

        # 2-1: First PvP with augment
        sm.advance_round()
        info = sm.get_current_round_info()
        assert not info.is_pve
        assert info.is_augment
        assert info.passive_gold == 4

    def test_stage_transition(self):
        """Test stage number increments correctly."""
        sm = StageManager()

        # Go through stage 1 (4 rounds)
        for _ in range(4):
            sm.advance_round()

        assert sm.current_stage == 2
        assert sm.current_round == 1

        # Go through stage 2 (7 rounds)
        for _ in range(7):
            sm.advance_round()

        assert sm.current_stage == 3
        assert sm.current_round == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
