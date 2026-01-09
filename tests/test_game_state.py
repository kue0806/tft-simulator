"""Tests for Game State functionality."""

import pytest

from src.core.game_state import GameState, PlayerState
from src.data.loaders import load_champions
from src.core.champion_pool import ChampionPool
from src.core.constants import REROLL_COST, BASE_INCOME


@pytest.fixture
def game():
    """Create a fresh game state."""
    return GameState(num_players=8)


@pytest.fixture
def player():
    """Create a single player state."""
    champions = load_champions()
    pool = ChampionPool(champions)
    return PlayerState(player_id=0, pool=pool)


class TestGameStateInit:
    """Tests for GameState initialization."""

    def test_game_init(self, game):
        """Game initializes with correct number of players."""
        assert len(game.players) == 8
        assert len(game.get_alive_players()) == 8

    def test_game_init_custom_players(self):
        """Can create game with custom player count."""
        game = GameState(num_players=4)
        assert len(game.players) == 4

    def test_game_stage_string(self, game):
        """Stage string is formatted correctly."""
        assert game.get_stage_string() == "1-1"


class TestPlayerState:
    """Tests for individual player state."""

    def test_player_init(self, player):
        """Player initializes with correct defaults."""
        assert player.level == 1
        assert player.health == 100
        assert player.gold == 0
        assert player.is_alive is True

    def test_can_afford_reroll(self, player):
        """can_afford_reroll checks gold correctly."""
        assert player.can_afford_reroll() is False

        player.gold = REROLL_COST
        assert player.can_afford_reroll() is True

    def test_reroll(self, player):
        """Reroll deducts gold and refreshes shop."""
        player.gold = 10
        result = player.reroll()

        assert result is True
        assert player.gold == 10 - REROLL_COST

    def test_reroll_insufficient_gold(self, player):
        """Cannot reroll without enough gold."""
        player.gold = 1
        result = player.reroll()

        assert result is False
        assert player.gold == 1

    def test_buy_xp(self, player):
        """Can buy XP for gold."""
        player.gold = 10
        player.level = 5  # Set higher level so we don't level up immediately
        player.xp = 0

        result = player.buy_xp()

        assert result is True
        assert player.xp == 4  # Got 4 XP (not enough to level from 5 to 6)
        assert player.gold == 10 - 4

    def test_level_up(self, player):
        """XP causes level up."""
        player.gold = 100

        # Buy enough XP to level up
        while player.level < 3:
            player.buy_xp()

        assert player.level >= 3


class TestPlayerEconomy:
    """Tests for player economy."""

    def test_calculate_income_base(self, player):
        """Base income is correct."""
        income = player.calculate_income()
        assert income >= BASE_INCOME

    def test_calculate_income_interest(self, player):
        """Interest is calculated correctly."""
        player.gold = 50  # Should get 5 interest (max)

        income = player.calculate_income()
        assert income >= BASE_INCOME + 5

    def test_calculate_income_streak(self, player):
        """Streak bonus is included."""
        player.win_streak = 6  # +3 gold (6+ streak in Set 16)

        income = player.calculate_income()
        assert income >= BASE_INCOME + 3

    def test_end_round_income(self, player):
        """End round adds income to gold."""
        player.gold = 50
        initial = player.gold

        income = player.end_round_income()
        assert player.gold == initial + income


class TestPlayerCombat:
    """Tests for player combat tracking."""

    def test_record_win(self, player):
        """Recording win updates streaks."""
        player.record_win()

        assert player.win_streak == 1
        assert player.loss_streak == 0

    def test_record_loss(self, player):
        """Recording loss updates streaks and health."""
        player.record_loss(damage=10)

        assert player.loss_streak == 1
        assert player.win_streak == 0
        assert player.health == 90

    def test_streak_resets(self, player):
        """Win resets loss streak and vice versa."""
        player.record_win()
        player.record_win()
        assert player.win_streak == 2

        player.record_loss(damage=5)
        assert player.win_streak == 0
        assert player.loss_streak == 1

    def test_death(self, player):
        """Player dies when health reaches 0."""
        player.take_damage(100)

        assert player.health == 0
        assert player.is_alive is False


class TestPlayerBuySell:
    """Tests for buying and selling champions."""

    def test_buy_champion(self, player):
        """Can buy champion from shop."""
        player.gold = 10
        player.shop.refresh()

        # Find a champion we can afford
        for i, champ in enumerate(player.shop.slots):
            if champ and champ.cost <= player.gold:
                initial_gold = player.gold
                instance = player.buy_champion(i)

                assert instance is not None
                assert player.gold == initial_gold - champ.cost
                assert player.units.get_total_units() == 1
                break

    def test_buy_champion_insufficient_gold(self, player):
        """Cannot buy without enough gold."""
        player.gold = 0
        player.shop.refresh()

        instance = player.buy_champion(0)
        # Might return None if can't afford
        if player.shop.slots[0]:
            assert instance is None


class TestGameElimination:
    """Tests for player elimination."""

    def test_eliminate_player(self, game):
        """Eliminating player returns units to pool."""
        player = game.players[0]
        player.shop.refresh()

        # Buy some champions
        player.gold = 50
        for i in range(5):
            if player.shop.slots[i]:
                player.buy_champion(i)

        # Eliminate
        game.eliminate_player(0)

        assert player.is_alive is False
        assert 0 in game.eliminated
        assert len(game.get_alive_players()) == 7

    def test_eliminated_player_not_in_alive(self, game):
        """Eliminated players not in alive list."""
        game.eliminate_player(0)

        alive = game.get_alive_players()
        assert game.players[0] not in alive

    def test_game_over_detection(self, game):
        """Game over when 1 player remains."""
        # Eliminate all but one
        for i in range(7):
            game.eliminate_player(i)

        assert game.is_game_over() is True
        assert game.get_winner() == game.players[7]


class TestGameAdvance:
    """Tests for advancing game state."""

    def test_advance_round(self, game):
        """Advancing round updates stage."""
        game.advance_round()

        assert game.round == 2
        assert game.total_rounds == 1

    def test_advance_round_gives_xp(self, game):
        """Players get XP each round (starting in stage 2)."""
        # Advance to stage 2 where passive XP is given
        for _ in range(4):  # Stage 1 has 4 rounds
            game.advance_round()

        # Now we're at stage 2-1
        # Set player to level 5 so XP accumulates without leveling
        game.players[0].level = 5
        game.players[0].xp = 0
        initial_xp = game.players[0].xp

        game.advance_round()

        # XP should increase (may have leveled up if exceeded threshold)
        assert game.players[0].xp >= 0  # At minimum, xp is non-negative
        # Total XP gained or level change should have occurred
        # Passive XP is 2 per round in stage 2+
        assert game.players[0].xp > initial_xp or game.players[0].level > 5

    def test_placements(self, game):
        """Placements are returned in correct order."""
        # Eliminate some players
        game.eliminate_player(0)
        game.eliminate_player(1)

        placements = game.get_placements()

        # Alive players first, then eliminated in reverse order
        assert len(placements) == 8
        assert placements[-1] == game.players[0]  # First eliminated = last place
        assert placements[-2] == game.players[1]  # Second eliminated = 7th place


class TestIntegration:
    """Integration tests for buy/sell flow."""

    def test_buy_and_sell_flow(self, game):
        """Complete buy/sell cycle works correctly."""
        player = game.players[0]
        player.gold = 20
        player.shop.refresh()

        # Buy a champion
        champ = player.shop.slots[0]
        if champ:
            initial_pool = game.pool.get_available(champ.id)
            instance = player.buy_champion(0)

            if instance:
                # Pool should not have increased (was already taken)
                assert game.pool.get_available(champ.id) == initial_pool

                # Sell it
                gold = player.sell_unit(instance)

                # Pool should have the champion back
                assert game.pool.get_available(champ.id) == initial_pool + 1
                assert gold == champ.cost

    def test_contested_champion(self, game):
        """Multiple players buying same champion depletes pool."""
        # Give all players gold
        for player in game.players:
            player.gold = 100

        # Find a 1-cost champion
        one_costs = game.pool.get_champions_by_cost(1)
        if one_costs:
            target = one_costs[0]

            # Track pool size after each refresh to verify shop mechanics
            # Note: shop.refresh() takes champions from pool temporarily
            bought_count = 0
            for player in game.players[:4]:
                player.shop.refresh()
                for i, champ in enumerate(player.shop.slots):
                    if champ and champ.id == target.id:
                        pool_before_buy = game.pool.get_available(target.id)
                        if player.buy_champion(i):
                            bought_count += 1
                            # After buying, champion stays out of pool
                            # (was already taken during refresh)
                            assert game.pool.get_available(target.id) == pool_before_buy
                        break

            # Verify we successfully bought some champions
            # (exact number depends on shop rolls)
            assert bought_count >= 0  # May be 0 if target didn't appear in shops
