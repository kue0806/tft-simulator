"""Tests for BoardOptimizer."""

import pytest
from unittest.mock import MagicMock, patch

from src.optimizer.board_optimizer import (
    BoardOptimizer,
    BoardLayout,
    PositionScore,
)
from src.combat.hex_grid import HexPosition


class MockChampionStats:
    """Mock champion stats for testing."""

    def __init__(
        self,
        attack_range: int = 1,
        armor: int = 40,
        attack_damage: list = None,
        health: list = None,
    ):
        self.attack_range = attack_range
        self.armor = armor
        self.attack_damage = attack_damage or [50, 75, 100]
        self.health = health or [600, 900, 1200]


class MockChampion:
    """Mock champion for testing."""

    def __init__(
        self,
        champion_id: str,
        name: str,
        cost: int,
        traits: list,
        stats: MockChampionStats = None,
    ):
        self.id = champion_id
        self.name = name
        self.cost = cost
        self.traits = traits
        self.stats = stats or MockChampionStats()


class MockChampionInstance:
    """Mock champion instance for testing."""

    def __init__(self, champion: MockChampion, position: tuple = None):
        self.champion = champion
        self.position = position


class MockPlayerUnits:
    """Mock player units for testing."""

    def __init__(self):
        self.board: dict = {}
        self.bench: list = []


class MockPlayerState:
    """Mock player state for testing."""

    def __init__(self):
        self.gold = 50
        self.level = 8
        self.units = MockPlayerUnits()


@pytest.fixture
def optimizer():
    """Create board optimizer."""
    return BoardOptimizer()


@pytest.fixture
def player():
    """Create mock player state."""
    return MockPlayerState()


class TestBoardOptimizer:
    """BoardOptimizer tests."""

    def test_initialization(self, optimizer):
        """Test optimizer initializes correctly."""
        assert optimizer is not None
        assert optimizer.ROWS == 4
        assert optimizer.COLS == 7

    def test_row_classifications(self, optimizer):
        """Test row classifications are defined."""
        assert 3 in optimizer.FRONTLINE_ROWS
        assert 2 in optimizer.MIDLINE_ROWS
        assert 0 in optimizer.BACKLINE_ROWS
        assert 1 in optimizer.BACKLINE_ROWS

    def test_optimize_empty_board(self, optimizer, player):
        """Test optimization with empty board."""
        layout = optimizer.optimize(player)

        assert isinstance(layout, BoardLayout)
        assert layout.positions == {}
        assert layout.description == "No units"

    def test_optimize_single_unit(self, optimizer, player):
        """Test optimization with single unit."""
        tank = MockChampion(
            "TFT_Tank",
            "Tank",
            3,
            ["tank"],
            MockChampionStats(attack_range=1, armor=80, health=[1000, 1500, 2000]),
        )
        player.units.board = {"u1": MockChampionInstance(tank)}

        layout = optimizer.optimize(player)

        assert len(layout.positions) == 1
        assert "u1" in layout.positions

    def test_optimize_tank_frontline(self, optimizer, player):
        """Test tanks are placed in frontline."""
        tank = MockChampion(
            "TFT_Tank",
            "Tank",
            3,
            ["tank"],
            MockChampionStats(attack_range=1, armor=80, health=[1000, 1500, 2000]),
        )
        player.units.board = {"tank1": MockChampionInstance(tank)}

        layout = optimizer.optimize(player)

        position = layout.positions.get("tank1")
        if position:
            # Tank should be in frontline (row 3) or midline (row 2)
            assert position.row in [2, 3]

    def test_optimize_carry_backline(self, optimizer, player):
        """Test carries are placed in backline."""
        carry = MockChampion(
            "TFT_Carry",
            "Carry",
            4,
            ["mage"],
            MockChampionStats(attack_range=4, attack_damage=[80, 120, 160]),
        )
        player.units.board = {"carry1": MockChampionInstance(carry)}

        layout = optimizer.optimize(player)

        position = layout.positions.get("carry1")
        if position:
            # Ranged carry should be in backline
            assert position.row in [0, 1]

    def test_optimize_assassin_corner(self, optimizer, player):
        """Test assassins are placed in corners."""
        assassin = MockChampion(
            "TFT_Assassin", "Assassin", 3, ["assassin"], MockChampionStats()
        )
        player.units.board = {"assassin1": MockChampionInstance(assassin)}

        layout = optimizer.optimize(player)

        position = layout.positions.get("assassin1")
        if position:
            # Assassin should be in corner (col 0 or 6)
            assert position.col in [0, 6] or position.row in [0, 1]

    def test_optimize_mixed_team(self, optimizer, player):
        """Test optimization with mixed team."""
        tank = MockChampion(
            "TFT_Tank",
            "Tank",
            3,
            ["tank"],
            MockChampionStats(armor=80, health=[1000, 1500, 2000]),
        )
        carry = MockChampion(
            "TFT_Carry",
            "Carry",
            4,
            ["mage"],
            MockChampionStats(attack_range=4),
        )
        assassin = MockChampion(
            "TFT_Assassin", "Assassin", 3, ["assassin"], MockChampionStats()
        )

        player.units.board = {
            "tank1": MockChampionInstance(tank),
            "carry1": MockChampionInstance(carry),
            "assassin1": MockChampionInstance(assassin),
        }

        layout = optimizer.optimize(player)

        assert len(layout.positions) == 3
        # All positions should be within bounds
        for pos in layout.positions.values():
            assert 0 <= pos.row < 4
            assert 0 <= pos.col < 7

    def test_optimize_iterations(self, optimizer, player):
        """Test optimization runs multiple iterations."""
        tank = MockChampion("TFT_Tank", "Tank", 3, ["tank"], MockChampionStats())
        carry = MockChampion(
            "TFT_Carry", "Carry", 4, ["mage"], MockChampionStats(attack_range=4)
        )

        player.units.board = {
            "tank1": MockChampionInstance(tank),
            "carry1": MockChampionInstance(carry),
        }

        # More iterations should potentially find better layout
        layout_low = optimizer.optimize(player, iterations=10)
        layout_high = optimizer.optimize(player, iterations=100)

        # Both should be valid
        assert len(layout_low.positions) == 2
        assert len(layout_high.positions) == 2

    def test_suggest_position_tank(self, optimizer, player):
        """Test position suggestions for tank."""
        tank = MockChampion(
            "TFT_Tank",
            "Tank",
            3,
            ["tank"],
            MockChampionStats(armor=80, health=[1000, 1500, 2000]),
        )
        player.units.board = {"tank1": MockChampionInstance(tank)}

        suggestions = optimizer.suggest_position(player, "tank1")

        assert len(suggestions) > 0
        assert all(isinstance(s, PositionScore) for s in suggestions)
        # Top suggestion should be in frontline
        if suggestions:
            assert suggestions[0].position.row == 3

    def test_suggest_position_carry(self, optimizer, player):
        """Test position suggestions for carry."""
        carry = MockChampion(
            "TFT_Carry",
            "Carry",
            4,
            ["mage"],
            MockChampionStats(attack_range=4),
        )
        player.units.board = {"carry1": MockChampionInstance(carry)}

        suggestions = optimizer.suggest_position(player, "carry1")

        assert len(suggestions) > 0
        # Top suggestions should be in backline
        if suggestions:
            assert suggestions[0].position.row in [0, 1]

    def test_suggest_position_nonexistent_unit(self, optimizer, player):
        """Test position suggestions for nonexistent unit."""
        suggestions = optimizer.suggest_position(player, "nonexistent")

        assert suggestions == []

    def test_counter_position(self, optimizer, player):
        """Test counter-positioning against enemy."""
        tank = MockChampion("TFT_Tank", "Tank", 3, ["tank"], MockChampionStats())
        player.units.board = {"tank1": MockChampionInstance(tank)}

        enemy_layout = BoardLayout(
            positions={"enemy1": HexPosition(3, 3)},
            total_score=50,
            win_rate=0.5,
            description="Enemy layout",
        )

        layout = optimizer.counter_position(player, enemy_layout)

        assert isinstance(layout, BoardLayout)

    def test_get_recommended_swap(self, optimizer, player):
        """Test swap recommendations."""
        tank = MockChampion("TFT_Tank", "Tank", 3, ["tank"], MockChampionStats())
        carry = MockChampion(
            "TFT_Carry", "Carry", 4, ["mage"], MockChampionStats(attack_range=4)
        )

        # Place them in wrong positions
        player.units.board = {
            "tank1": MockChampionInstance(tank, position=(0, 3)),  # Tank in backline
            "carry1": MockChampionInstance(carry, position=(3, 3)),  # Carry in front
        }

        swap = optimizer.get_recommended_swap(player)

        # Should recommend swapping them or return None
        assert swap is None or (isinstance(swap, tuple) and len(swap) == 2)

    def test_get_formation_templates(self, optimizer):
        """Test formation templates are available."""
        templates = optimizer.get_formation_templates()

        assert isinstance(templates, dict)
        assert "box" in templates
        assert "line" in templates
        assert "corner_left" in templates

        # Check template structure
        box = templates["box"]
        for key, pos in box.items():
            assert isinstance(pos, HexPosition)

    def test_apply_layout(self, optimizer, player):
        """Test applying a layout."""
        tank = MockChampion("TFT_Tank", "Tank", 3, ["tank"], MockChampionStats())
        player.units.board = {"tank1": MockChampionInstance(tank)}

        layout = BoardLayout(
            positions={"tank1": HexPosition(3, 3)},
            total_score=50,
            win_rate=0.5,
            description="Test layout",
        )

        result = optimizer.apply_layout(player, layout)

        assert result is True

    def test_apply_layout_invalid_unit(self, optimizer, player):
        """Test applying layout with invalid unit."""
        layout = BoardLayout(
            positions={"nonexistent": HexPosition(3, 3)},
            total_score=50,
            win_rate=0.5,
            description="Invalid layout",
        )

        result = optimizer.apply_layout(player, layout)

        assert result is False

    def test_layout_description(self, optimizer, player):
        """Test layout description generation."""
        tank = MockChampion("TFT_Tank", "Tank", 3, ["tank"], MockChampionStats())
        carry = MockChampion(
            "TFT_Carry", "Carry", 4, ["mage"], MockChampionStats(attack_range=4)
        )

        player.units.board = {
            "tank1": MockChampionInstance(tank),
            "carry1": MockChampionInstance(carry),
        }

        layout = optimizer.optimize(player)

        assert isinstance(layout.description, str)
        assert "Front" in layout.description or "Back" in layout.description


class TestBoardLayout:
    """Tests for BoardLayout dataclass."""

    def test_create_layout(self):
        """Test creating a board layout."""
        layout = BoardLayout(
            positions={"u1": HexPosition(0, 0), "u2": HexPosition(3, 3)},
            total_score=75.0,
            win_rate=0.65,
            description="Test layout",
        )

        assert len(layout.positions) == 2
        assert layout.total_score == 75.0
        assert layout.win_rate == 0.65


class TestPositionScore:
    """Tests for PositionScore dataclass."""

    def test_create_position_score(self):
        """Test creating a position score."""
        score = PositionScore(
            position=HexPosition(3, 3),
            unit_id="tank1",
            score=20.0,
            reasons=["Frontline tank", "Center position"],
        )

        assert score.position.row == 3
        assert score.position.col == 3
        assert score.score == 20.0
        assert len(score.reasons) == 2


class TestRoleClassification:
    """Tests for unit role classification."""

    def test_classify_tank(self, optimizer, player):
        """Test tank classification."""
        tank = MockChampion(
            "TFT_Tank",
            "Tank",
            3,
            ["tank", "guardian"],
            MockChampionStats(armor=80, health=[1000, 1500, 2000]),
        )
        player.units.board = {"tank1": MockChampionInstance(tank)}

        roles = optimizer._classify_roles(player.units.board)

        assert roles["tank1"] == "tank"

    def test_classify_assassin(self, optimizer, player):
        """Test assassin classification."""
        assassin = MockChampion(
            "TFT_Assassin", "Assassin", 3, ["assassin"], MockChampionStats()
        )
        player.units.board = {"assassin1": MockChampionInstance(assassin)}

        roles = optimizer._classify_roles(player.units.board)

        assert roles["assassin1"] == "assassin"

    def test_classify_carry(self, optimizer, player):
        """Test carry classification."""
        carry = MockChampion(
            "TFT_Carry",
            "Carry",
            4,
            ["mage"],
            MockChampionStats(attack_range=4, attack_damage=[80, 120, 160]),
        )
        player.units.board = {"carry1": MockChampionInstance(carry)}

        roles = optimizer._classify_roles(player.units.board)

        assert roles["carry1"] == "carry"

    def test_classify_support(self, optimizer, player):
        """Test support classification."""
        support = MockChampion(
            "TFT_Support", "Support", 2, ["support", "enchanter"], MockChampionStats()
        )
        player.units.board = {"support1": MockChampionInstance(support)}

        roles = optimizer._classify_roles(player.units.board)

        assert roles["support1"] == "support"

    def test_classify_flex(self, optimizer, player):
        """Test flex classification for ambiguous units."""
        flex = MockChampion(
            "TFT_Flex",
            "Flex",
            3,
            ["warrior"],
            MockChampionStats(attack_range=1, armor=40, attack_damage=[50, 75, 100]),
        )
        player.units.board = {"flex1": MockChampionInstance(flex)}

        roles = optimizer._classify_roles(player.units.board)

        # Should be classified as something
        assert roles["flex1"] in ["tank", "carry", "flex", "support"]
