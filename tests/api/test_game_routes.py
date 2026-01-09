"""Tests for game API routes."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


class TestRootEndpoints:
    """Tests for root endpoints."""

    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["name"] == "TFT Simulator API"

    def test_health(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestGameRoutes:
    """Tests for game routes."""

    def test_create_game(self):
        """Test game creation."""
        response = client.post("/api/game/create", json={"player_count": 8})
        assert response.status_code == 200
        data = response.json()
        assert "game_id" in data
        assert len(data["players"]) == 8

    def test_create_game_with_2_players(self):
        """Test game creation with 2 players."""
        response = client.post("/api/game/create", json={"player_count": 2})
        assert response.status_code == 200
        assert len(response.json()["players"]) == 2

    def test_get_game(self):
        """Test game retrieval."""
        # Create game first
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        response = client.get(f"/api/game/{game_id}")
        assert response.status_code == 200
        assert response.json()["game_id"] == game_id

    def test_get_nonexistent_game(self):
        """Test getting nonexistent game."""
        response = client.get("/api/game/nonexistent")
        assert response.status_code == 404

    def test_get_player(self):
        """Test player retrieval."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        response = client.get(f"/api/game/{game_id}/player/0")
        assert response.status_code == 200
        data = response.json()
        assert data["player_id"] == 0
        assert data["hp"] == 100

    def test_setup_player(self):
        """Test player setup."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        response = client.post(
            f"/api/game/{game_id}/player/0/setup",
            json={"level": 5, "gold": 50, "hp": 80},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["level"] == 5
        assert data["gold"] == 50
        assert data["hp"] == 80

    def test_get_synergies(self):
        """Test synergies retrieval."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        response = client.get(f"/api/game/{game_id}/player/0/synergies")
        assert response.status_code == 200
        data = response.json()
        assert "synergies" in data
        assert "total_active" in data

    def test_next_round(self):
        """Test advancing to next round."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        response = client.post(f"/api/game/{game_id}/next-round")
        assert response.status_code == 200
        data = response.json()
        assert data["stage"] == "1-2"

    def test_delete_game(self):
        """Test game deletion."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        response = client.delete(f"/api/game/{game_id}")
        assert response.status_code == 200

        # Verify game is deleted
        get_response = client.get(f"/api/game/{game_id}")
        assert get_response.status_code == 404


class TestShopRoutes:
    """Tests for shop routes."""

    def test_get_shop(self):
        """Test shop retrieval."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        response = client.get(f"/api/shop/{game_id}/player/0")
        assert response.status_code == 200
        data = response.json()
        assert "slots" in data
        assert len(data["slots"]) == 5
        assert "is_locked" in data

    def test_refresh_shop(self):
        """Test shop refresh."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        # Give player gold first
        client.post(
            f"/api/game/{game_id}/player/0/setup", json={"level": 1, "gold": 50, "hp": 100}
        )

        response = client.post(f"/api/shop/{game_id}/player/0/refresh")
        assert response.status_code == 200

    def test_refresh_shop_no_gold(self):
        """Test shop refresh with insufficient gold."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        # Ensure no gold
        client.post(
            f"/api/game/{game_id}/player/0/setup", json={"level": 1, "gold": 0, "hp": 100}
        )

        response = client.post(f"/api/shop/{game_id}/player/0/refresh")
        assert response.status_code == 400

    def test_toggle_lock(self):
        """Test shop lock toggle."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        response = client.post(f"/api/shop/{game_id}/player/0/lock")
        assert response.status_code == 200
        assert "locked" in response.json()["message"]

    def test_buy_xp(self):
        """Test XP purchase."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        # Give player gold
        client.post(
            f"/api/game/{game_id}/player/0/setup", json={"level": 1, "gold": 10, "hp": 100}
        )

        response = client.post(f"/api/shop/{game_id}/player/0/levelup")
        assert response.status_code == 200

    def test_buy_xp_no_gold(self):
        """Test XP purchase with insufficient gold."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        # No gold
        client.post(
            f"/api/game/{game_id}/player/0/setup", json={"level": 1, "gold": 0, "hp": 100}
        )

        response = client.post(f"/api/shop/{game_id}/player/0/levelup")
        assert response.status_code == 400


class TestCombatRoutes:
    """Tests for combat routes."""

    def test_simulate_combat(self):
        """Test combat simulation."""
        response = client.post(
            "/api/combat/simulate",
            json={
                "team_blue": {
                    "units": [
                        {"champion_id": "TFT_Ahri", "star_level": 2, "items": [], "cost": 2}
                    ]
                },
                "team_red": {
                    "units": [
                        {"champion_id": "TFT_Garen", "star_level": 1, "items": [], "cost": 1}
                    ]
                },
                "iterations": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["iterations"] == 10
        assert "blue_wins" in data
        assert "red_wins" in data
        assert "blue_win_rate" in data

    def test_single_combat(self):
        """Test single combat."""
        response = client.post(
            "/api/combat/single",
            json={
                "team_blue": {"units": [{"champion_id": "TFT_Unit", "cost": 1}]},
                "team_red": {"units": [{"champion_id": "TFT_Unit", "cost": 1}]},
                "iterations": 1,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["winner"] in ["blue", "red"]

    def test_combat_stats(self):
        """Test combat stats retrieval."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        response = client.get(f"/api/combat/stats/{game_id}/player/0")
        assert response.status_code == 200


class TestOptimizerRoutes:
    """Tests for optimizer routes."""

    def test_get_comp_templates(self):
        """Test getting composition templates."""
        response = client.get("/api/optimizer/comp/templates")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:
            assert "name" in data[0]
            assert "core_units" in data[0]

    def test_pick_advice(self):
        """Test pick advice."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        response = client.post(
            f"/api/optimizer/pick/{game_id}",
            json={"player_id": 0},
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert "should_refresh" in data

    def test_rolldown_plan(self):
        """Test rolldown plan."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        # Setup player
        client.post(
            f"/api/game/{game_id}/player/0/setup", json={"level": 7, "gold": 50, "hp": 60}
        )

        response = client.post(
            f"/api/optimizer/rolldown/{game_id}",
            json={"player_id": 0, "target_units": ["TFT_Ahri"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "strategy" in data
        assert "advice" in data

    def test_comp_recommendations(self):
        """Test comp recommendations."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        response = client.post(
            f"/api/optimizer/comp/{game_id}",
            json={"player_id": 0, "top_n": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_pivot_advice(self):
        """Test pivot advice."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        response = client.post(
            f"/api/optimizer/pivot/{game_id}",
            json={"player_id": 0},
        )
        assert response.status_code == 200
        data = response.json()
        assert "should_pivot" in data
        assert "explanation" in data

    def test_optimize_board(self):
        """Test board optimization."""
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]

        response = client.post(
            f"/api/optimizer/board/{game_id}",
            json={"player_id": 0, "iterations": 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert "layout" in data


class TestDataRoutes:
    """Tests for data routes."""

    def test_get_champions(self):
        """Test getting champions."""
        response = client.get("/api/data/champions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_champions_by_cost(self):
        """Test getting champions by cost."""
        response = client.get("/api/data/champions?cost=3")
        assert response.status_code == 200
        data = response.json()
        for champ in data:
            assert champ["cost"] == 3

    def test_get_traits(self):
        """Test getting traits."""
        response = client.get("/api/data/traits")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_items(self):
        """Test getting items."""
        response = client.get("/api/data/items")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_shop_odds(self):
        """Test getting shop odds."""
        response = client.get("/api/data/constants/shop-odds")
        assert response.status_code == 200
        data = response.json()
        assert "1" in data or 1 in data

    def test_get_pool_sizes(self):
        """Test getting pool sizes."""
        response = client.get("/api/data/constants/pool-sizes")
        assert response.status_code == 200
        data = response.json()
        assert "1" in data or 1 in data

    def test_get_level_costs(self):
        """Test getting level costs."""
        response = client.get("/api/data/constants/level-costs")
        assert response.status_code == 200
