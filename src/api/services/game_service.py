"""
Game logic service.
"""

from typing import Dict, Optional, List, Any
import uuid

from src.core.synergy_calculator import SynergyCalculator
from src.data.loaders import load_champions

from ..schemas.game import (
    PlayerSetupRequest,
    PlaceUnitRequest,
    MoveUnitRequest,
    EquipItemRequest,
    GameStateSchema,
    PlayerStateSchema,
    ChampionInstanceSchema,
    SynergySchema,
    ShopSlotSchema,
    ShopStateSchema,
)


class MockChampionPool:
    """Mock champion pool for API usage."""

    def __init__(self):
        self._champions = {c.id: c for c in load_champions()}
        self._available: Dict[str, int] = {}
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize pool sizes based on cost."""
        pool_sizes = {1: 30, 2: 25, 3: 18, 4: 12, 5: 10}
        for champ in self._champions.values():
            self._available[champ.id] = pool_sizes.get(champ.cost, 10)

    def take(self, champion_id: str, count: int = 1) -> int:
        """Take champions from pool."""
        available = self._available.get(champion_id, 0)
        taken = min(available, count)
        self._available[champion_id] = available - taken
        return taken

    def return_champion(self, champion_id: str, count: int = 1):
        """Return champions to pool."""
        if champion_id in self._available:
            self._available[champion_id] += count

    def get_champion(self, champion_id: str):
        """Get champion data."""
        return self._champions.get(champion_id)


class MockShop:
    """Mock shop for API usage."""

    def __init__(self, pool: MockChampionPool, player_level: int = 1):
        self.pool = pool
        self.player_level = player_level
        self.slots: List[Optional[Dict[str, Any]]] = [None] * 5
        self.is_locked = False
        self._refresh()

    def _refresh(self):
        """Refresh shop slots."""
        import random

        champions = list(self.pool._champions.values())
        # Filter by level odds
        odds = self._get_odds()
        for i in range(5):
            # Roll for cost
            roll = random.random()
            cumulative = 0
            selected_cost = 1
            for cost, prob in odds.items():
                cumulative += prob
                if roll <= cumulative:
                    selected_cost = cost
                    break

            # Get random champion of that cost
            cost_champs = [c for c in champions if c.cost == selected_cost]
            if cost_champs:
                champ = random.choice(cost_champs)
                self.slots[i] = {
                    "champion": champ,
                    "is_purchased": False,
                }
            else:
                self.slots[i] = None

    def _get_odds(self) -> Dict[int, float]:
        """Get shop odds based on level."""
        odds_table = {
            1: {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
            2: {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
            3: {1: 0.75, 2: 0.25, 3: 0.0, 4: 0.0, 5: 0.0},
            4: {1: 0.55, 2: 0.30, 3: 0.15, 4: 0.0, 5: 0.0},
            5: {1: 0.45, 2: 0.33, 3: 0.20, 4: 0.02, 5: 0.0},
            6: {1: 0.30, 2: 0.40, 3: 0.25, 4: 0.05, 5: 0.0},
            7: {1: 0.19, 2: 0.35, 3: 0.35, 4: 0.10, 5: 0.01},
            8: {1: 0.18, 2: 0.25, 3: 0.32, 4: 0.22, 5: 0.03},
            9: {1: 0.10, 2: 0.20, 3: 0.25, 4: 0.35, 5: 0.10},
            10: {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.40, 5: 0.25},
        }
        return odds_table.get(self.player_level, odds_table[1])

    def refresh(self):
        """Manually refresh shop."""
        if not self.is_locked:
            self._refresh()

    def toggle_lock(self):
        """Toggle shop lock."""
        self.is_locked = not self.is_locked

    def purchase(self, slot_index: int, player: "MockPlayerState") -> bool:
        """Purchase champion from slot."""
        if slot_index < 0 or slot_index >= 5:
            return False

        slot = self.slots[slot_index]
        if not slot or slot["is_purchased"]:
            return False

        champion = slot["champion"]
        if player.gold < champion.cost:
            return False

        # Deduct gold and mark as purchased
        player.gold -= champion.cost
        slot["is_purchased"] = True

        # Add to player
        player.add_unit(champion)

        return True


class MockChampionInstance:
    """Mock champion instance."""

    def __init__(self, champion, star_level: int = 1):
        self.id = str(uuid.uuid4())[:8]
        self.champion = champion
        self.star_level = star_level
        self.items: List[str] = []
        self.position: Optional[tuple] = None

    def to_schema(self) -> ChampionInstanceSchema:
        """Convert to schema."""
        return ChampionInstanceSchema(
            id=self.id,
            champion_id=self.champion.id,
            name=self.champion.name,
            cost=self.champion.cost,
            star_level=self.star_level,
            items=self.items,
            traits=self.champion.traits,
            position=(
                {"row": self.position[0], "col": self.position[1]}
                if self.position
                else None
            ),
        )


class MockPlayerUnits:
    """Mock player units container."""

    def __init__(self):
        self.board: Dict[str, MockChampionInstance] = {}
        self.bench: List[Optional[MockChampionInstance]] = [None] * 9

    def add_to_bench(self, instance: MockChampionInstance) -> bool:
        """Add unit to bench."""
        for i in range(9):
            if self.bench[i] is None:
                self.bench[i] = instance
                return True
        return False

    def place_on_board(self, unit_id: str, position: tuple) -> bool:
        """Place unit on board from bench."""
        # Find unit on bench
        for i, unit in enumerate(self.bench):
            if unit and unit.id == unit_id:
                unit.position = position
                self.board[unit_id] = unit
                self.bench[i] = None
                return True
        return False

    def move_unit(self, unit_id: str, new_position: tuple) -> bool:
        """Move unit to new position."""
        if unit_id in self.board:
            self.board[unit_id].position = new_position
            return True
        return False

    def equip_item(self, unit_id: str, item_id: str) -> bool:
        """Equip item to unit."""
        if unit_id in self.board:
            unit = self.board[unit_id]
            if len(unit.items) < 3:
                unit.items.append(item_id)
                return True
        return False

    def sell_unit(self, unit_id: str, pool: MockChampionPool) -> int:
        """Sell unit and return gold."""
        # Check board
        if unit_id in self.board:
            unit = self.board.pop(unit_id)
            sell_value = unit.champion.cost * (3 ** (unit.star_level - 1))
            pool.return_champion(unit.champion.id, 3 ** (unit.star_level - 1))
            return sell_value

        # Check bench
        for i, unit in enumerate(self.bench):
            if unit and unit.id == unit_id:
                self.bench[i] = None
                sell_value = unit.champion.cost * (3 ** (unit.star_level - 1))
                pool.return_champion(unit.champion.id, 3 ** (unit.star_level - 1))
                return sell_value

        return 0


class MockPlayerState:
    """Mock player state."""

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.hp = 100
        self.gold = 0
        self.level = 1
        self.xp = 0
        self.streak = 0
        self.units = MockPlayerUnits()
        self._pending_upgrades: Dict[str, List[MockChampionInstance]] = {}

    def add_unit(self, champion):
        """Add a champion unit."""
        # Check for upgrade potential
        champ_id = champion.id
        if champ_id not in self._pending_upgrades:
            self._pending_upgrades[champ_id] = []

        # Count existing copies
        existing = self._pending_upgrades[champ_id]
        existing.append(MockChampionInstance(champion))

        if len(existing) >= 3:
            # Upgrade to 2-star
            for inst in existing[:3]:
                # Remove from bench if present
                for i, bench_unit in enumerate(self.units.bench):
                    if bench_unit and bench_unit.id == inst.id:
                        self.units.bench[i] = None
                        break
                # Remove from board if present
                if inst.id in self.units.board:
                    del self.units.board[inst.id]

            # Create 2-star
            upgraded = MockChampionInstance(champion, star_level=2)
            self._pending_upgrades[champ_id] = existing[3:]
            self._pending_upgrades[champ_id].append(upgraded)

            # Check for 3-star
            if len(self._pending_upgrades[champ_id]) >= 3:
                # Similar logic for 3-star
                pass

            self.units.add_to_bench(upgraded)
        else:
            # Just add to bench
            self.units.add_to_bench(existing[-1])

    def add_xp(self, amount: int):
        """Add XP and level up if needed."""
        xp_to_level = {
            1: 2,
            2: 2,
            3: 6,
            4: 10,
            5: 20,
            6: 36,
            7: 56,
            8: 80,
            9: 100,
        }

        self.xp += amount
        while self.level < 10:
            needed = xp_to_level.get(self.level, 999)
            if self.xp >= needed:
                self.xp -= needed
                self.level += 1
            else:
                break

    def to_schema(self) -> PlayerStateSchema:
        """Convert to schema."""
        return PlayerStateSchema(
            player_id=self.player_id,
            hp=self.hp,
            gold=self.gold,
            level=self.level,
            xp=self.xp,
            streak=self.streak,
            board=[u.to_schema() for u in self.units.board.values()],
            bench=[u.to_schema() if u else None for u in self.units.bench],
        )


class MockStageManager:
    """Mock stage manager."""

    def __init__(self):
        self.stage = 1
        self.round = 1

    def get_stage_string(self) -> str:
        """Get stage string."""
        return f"{self.stage}-{self.round}"

    def get_round_type(self) -> str:
        """Get round type."""
        if self.stage == 1:
            return "pve"
        if self.round == 4 and self.stage > 1:
            return "carousel"
        return "pvp"

    def advance_round(self):
        """Advance to next round."""
        self.round += 1
        if self.round > 7:
            self.round = 1
            self.stage += 1


class MockGameState:
    """Mock game state."""

    def __init__(self, game_id: str, player_count: int):
        self.game_id = game_id
        self.player_count = player_count
        self.players = [MockPlayerState(i) for i in range(player_count)]
        self.stage_manager = MockStageManager()

    def calculate_income(self, player: MockPlayerState) -> int:
        """Calculate player income."""
        base = 5
        interest = min(player.gold // 10, 5)
        streak_bonus = min(abs(player.streak), 3)
        return base + interest + streak_bonus

    def to_schema(self) -> GameStateSchema:
        """Convert to schema."""
        return GameStateSchema(
            game_id=self.game_id,
            stage=self.stage_manager.get_stage_string(),
            round_type=self.stage_manager.get_round_type(),
            players=[p.to_schema() for p in self.players],
        )


class GameService:
    """Game management service."""

    def __init__(self):
        self._games: Dict[str, MockGameState] = {}
        self._shops: Dict[str, Dict[int, MockShop]] = {}
        self._pools: Dict[str, MockChampionPool] = {}
        self.synergy_calc = SynergyCalculator()

    def create_game(self, player_count: int) -> GameStateSchema:
        """Create a new game."""
        game_id = str(uuid.uuid4())

        # Create champion pool
        pool = MockChampionPool()
        self._pools[game_id] = pool

        # Create game state
        game = MockGameState(game_id=game_id, player_count=player_count)
        self._games[game_id] = game

        # Create shops for each player
        self._shops[game_id] = {}
        for i in range(player_count):
            shop = MockShop(pool=pool, player_level=1)
            self._shops[game_id][i] = shop

        return game.to_schema()

    def get_game(self, game_id: str) -> Optional[GameStateSchema]:
        """Get game state."""
        game = self._games.get(game_id)
        return game.to_schema() if game else None

    def get_player(self, game_id: str, player_id: int) -> Optional[PlayerStateSchema]:
        """Get player state."""
        game = self._games.get(game_id)
        if game and 0 <= player_id < len(game.players):
            return game.players[player_id].to_schema()
        return None

    def get_player_raw(self, game_id: str, player_id: int) -> Optional[MockPlayerState]:
        """Get raw player object."""
        game = self._games.get(game_id)
        if game and 0 <= player_id < len(game.players):
            return game.players[player_id]
        return None

    def setup_player(
        self, game_id: str, player_id: int, request: PlayerSetupRequest
    ) -> PlayerStateSchema:
        """Setup player."""
        player = self.get_player_raw(game_id, player_id)
        if not player:
            raise ValueError("Player not found")

        player.level = request.level
        player.gold = request.gold
        player.hp = request.hp

        # Update shop level
        if game_id in self._shops and player_id in self._shops[game_id]:
            self._shops[game_id][player_id].player_level = request.level

        return player.to_schema()

    def place_unit(
        self, game_id: str, player_id: int, request: PlaceUnitRequest
    ) -> bool:
        """Place unit on board."""
        player = self.get_player_raw(game_id, player_id)
        if not player:
            return False

        position = (request.position["row"], request.position["col"])

        # Find unit by champion_id on bench
        for unit in player.units.bench:
            if unit and unit.champion.id == request.champion_id:
                return player.units.place_on_board(unit.id, position)

        return False

    def move_unit(self, game_id: str, player_id: int, request: MoveUnitRequest) -> bool:
        """Move unit."""
        player = self.get_player_raw(game_id, player_id)
        if not player:
            return False

        new_position = (request.new_position["row"], request.new_position["col"])
        return player.units.move_unit(request.unit_id, new_position)

    def equip_item(
        self, game_id: str, player_id: int, request: EquipItemRequest
    ) -> bool:
        """Equip item."""
        player = self.get_player_raw(game_id, player_id)
        if not player:
            return False

        return player.units.equip_item(request.unit_id, request.item_id)

    def get_player_synergies(self, game_id: str, player_id: int) -> dict:
        """Get player synergies."""
        player = self.get_player_raw(game_id, player_id)
        if not player:
            return {"synergies": [], "total_active": 0}

        # Build champion list for synergy calculation
        board_champions = []
        for unit in player.units.board.values():
            board_champions.append(unit.champion)

        # Calculate synergies
        synergy_result = self.synergy_calc.calculate_synergies(board_champions)

        synergies = []
        for trait_id, data in synergy_result.items():
            synergies.append(
                SynergySchema(
                    trait_id=trait_id,
                    name=data.get("name", trait_id),
                    count=data.count,
                    breakpoints=data.breakpoints,
                    active_breakpoint=data.active_breakpoint,
                    is_active=data.is_active,
                    style=data.style,
                )
            )

        return {
            "synergies": synergies,
            "total_active": sum(1 for s in synergies if s.is_active),
        }

    def get_shop(self, game_id: str, player_id: int) -> ShopStateSchema:
        """Get shop state."""
        if game_id not in self._shops or player_id not in self._shops[game_id]:
            return ShopStateSchema(slots=[], is_locked=False, refresh_cost=2)

        shop = self._shops[game_id][player_id]
        slots = []
        for i, slot in enumerate(shop.slots):
            if slot and not slot["is_purchased"]:
                champ = slot["champion"]
                slots.append(
                    ShopSlotSchema(
                        index=i,
                        champion_id=champ.id,
                        champion_name=champ.name,
                        cost=champ.cost,
                        is_purchased=False,
                    )
                )
            else:
                slots.append(
                    ShopSlotSchema(
                        index=i,
                        champion_id=None,
                        champion_name=None,
                        cost=None,
                        is_purchased=True,
                    )
                )

        return ShopStateSchema(
            slots=slots,
            is_locked=shop.is_locked,
            refresh_cost=2,
        )

    def refresh_shop(self, game_id: str, player_id: int) -> ShopStateSchema:
        """Refresh shop."""
        player = self.get_player_raw(game_id, player_id)
        shop = self._shops.get(game_id, {}).get(player_id)

        if not player or not shop:
            raise ValueError("Game or player not found")

        if player.gold < 2:
            raise ValueError("Not enough gold")

        player.gold -= 2
        shop.refresh()

        return self.get_shop(game_id, player_id)

    def buy_champion(self, game_id: str, player_id: int, slot_index: int) -> bool:
        """Buy champion from shop."""
        player = self.get_player_raw(game_id, player_id)
        shop = self._shops.get(game_id, {}).get(player_id)

        if not player or not shop:
            return False

        return shop.purchase(slot_index, player)

    def sell_unit(self, game_id: str, player_id: int, unit_id: str) -> bool:
        """Sell unit."""
        player = self.get_player_raw(game_id, player_id)
        pool = self._pools.get(game_id)

        if not player or not pool:
            return False

        gold = player.units.sell_unit(unit_id, pool)
        if gold > 0:
            player.gold += gold
            return True
        return False

    def toggle_shop_lock(self, game_id: str, player_id: int) -> bool:
        """Toggle shop lock."""
        shop = self._shops.get(game_id, {}).get(player_id)
        if not shop:
            return False

        shop.toggle_lock()
        return shop.is_locked

    def buy_xp(self, game_id: str, player_id: int) -> bool:
        """Buy XP."""
        player = self.get_player_raw(game_id, player_id)
        if not player or player.gold < 4:
            return False

        player.gold -= 4
        player.add_xp(4)

        # Update shop level
        shop = self._shops.get(game_id, {}).get(player_id)
        if shop:
            shop.player_level = player.level

        return True

    def advance_round(self, game_id: str) -> GameStateSchema:
        """Advance to next round."""
        game = self._games.get(game_id)
        if not game:
            raise ValueError("Game not found")

        game.stage_manager.advance_round()

        # Give income and refresh shops
        for i, player in enumerate(game.players):
            income = game.calculate_income(player)
            player.gold += income

            shop = self._shops.get(game_id, {}).get(i)
            if shop and not shop.is_locked:
                shop.refresh()

        return game.to_schema()

    def delete_game(self, game_id: str) -> None:
        """Delete game."""
        self._games.pop(game_id, None)
        self._shops.pop(game_id, None)
        self._pools.pop(game_id, None)
