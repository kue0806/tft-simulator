# TFT Simulator - Phase 2: Champion Pool & Shop System

## Overview
Implement the core champion pool management and shop system that simulates TFT's unit economy.

## Part 1: Champion Pool Manager

Create `src/core/champion_pool.py`:

```python
from typing import Optional
from src.data.models.champion import Champion
from src.core.constants import POOL_SIZE

class ChampionPool:
    """
    Manages the shared champion pool for all players.
    In TFT, all players draw from the same pool.
    """
    
    def __init__(self, champions: list[Champion], num_players: int = 8):
        """
        Initialize pool with all champions.
        Each champion has POOL_SIZE[cost] copies available.
        """
        self.pool: dict[str, int] = {}  # champion_id -> remaining count
        self.champions: dict[str, Champion] = {}  # champion_id -> Champion
        self._initialize_pool(champions)
    
    def _initialize_pool(self, champions: list[Champion]) -> None:
        """Set up initial pool counts based on champion costs."""
        pass
    
    def take(self, champion_id: str, count: int = 1) -> int:
        """
        Remove champions from pool (when purchased).
        Returns actual count taken (may be less if pool depleted).
        """
        pass
    
    def return_champion(self, champion_id: str, count: int = 1) -> None:
        """Return champions to pool (when sold or player eliminated)."""
        pass
    
    def get_available(self, champion_id: str) -> int:
        """Get remaining count of a specific champion."""
        pass
    
    def get_probability(self, champion_id: str, level: int) -> float:
        """
        Calculate probability of seeing this champion in shop.
        Considers: level odds, remaining pool, total pool for that cost.
        """
        pass
    
    def get_champions_by_cost(self, cost: int) -> list[Champion]:
        """Get all champions of a specific cost tier."""
        pass
    
    def get_total_available_by_cost(self, cost: int) -> int:
        """Get total remaining champions of a cost tier."""
        pass
```

### Key Mechanics to Implement

1. **Pool Depletion**: When champions are purchased, they're removed from the shared pool
2. **Contested Units**: If multiple players want the same champion, pool depletes faster
3. **Return on Sell**: Selling a champion returns it to the pool immediately
4. **Player Elimination**: When a player is eliminated, all their champions return to pool

## Part 2: Shop System

Create `src/core/shop.py`:

```python
from typing import Optional
from src.core.champion_pool import ChampionPool
from src.core.constants import SHOP_ODDS, REROLL_COST

class Shop:
    """
    Manages a player's shop with 5 slots.
    """
    SHOP_SIZE = 5
    
    def __init__(self, pool: ChampionPool, player_level: int = 1):
        self.pool = pool
        self.level = player_level
        self.slots: list[Optional[Champion]] = [None] * self.SHOP_SIZE
        self.locked = False  # Shop lock feature
    
    def refresh(self) -> list[Champion]:
        """
        Refresh all shop slots based on current level odds.
        Returns the new shop contents.
        
        Algorithm:
        1. Return any unbought champions to pool
        2. For each slot:
           a. Roll cost tier based on level odds
           b. Pick random available champion of that cost
           c. Remove from pool temporarily
        """
        pass
    
    def _roll_cost_tier(self) -> int:
        """
        Roll which cost tier to show based on level odds.
        Uses weighted random selection.
        """
        pass
    
    def _pick_champion_of_cost(self, cost: int) -> Optional[Champion]:
        """
        Pick a random available champion of given cost.
        Weighted by remaining pool count.
        """
        pass
    
    def purchase(self, slot_index: int) -> Optional[Champion]:
        """
        Purchase champion from shop slot.
        Returns the champion if successful, None if slot empty.
        """
        pass
    
    def toggle_lock(self) -> bool:
        """Toggle shop lock. Returns new lock state."""
        pass
    
    def set_level(self, level: int) -> None:
        """Update player level (affects shop odds)."""
        pass
    
    def get_odds(self) -> list[float]:
        """Get current shop odds based on level."""
        pass
```

## Part 3: Player Bench & Board

Create `src/core/player_units.py`:

```python
class PlayerUnits:
    """
    Manages a player's owned champions (bench + board).
    """
    BENCH_SIZE = 9
    BOARD_SIZE = 28  # 4x7 hexagonal grid
    
    def __init__(self):
        self.bench: list[Optional[ChampionInstance]] = [None] * self.BENCH_SIZE
        self.board: dict[tuple[int, int], ChampionInstance] = {}  # (x, y) -> unit
        self.champion_counts: dict[str, int] = {}  # For tracking 3-star progress
    
    def add_to_bench(self, champion: Champion) -> bool:
        """
        Add champion to first empty bench slot.
        Returns False if bench is full.
        """
        pass
    
    def can_upgrade(self, champion_id: str) -> bool:
        """Check if player has 3 copies to upgrade."""
        pass
    
    def upgrade_champion(self, champion_id: str) -> Optional[ChampionInstance]:
        """
        Combine 3 copies into star-up.
        Returns the upgraded champion.
        """
        pass
    
    def sell(self, champion: ChampionInstance) -> int:
        """
        Sell a champion. Returns gold value.
        1-star: cost, 2-star: cost*3, 3-star: cost*9
        """
        pass
    
    def get_total_champions(self) -> int:
        """Count all owned champions."""
        pass
    
    def get_champion_count(self, champion_id: str) -> int:
        """Get count of specific champion owned."""
        pass


class ChampionInstance:
    """
    An instance of a champion owned by a player.
    Tracks star level, items, position, etc.
    """
    def __init__(self, champion: Champion, star_level: int = 1):
        self.champion = champion
        self.star_level = star_level  # 1, 2, or 3
        self.items: list[Item] = []  # Max 3 items
        self.position: Optional[tuple[int, int]] = None
    
    def get_stats(self) -> dict:
        """Calculate current stats based on star level and items."""
        pass
    
    def can_add_item(self) -> bool:
        """Check if champion can hold more items."""
        return len(self.items) < 3
    
    def add_item(self, item: Item) -> bool:
        """Add item if possible."""
        pass
    
    def get_sell_value(self) -> int:
        """Calculate sell value based on cost and star level."""
        pass
```

## Part 4: Probability Calculator

Create `src/core/probability.py`:

```python
class ProbabilityCalculator:
    """
    Calculate various probabilities for decision making.
    """
    
    @staticmethod
    def chance_to_hit(
        champion_id: str,
        pool: ChampionPool,
        level: int,
        copies_needed: int = 1,
        num_rolls: int = 1
    ) -> float:
        """
        Calculate probability of finding X copies in Y rolls.
        Uses binomial distribution.
        """
        pass
    
    @staticmethod
    def expected_rolls_for_copies(
        champion_id: str,
        pool: ChampionPool,
        level: int,
        copies_needed: int
    ) -> float:
        """
        Calculate expected number of rolls to find X copies.
        """
        pass
    
    @staticmethod
    def expected_gold_for_copies(
        champion_id: str,
        pool: ChampionPool,
        level: int,
        copies_needed: int
    ) -> float:
        """
        Calculate expected gold cost to find X copies.
        gold = expected_rolls * 2 (reroll cost)
        """
        pass
    
    @staticmethod
    def two_star_probability(
        champion_id: str,
        pool: ChampionPool,
        level: int,
        current_copies: int,
        gold_budget: int
    ) -> float:
        """
        Probability of hitting 2-star given budget.
        """
        pass
```

## Part 5: Integration & Game State

Create `src/core/game_state.py`:

```python
class PlayerState:
    """Complete state for one player."""
    def __init__(self, player_id: int, pool: ChampionPool):
        self.player_id = player_id
        self.level = 1
        self.xp = 0
        self.gold = 0
        self.health = 100
        self.shop = Shop(pool, self.level)
        self.units = PlayerUnits()
        self.win_streak = 0
        self.loss_streak = 0
    
    def can_afford_reroll(self) -> bool:
        return self.gold >= REROLL_COST
    
    def reroll(self) -> bool:
        """Spend gold to refresh shop."""
        pass
    
    def buy_xp(self) -> bool:
        """Spend 4 gold for 4 XP."""
        pass
    
    def level_up_check(self) -> bool:
        """Check and process level up."""
        pass


class GameState:
    """
    Complete game state for simulation.
    """
    def __init__(self, num_players: int = 8):
        self.stage = "1-1"
        self.round = 0
        self.pool = ChampionPool(load_all_champions())
        self.players = [PlayerState(i, self.pool) for i in range(num_players)]
        self.eliminated: list[int] = []
    
    def eliminate_player(self, player_id: int) -> None:
        """
        Handle player elimination.
        Returns all their champions to pool.
        """
        pass
    
    def advance_round(self) -> None:
        """Process end of round."""
        pass
```

## Part 6: Tests

Create comprehensive tests in `tests/test_champion_pool.py` and `tests/test_shop.py`:

```python
# Test cases to implement:

# Champion Pool Tests
def test_pool_initialization():
    """Verify correct initial counts for each cost tier."""
    pass

def test_take_champion():
    """Taking reduces pool count."""
    pass

def test_return_champion():
    """Returning increases pool count."""
    pass

def test_pool_depletion():
    """Cannot take more than available."""
    pass

def test_probability_calculation():
    """Verify probability math is correct."""
    pass

# Shop Tests
def test_shop_refresh():
    """Shop shows 5 champions based on level odds."""
    pass

def test_shop_level_odds():
    """Higher level = higher cost champions appear."""
    pass

def test_shop_purchase():
    """Purchasing removes from slot and pool."""
    pass

def test_shop_lock():
    """Locked shop doesn't refresh."""
    pass

# Integration Tests
def test_buy_and_sell_flow():
    """Complete buy/sell cycle."""
    pass

def test_upgrade_to_2_star():
    """3 copies combine into 2-star."""
    pass

def test_contested_champion():
    """Multiple players buying same champion depletes pool."""
    pass
```

## Expected Output

1. `src/core/champion_pool.py` - Pool management
2. `src/core/shop.py` - Shop system
3. `src/core/player_units.py` - Bench/board management
4. `src/core/probability.py` - Probability calculations
5. `src/core/game_state.py` - Game state integration
6. All tests passing

## Verification

After implementation, verify with:

```python
# Quick sanity check
pool = ChampionPool(load_all_champions())
shop = Shop(pool, level=5)

# Check initial pool
assert pool.get_available("jinx") == 10  # 4-cost = 10 copies

# Test shop refresh
shop.refresh()
assert len([s for s in shop.slots if s]) == 5

# Test purchase
champ = shop.purchase(0)
assert pool.get_available(champ.id) == original - 1

# Test probability
prob = pool.get_probability("jinx", level=7)
assert 0 < prob < 1
```

## Priority

1. **Must**: ChampionPool with take/return
2. **Must**: Shop with refresh/purchase
3. **Must**: PlayerUnits with upgrade logic
4. **Should**: Probability calculator
5. **Should**: Full GameState integration
6. **Must**: Tests for all core functionality
