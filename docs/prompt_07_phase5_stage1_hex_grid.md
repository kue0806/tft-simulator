# Phase 5 - Stage 5.1: 헥스 그리드 & 전투 기본 구조

## 목표
TFT 전투 시뮬레이션의 기반이 되는 헥스 그리드 시스템과 전투 유닛 구조를 구현합니다.

## TFT 보드 구조

```
TFT 보드는 4x7 헥사곤 그리드 (플레이어당)
전투 시에는 두 보드가 마주보며 8x7 형태가 됨

플레이어 1 보드 (row 0-3):
    [0,0] [0,1] [0,2] [0,3] [0,4] [0,5] [0,6]
      [1,0] [1,1] [1,2] [1,3] [1,4] [1,5] [1,6]
    [2,0] [2,1] [2,2] [2,3] [2,4] [2,5] [2,6]
      [3,0] [3,1] [3,2] [3,3] [3,4] [3,5] [3,6]
--- 중앙선 ---
      [4,0] [4,1] [4,2] [4,3] [4,4] [4,5] [4,6]
    [5,0] [5,1] [5,2] [5,3] [5,4] [5,5] [5,6]
      [6,0] [6,1] [6,2] [6,3] [6,4] [6,5] [6,6]
    [7,0] [7,1] [7,2] [7,3] [7,4] [7,5] [7,6]
플레이어 2 보드 (row 4-7, 미러링됨)

홀수 행은 0.5칸 오프셋 (offset coordinates)
```

## 구현할 파일들

### 1. `src/combat/hex_grid.py`

```python
"""
헥스 그리드 시스템
- Offset coordinates 사용 (odd-r layout)
- 거리 계산, 인접 헥스, 경로 탐색 지원
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
from enum import Enum

class Team(Enum):
    """팀 구분"""
    BLUE = "blue"   # 플레이어 1 (하단)
    RED = "red"     # 플레이어 2 (상단)


@dataclass(frozen=True)
class HexPosition:
    """
    헥스 좌표 (offset coordinates, odd-r layout)
    
    Attributes:
        row: 행 (0-7, 0-3은 BLUE팀, 4-7은 RED팀)
        col: 열 (0-6)
    """
    row: int
    col: int
    
    def to_cube(self) -> Tuple[int, int, int]:
        """Cube coordinates로 변환 (거리 계산용)"""
        # odd-r offset to cube
        x = self.col - (self.row - (self.row & 1)) // 2
        z = self.row
        y = -x - z
        return (x, y, z)
    
    @classmethod
    def from_cube(cls, x: int, y: int, z: int) -> 'HexPosition':
        """Cube coordinates에서 변환"""
        col = x + (z - (z & 1)) // 2
        row = z
        return cls(row, col)
    
    def distance_to(self, other: 'HexPosition') -> int:
        """다른 헥스까지의 거리 (헥스 단위)"""
        ax, ay, az = self.to_cube()
        bx, by, bz = other.to_cube()
        return (abs(ax - bx) + abs(ay - by) + abs(az - bz)) // 2
    
    def get_neighbors(self) -> List['HexPosition']:
        """인접한 6개 헥스 반환 (보드 범위 무시)"""
        # odd-r layout의 6방향
        if self.row % 2 == 0:  # 짝수 행
            directions = [
                (-1, 0), (-1, -1),  # 상단
                (0, -1), (0, 1),    # 좌우
                (1, 0), (1, -1)     # 하단
            ]
        else:  # 홀수 행
            directions = [
                (-1, 1), (-1, 0),   # 상단
                (0, -1), (0, 1),    # 좌우
                (1, 1), (1, 0)      # 하단
            ]
        
        neighbors = []
        for dr, dc in directions:
            neighbors.append(HexPosition(self.row + dr, self.col + dc))
        return neighbors
    
    def is_valid(self, rows: int = 8, cols: int = 7) -> bool:
        """보드 범위 내인지 확인"""
        return 0 <= self.row < rows and 0 <= self.col < cols
    
    def __repr__(self) -> str:
        return f"Hex({self.row}, {self.col})"


class HexGrid:
    """
    TFT 전투 보드 (8x7 헥스 그리드)
    
    - row 0-3: BLUE 팀 영역
    - row 4-7: RED 팀 영역 (상대방 보드가 미러링되어 배치)
    """
    
    ROWS = 8
    COLS = 7
    BLUE_ROWS = range(0, 4)  # 플레이어 1 영역
    RED_ROWS = range(4, 8)   # 플레이어 2 영역
    
    def __init__(self):
        # 각 헥스에 있는 유닛 ID (또는 None)
        self._grid: dict[HexPosition, Optional[str]] = {}
        self._unit_positions: dict[str, HexPosition] = {}
    
    def place_unit(self, unit_id: str, position: HexPosition) -> bool:
        """
        유닛을 헥스에 배치
        
        Returns:
            True if 성공, False if 이미 유닛이 있음
        """
        if not position.is_valid(self.ROWS, self.COLS):
            raise ValueError(f"Invalid position: {position}")
        
        if self._grid.get(position) is not None:
            return False
        
        # 기존 위치에서 제거
        if unit_id in self._unit_positions:
            old_pos = self._unit_positions[unit_id]
            self._grid[old_pos] = None
        
        self._grid[position] = unit_id
        self._unit_positions[unit_id] = position
        return True
    
    def remove_unit(self, unit_id: str) -> Optional[HexPosition]:
        """유닛 제거, 이전 위치 반환"""
        if unit_id not in self._unit_positions:
            return None
        
        position = self._unit_positions.pop(unit_id)
        self._grid[position] = None
        return position
    
    def move_unit(self, unit_id: str, new_position: HexPosition) -> bool:
        """유닛 이동"""
        if not new_position.is_valid(self.ROWS, self.COLS):
            return False
        
        if self._grid.get(new_position) is not None:
            return False
        
        if unit_id not in self._unit_positions:
            return False
        
        old_position = self._unit_positions[unit_id]
        self._grid[old_position] = None
        self._grid[new_position] = unit_id
        self._unit_positions[unit_id] = new_position
        return True
    
    def get_unit_at(self, position: HexPosition) -> Optional[str]:
        """해당 위치의 유닛 ID 반환"""
        return self._grid.get(position)
    
    def get_unit_position(self, unit_id: str) -> Optional[HexPosition]:
        """유닛의 현재 위치 반환"""
        return self._unit_positions.get(unit_id)
    
    def get_valid_neighbors(self, position: HexPosition) -> List[HexPosition]:
        """보드 내의 유효한 인접 헥스들"""
        return [n for n in position.get_neighbors() 
                if n.is_valid(self.ROWS, self.COLS)]
    
    def get_empty_neighbors(self, position: HexPosition) -> List[HexPosition]:
        """비어있는 인접 헥스들"""
        return [n for n in self.get_valid_neighbors(position) 
                if self._grid.get(n) is None]
    
    def get_units_in_range(self, position: HexPosition, range_: int) -> List[str]:
        """특정 범위 내의 모든 유닛 ID"""
        units = []
        for uid, pos in self._unit_positions.items():
            if position.distance_to(pos) <= range_:
                units.append(uid)
        return units
    
    def get_all_units(self) -> dict[str, HexPosition]:
        """모든 유닛과 위치"""
        return self._unit_positions.copy()
    
    def clear(self):
        """그리드 초기화"""
        self._grid.clear()
        self._unit_positions.clear()
    
    def get_team_for_position(self, position: HexPosition) -> Team:
        """해당 위치가 속한 팀 영역"""
        if position.row in self.BLUE_ROWS:
            return Team.BLUE
        return Team.RED
    
    @staticmethod
    def mirror_position(position: HexPosition) -> HexPosition:
        """
        상대방 보드에서의 미러 위치 계산
        (상대 보드를 우리 보드에 배치할 때 사용)
        
        예: 상대의 (0, 3) -> 우리 보드의 (7, 3)
            상대의 (3, 0) -> 우리 보드의 (4, 6)
        """
        # 행은 뒤집고 (0->7, 1->6, 2->5, 3->4)
        # 열도 뒤집음 (0->6, 1->5, ...)
        new_row = 7 - position.row
        new_col = 6 - position.col
        return HexPosition(new_row, new_col)
```

### 2. `src/combat/combat_unit.py`

```python
"""
전투 중 유닛 상태 관리
- 실시간 HP, 마나, 상태 효과 등 추적
- ChampionInstance와 연동
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum, auto
import uuid

if TYPE_CHECKING:
    from src.data.models.champion import Champion
    from src.core.player_units import ChampionInstance


class UnitState(Enum):
    """유닛 상태"""
    IDLE = auto()       # 대기
    MOVING = auto()     # 이동 중
    ATTACKING = auto()  # 공격 중
    CASTING = auto()    # 스킬 시전 중
    STUNNED = auto()    # 기절
    DEAD = auto()       # 사망


@dataclass
class CombatStats:
    """
    전투 중 실시간 스탯
    (기본 스탯 + 아이템 + 시너지에서 계산된 최종 값)
    """
    # 기본 스탯
    max_hp: float
    current_hp: float
    attack_damage: float
    ability_power: float
    armor: float
    magic_resist: float
    attack_speed: float  # 초당 공격 횟수
    crit_chance: float   # 0.0 ~ 1.0
    crit_damage: float   # 기본 1.4 (140%)
    
    # 마나
    max_mana: float
    current_mana: float
    starting_mana: float
    
    # 기타
    attack_range: int    # 헥스 단위 (1 = 근접, 4 = 원거리 등)
    dodge_chance: float  # 회피율
    
    # 흡혈/회복
    omnivamp: float      # 모든 데미지의 일부를 회복
    
    # 데미지 증폭/감소
    damage_amp: float    # 데미지 증폭 (1.0 = 100%)
    damage_reduction: float  # 데미지 감소 (0.0 = 감소 없음)


@dataclass
class CombatUnit:
    """
    전투 중인 유닛
    
    ChampionInstance의 전투 버전으로, 실시간 상태를 추적합니다.
    """
    
    # 식별자
    id: str
    name: str
    champion_id: str
    star_level: int
    team: 'Team'  # from hex_grid import Team
    
    # 전투 스탯 (초기화 시 계산됨)
    stats: CombatStats
    
    # 상태
    state: UnitState = UnitState.IDLE
    
    # 타겟팅
    current_target_id: Optional[str] = None
    
    # 공격 타이머 (초 단위)
    attack_cooldown: float = 0.0
    
    # 스킬 관련
    is_casting: bool = False
    cast_time_remaining: float = 0.0
    
    # 이동 관련
    move_progress: float = 0.0  # 0.0 ~ 1.0, 다음 헥스까지 진행률
    move_target: Optional['HexPosition'] = None
    
    # 전투 통계
    total_damage_dealt: float = 0.0
    total_damage_taken: float = 0.0
    total_healing_done: float = 0.0
    kills: int = 0
    
    # 상태 효과 (Stage 5.5에서 구현)
    status_effects: List[Any] = field(default_factory=list)
    
    # 원본 참조 (아이템, 특성 정보 접근용)
    source_instance: Optional['ChampionInstance'] = None
    
    @classmethod
    def from_champion_instance(
        cls, 
        instance: 'ChampionInstance', 
        team: 'Team',
        calculated_stats: Dict[str, float]
    ) -> 'CombatUnit':
        """
        ChampionInstance에서 CombatUnit 생성
        
        Args:
            instance: 원본 챔피언 인스턴스
            team: 소속 팀
            calculated_stats: StatCalculator로 계산된 최종 스탯
        """
        champion = instance.champion
        
        # CombatStats 생성
        stats = CombatStats(
            max_hp=calculated_stats.get('hp', champion.stats.hp),
            current_hp=calculated_stats.get('hp', champion.stats.hp),
            attack_damage=calculated_stats.get('attack_damage', champion.stats.attack_damage),
            ability_power=calculated_stats.get('ability_power', 100.0),
            armor=calculated_stats.get('armor', champion.stats.armor),
            magic_resist=calculated_stats.get('magic_resist', champion.stats.magic_resist),
            attack_speed=calculated_stats.get('attack_speed', champion.stats.attack_speed),
            crit_chance=calculated_stats.get('crit_chance', champion.stats.crit_chance),
            crit_damage=calculated_stats.get('crit_damage', champion.stats.crit_damage),
            max_mana=champion.stats.mana,
            current_mana=calculated_stats.get('starting_mana', champion.stats.starting_mana),
            starting_mana=calculated_stats.get('starting_mana', champion.stats.starting_mana),
            attack_range=champion.stats.attack_range,
            dodge_chance=calculated_stats.get('dodge_chance', 0.0),
            omnivamp=calculated_stats.get('omnivamp', 0.0),
            damage_amp=calculated_stats.get('damage_amp', 1.0),
            damage_reduction=calculated_stats.get('damage_reduction', 0.0),
        )
        
        return cls(
            id=str(uuid.uuid4()),
            name=champion.name,
            champion_id=champion.champion_id,
            star_level=instance.star_level,
            team=team,
            stats=stats,
            source_instance=instance,
        )
    
    @property
    def is_alive(self) -> bool:
        """생존 여부"""
        return self.stats.current_hp > 0 and self.state != UnitState.DEAD
    
    @property
    def is_targetable(self) -> bool:
        """타겟 가능 여부"""
        return self.is_alive  # 추후 무적 등 상태 추가
    
    @property
    def can_act(self) -> bool:
        """행동 가능 여부"""
        return (self.is_alive and 
                self.state not in [UnitState.STUNNED, UnitState.DEAD])
    
    @property
    def can_attack(self) -> bool:
        """공격 가능 여부"""
        return self.can_act and self.attack_cooldown <= 0 and not self.is_casting
    
    @property
    def can_cast(self) -> bool:
        """스킬 시전 가능 여부"""
        return (self.can_act and 
                self.stats.current_mana >= self.stats.max_mana and
                not self.is_casting)
    
    @property
    def attack_interval(self) -> float:
        """공격 간격 (초)"""
        if self.stats.attack_speed <= 0:
            return float('inf')
        return 1.0 / self.stats.attack_speed
    
    def take_damage(self, amount: float, damage_type: str = "physical") -> float:
        """
        데미지 받기
        
        Args:
            amount: 데미지 양
            damage_type: "physical", "magical", "true"
            
        Returns:
            실제로 받은 데미지
        """
        if not self.is_alive:
            return 0.0
        
        # 데미지 감소 적용
        effective_damage = amount * (1 - self.stats.damage_reduction)
        
        # 방어력/마저 적용
        if damage_type == "physical":
            # 물리 데미지 감소 = armor / (armor + 100)
            reduction = self.stats.armor / (self.stats.armor + 100)
            effective_damage *= (1 - reduction)
        elif damage_type == "magical":
            # 마법 데미지 감소 = mr / (mr + 100)
            reduction = self.stats.magic_resist / (self.stats.magic_resist + 100)
            effective_damage *= (1 - reduction)
        # true 데미지는 감소 없음
        
        effective_damage = max(0, effective_damage)
        
        self.stats.current_hp -= effective_damage
        self.total_damage_taken += effective_damage
        
        if self.stats.current_hp <= 0:
            self.stats.current_hp = 0
            self.state = UnitState.DEAD
        
        return effective_damage
    
    def heal(self, amount: float) -> float:
        """
        체력 회복
        
        Returns:
            실제로 회복한 양
        """
        if not self.is_alive:
            return 0.0
        
        old_hp = self.stats.current_hp
        self.stats.current_hp = min(
            self.stats.current_hp + amount,
            self.stats.max_hp
        )
        healed = self.stats.current_hp - old_hp
        self.total_healing_done += healed
        return healed
    
    def gain_mana(self, amount: float) -> None:
        """마나 획득"""
        if not self.is_alive:
            return
        self.stats.current_mana = min(
            self.stats.current_mana + amount,
            self.stats.max_mana
        )
    
    def spend_mana(self) -> None:
        """스킬 시전 후 마나 소모"""
        self.stats.current_mana = self.stats.starting_mana
    
    def reset_for_combat(self) -> None:
        """전투 시작 전 상태 초기화"""
        self.stats.current_hp = self.stats.max_hp
        self.stats.current_mana = self.stats.starting_mana
        self.state = UnitState.IDLE
        self.current_target_id = None
        self.attack_cooldown = 0.0
        self.is_casting = False
        self.cast_time_remaining = 0.0
        self.move_progress = 0.0
        self.move_target = None
        self.total_damage_dealt = 0.0
        self.total_damage_taken = 0.0
        self.total_healing_done = 0.0
        self.kills = 0
        self.status_effects.clear()


@dataclass
class CombatResult:
    """전투 결과"""
    winner: Team
    winning_units_remaining: int
    losing_units_remaining: int  # 보통 0
    rounds_taken: int  # 전투에 걸린 틱/라운드 수
    total_damage_to_loser: float  # 패배자가 받는 플레이어 데미지
    
    # 유닛별 통계
    unit_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # {unit_id: {"damage_dealt": x, "damage_taken": y, "kills": z}}
```

### 3. `src/combat/__init__.py`

```python
"""Combat simulation module"""

from .hex_grid import HexPosition, HexGrid, Team
from .combat_unit import CombatUnit, CombatStats, CombatResult, UnitState

__all__ = [
    'HexPosition',
    'HexGrid', 
    'Team',
    'CombatUnit',
    'CombatStats',
    'CombatResult',
    'UnitState',
]
```

### 4. `tests/test_hex_grid.py`

```python
"""HexGrid 및 HexPosition 테스트"""

import pytest
from src.combat.hex_grid import HexPosition, HexGrid, Team


class TestHexPosition:
    """HexPosition 테스트"""
    
    def test_create_position(self):
        """위치 생성"""
        pos = HexPosition(0, 0)
        assert pos.row == 0
        assert pos.col == 0
    
    def test_position_equality(self):
        """위치 동등성"""
        pos1 = HexPosition(1, 2)
        pos2 = HexPosition(1, 2)
        pos3 = HexPosition(1, 3)
        assert pos1 == pos2
        assert pos1 != pos3
    
    def test_position_hashable(self):
        """위치를 딕셔너리 키로 사용"""
        pos1 = HexPosition(1, 2)
        pos2 = HexPosition(1, 2)
        d = {pos1: "unit1"}
        assert d[pos2] == "unit1"
    
    def test_distance_same_position(self):
        """같은 위치 거리 = 0"""
        pos = HexPosition(3, 3)
        assert pos.distance_to(pos) == 0
    
    def test_distance_adjacent(self):
        """인접 헥스 거리 = 1"""
        pos = HexPosition(2, 2)
        for neighbor in pos.get_neighbors():
            assert pos.distance_to(neighbor) == 1
    
    def test_distance_two_hexes(self):
        """2칸 거리"""
        pos1 = HexPosition(0, 0)
        pos2 = HexPosition(2, 0)
        assert pos1.distance_to(pos2) == 2
    
    def test_get_neighbors_count(self):
        """인접 헥스는 6개"""
        pos = HexPosition(3, 3)  # 중앙 위치
        neighbors = pos.get_neighbors()
        assert len(neighbors) == 6
    
    def test_get_neighbors_even_row(self):
        """짝수 행의 인접 헥스"""
        pos = HexPosition(2, 3)
        neighbors = pos.get_neighbors()
        expected = [
            HexPosition(1, 3), HexPosition(1, 2),  # 상단
            HexPosition(2, 2), HexPosition(2, 4),  # 좌우
            HexPosition(3, 3), HexPosition(3, 2),  # 하단
        ]
        assert set(neighbors) == set(expected)
    
    def test_get_neighbors_odd_row(self):
        """홀수 행의 인접 헥스"""
        pos = HexPosition(3, 3)
        neighbors = pos.get_neighbors()
        expected = [
            HexPosition(2, 4), HexPosition(2, 3),  # 상단
            HexPosition(3, 2), HexPosition(3, 4),  # 좌우
            HexPosition(4, 4), HexPosition(4, 3),  # 하단
        ]
        assert set(neighbors) == set(expected)
    
    def test_is_valid(self):
        """유효 범위 체크"""
        assert HexPosition(0, 0).is_valid()
        assert HexPosition(7, 6).is_valid()
        assert not HexPosition(-1, 0).is_valid()
        assert not HexPosition(8, 0).is_valid()
        assert not HexPosition(0, 7).is_valid()
    
    def test_cube_conversion_roundtrip(self):
        """Cube 좌표 변환 왕복"""
        for row in range(8):
            for col in range(7):
                pos = HexPosition(row, col)
                cube = pos.to_cube()
                back = HexPosition.from_cube(*cube)
                assert pos == back


class TestHexGrid:
    """HexGrid 테스트"""
    
    def test_create_grid(self):
        """그리드 생성"""
        grid = HexGrid()
        assert grid.ROWS == 8
        assert grid.COLS == 7
    
    def test_place_unit(self):
        """유닛 배치"""
        grid = HexGrid()
        pos = HexPosition(0, 0)
        assert grid.place_unit("unit1", pos)
        assert grid.get_unit_at(pos) == "unit1"
        assert grid.get_unit_position("unit1") == pos
    
    def test_place_unit_occupied(self):
        """이미 점유된 위치에 배치 실패"""
        grid = HexGrid()
        pos = HexPosition(0, 0)
        grid.place_unit("unit1", pos)
        assert not grid.place_unit("unit2", pos)
    
    def test_place_unit_invalid_position(self):
        """유효하지 않은 위치에 배치"""
        grid = HexGrid()
        with pytest.raises(ValueError):
            grid.place_unit("unit1", HexPosition(10, 10))
    
    def test_remove_unit(self):
        """유닛 제거"""
        grid = HexGrid()
        pos = HexPosition(0, 0)
        grid.place_unit("unit1", pos)
        
        removed_pos = grid.remove_unit("unit1")
        assert removed_pos == pos
        assert grid.get_unit_at(pos) is None
        assert grid.get_unit_position("unit1") is None
    
    def test_remove_nonexistent_unit(self):
        """존재하지 않는 유닛 제거"""
        grid = HexGrid()
        assert grid.remove_unit("nonexistent") is None
    
    def test_move_unit(self):
        """유닛 이동"""
        grid = HexGrid()
        pos1 = HexPosition(0, 0)
        pos2 = HexPosition(1, 1)
        grid.place_unit("unit1", pos1)
        
        assert grid.move_unit("unit1", pos2)
        assert grid.get_unit_at(pos1) is None
        assert grid.get_unit_at(pos2) == "unit1"
        assert grid.get_unit_position("unit1") == pos2
    
    def test_move_unit_to_occupied(self):
        """점유된 위치로 이동 실패"""
        grid = HexGrid()
        grid.place_unit("unit1", HexPosition(0, 0))
        grid.place_unit("unit2", HexPosition(1, 1))
        
        assert not grid.move_unit("unit1", HexPosition(1, 1))
    
    def test_get_valid_neighbors(self):
        """보드 내 유효한 인접 헥스"""
        grid = HexGrid()
        
        # 코너는 인접 헥스가 적음
        corner = HexPosition(0, 0)
        valid = grid.get_valid_neighbors(corner)
        assert len(valid) < 6
        
        # 중앙은 6개 모두 유효
        center = HexPosition(3, 3)
        valid = grid.get_valid_neighbors(center)
        assert len(valid) == 6
    
    def test_get_empty_neighbors(self):
        """비어있는 인접 헥스"""
        grid = HexGrid()
        center = HexPosition(3, 3)
        
        # 처음에는 모두 비어있음
        empty = grid.get_empty_neighbors(center)
        assert len(empty) == 6
        
        # 하나 채우면 5개
        grid.place_unit("unit1", HexPosition(2, 3))
        empty = grid.get_empty_neighbors(center)
        assert len(empty) == 5
    
    def test_get_units_in_range(self):
        """범위 내 유닛 찾기"""
        grid = HexGrid()
        center = HexPosition(3, 3)
        grid.place_unit("center", center)
        grid.place_unit("adjacent", HexPosition(2, 3))  # 거리 1
        grid.place_unit("far", HexPosition(0, 0))  # 거리 많음
        
        # 범위 1: 중심 + 인접
        in_range = grid.get_units_in_range(center, 1)
        assert "center" in in_range
        assert "adjacent" in in_range
        assert "far" not in in_range
    
    def test_get_all_units(self):
        """모든 유닛 조회"""
        grid = HexGrid()
        grid.place_unit("unit1", HexPosition(0, 0))
        grid.place_unit("unit2", HexPosition(1, 1))
        
        all_units = grid.get_all_units()
        assert len(all_units) == 2
        assert "unit1" in all_units
        assert "unit2" in all_units
    
    def test_clear(self):
        """그리드 초기화"""
        grid = HexGrid()
        grid.place_unit("unit1", HexPosition(0, 0))
        grid.clear()
        
        assert len(grid.get_all_units()) == 0
    
    def test_team_for_position(self):
        """위치별 팀 영역"""
        grid = HexGrid()
        
        # Row 0-3은 BLUE
        assert grid.get_team_for_position(HexPosition(0, 0)) == Team.BLUE
        assert grid.get_team_for_position(HexPosition(3, 6)) == Team.BLUE
        
        # Row 4-7은 RED
        assert grid.get_team_for_position(HexPosition(4, 0)) == Team.RED
        assert grid.get_team_for_position(HexPosition(7, 6)) == Team.RED
    
    def test_mirror_position(self):
        """미러 위치 계산"""
        # (0, 0) -> (7, 6)
        assert HexGrid.mirror_position(HexPosition(0, 0)) == HexPosition(7, 6)
        
        # (0, 6) -> (7, 0)
        assert HexGrid.mirror_position(HexPosition(0, 6)) == HexPosition(7, 0)
        
        # (3, 3) -> (4, 3)
        assert HexGrid.mirror_position(HexPosition(3, 3)) == HexPosition(4, 3)
        
        # 두 번 미러링하면 원래대로
        pos = HexPosition(2, 4)
        assert HexGrid.mirror_position(HexGrid.mirror_position(pos)) == pos


class TestTeam:
    """Team enum 테스트"""
    
    def test_team_values(self):
        """팀 값"""
        assert Team.BLUE.value == "blue"
        assert Team.RED.value == "red"
```

### 5. `tests/test_combat_unit.py`

```python
"""CombatUnit 테스트"""

import pytest
from src.combat.combat_unit import CombatUnit, CombatStats, UnitState
from src.combat.hex_grid import Team


class TestCombatStats:
    """CombatStats 테스트"""
    
    def test_create_stats(self):
        """스탯 생성"""
        stats = CombatStats(
            max_hp=1000,
            current_hp=1000,
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
        assert stats.max_hp == 1000
        assert stats.attack_damage == 100


class TestCombatUnit:
    """CombatUnit 테스트"""
    
    @pytest.fixture
    def basic_unit(self):
        """기본 테스트 유닛"""
        stats = CombatStats(
            max_hp=1000,
            current_hp=1000,
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
        return CombatUnit(
            id="test-unit-1",
            name="Test Champion",
            champion_id="TFT_Test",
            star_level=1,
            team=Team.BLUE,
            stats=stats,
        )
    
    def test_is_alive(self, basic_unit):
        """생존 여부 체크"""
        assert basic_unit.is_alive
        basic_unit.stats.current_hp = 0
        assert not basic_unit.is_alive
    
    def test_can_act(self, basic_unit):
        """행동 가능 체크"""
        assert basic_unit.can_act
        basic_unit.state = UnitState.STUNNED
        assert not basic_unit.can_act
    
    def test_attack_interval(self, basic_unit):
        """공격 간격 계산"""
        # 공속 1.0 = 1초에 1회
        assert basic_unit.attack_interval == 1.0
        
        # 공속 2.0 = 0.5초에 1회
        basic_unit.stats.attack_speed = 2.0
        assert basic_unit.attack_interval == 0.5
    
    def test_take_physical_damage(self, basic_unit):
        """물리 데미지"""
        # armor 50 = 50/(50+100) = 33.3% 감소
        damage = basic_unit.take_damage(100, "physical")
        expected = 100 * (1 - 50/150)  # ~66.67
        assert abs(damage - expected) < 0.01
        assert basic_unit.stats.current_hp < 1000
    
    def test_take_magical_damage(self, basic_unit):
        """마법 데미지"""
        # mr 50 = 50/(50+100) = 33.3% 감소
        damage = basic_unit.take_damage(100, "magical")
        expected = 100 * (1 - 50/150)
        assert abs(damage - expected) < 0.01
    
    def test_take_true_damage(self, basic_unit):
        """트루 데미지"""
        damage = basic_unit.take_damage(100, "true")
        assert damage == 100
    
    def test_damage_kills_unit(self, basic_unit):
        """데미지로 사망"""
        basic_unit.take_damage(10000, "true")
        assert not basic_unit.is_alive
        assert basic_unit.state == UnitState.DEAD
        assert basic_unit.stats.current_hp == 0
    
    def test_damage_reduction(self, basic_unit):
        """데미지 감소 효과"""
        basic_unit.stats.damage_reduction = 0.2  # 20% 감소
        damage = basic_unit.take_damage(100, "true")
        assert damage == 80
    
    def test_heal(self, basic_unit):
        """체력 회복"""
        basic_unit.stats.current_hp = 500
        healed = basic_unit.heal(200)
        assert healed == 200
        assert basic_unit.stats.current_hp == 700
    
    def test_heal_no_overheal(self, basic_unit):
        """최대 체력 초과 회복 불가"""
        basic_unit.stats.current_hp = 900
        healed = basic_unit.heal(200)
        assert healed == 100
        assert basic_unit.stats.current_hp == 1000
    
    def test_heal_dead_unit(self, basic_unit):
        """사망한 유닛 회복 불가"""
        basic_unit.state = UnitState.DEAD
        basic_unit.stats.current_hp = 0
        healed = basic_unit.heal(500)
        assert healed == 0
    
    def test_gain_mana(self, basic_unit):
        """마나 획득"""
        basic_unit.stats.current_mana = 50
        basic_unit.gain_mana(30)
        assert basic_unit.stats.current_mana == 80
    
    def test_gain_mana_cap(self, basic_unit):
        """마나 최대치"""
        basic_unit.stats.current_mana = 90
        basic_unit.gain_mana(30)
        assert basic_unit.stats.current_mana == 100
    
    def test_spend_mana(self, basic_unit):
        """마나 소모 (스킬 후 시작 마나로)"""
        basic_unit.stats.current_mana = 100
        basic_unit.spend_mana()
        assert basic_unit.stats.current_mana == basic_unit.stats.starting_mana
    
    def test_can_cast(self, basic_unit):
        """스킬 시전 가능 여부"""
        basic_unit.stats.current_mana = 50
        assert not basic_unit.can_cast
        
        basic_unit.stats.current_mana = 100
        assert basic_unit.can_cast
        
        basic_unit.is_casting = True
        assert not basic_unit.can_cast
    
    def test_reset_for_combat(self, basic_unit):
        """전투 리셋"""
        basic_unit.stats.current_hp = 500
        basic_unit.stats.current_mana = 100
        basic_unit.total_damage_dealt = 1000
        basic_unit.state = UnitState.ATTACKING
        
        basic_unit.reset_for_combat()
        
        assert basic_unit.stats.current_hp == basic_unit.stats.max_hp
        assert basic_unit.stats.current_mana == basic_unit.stats.starting_mana
        assert basic_unit.total_damage_dealt == 0
        assert basic_unit.state == UnitState.IDLE
    
    def test_damage_tracking(self, basic_unit):
        """데미지 통계 추적"""
        basic_unit.take_damage(100, "true")
        basic_unit.take_damage(50, "true")
        assert basic_unit.total_damage_taken == 150
```

## 디렉토리 구조

```
tft-simulator/
├── src/
│   ├── combat/                    # NEW
│   │   ├── __init__.py
│   │   ├── hex_grid.py           # 헥스 그리드 시스템
│   │   └── combat_unit.py        # 전투 유닛
│   ├── core/
│   │   └── ... (기존 파일들)
│   └── data/
│       └── ... (기존 파일들)
├── tests/
│   ├── test_hex_grid.py          # NEW
│   ├── test_combat_unit.py       # NEW
│   └── ... (기존 테스트들)
└── data/
    └── ... (기존 데이터)
```

## 체크리스트

- [ ] `src/combat/` 디렉토리 생성
- [ ] `hex_grid.py` 구현
  - [ ] HexPosition 클래스
  - [ ] Cube 좌표 변환
  - [ ] 거리 계산
  - [ ] 인접 헥스 탐색
  - [ ] HexGrid 클래스
  - [ ] 유닛 배치/이동/제거
  - [ ] 미러링
- [ ] `combat_unit.py` 구현
  - [ ] CombatStats 데이터클래스
  - [ ] CombatUnit 클래스
  - [ ] 데미지 계산 (물리/마법/트루)
  - [ ] 체력/마나 관리
  - [ ] from_champion_instance 팩토리
- [ ] `__init__.py` exports
- [ ] 테스트 작성 및 통과

## 다음 스테이지 예고

**Stage 5.2: 타겟팅 & 이동 시스템**
- 타겟 선택 로직 (최근접 적, 어그로 등)
- A* 경로 탐색
- 헥스 기반 이동
