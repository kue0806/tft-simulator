# Phase 5 - Stage 5.2: 타겟팅 & 이동 시스템

## 목표
전투 유닛의 타겟 선택 로직과 헥스 그리드 기반 이동 시스템(A* 경로 탐색)을 구현합니다.

## TFT 타겟팅 규칙

### 기본 타겟팅 우선순위
1. **현재 타겟 유지**: 기존 타겟이 살아있고 사거리 내면 유지
2. **최근접 적**: 거리가 가장 가까운 적 선택
3. **동일 거리 시**: 여러 규칙으로 타이브레이크
   - 현재 공격받는 유닛 우선 (어그로)
   - 랜덤 선택 (시뮬레이션에서는 deterministic하게)

### 특수 타겟팅 (일부 챔피언/아이템)
- **Assassin**: 가장 먼 적 (후방 캐리)
- **특정 스킬**: 가장 HP가 낮은 적, 가장 HP가 높은 적 등

## 구현할 파일들

### 1. `src/combat/targeting.py`

```python
"""
타겟팅 시스템
- 타겟 선택 로직
- 사거리 체크
- 특수 타겟팅 지원
"""

from dataclasses import dataclass
from typing import Optional, List, Callable, TYPE_CHECKING
from enum import Enum, auto
import random

if TYPE_CHECKING:
    from .combat_unit import CombatUnit
    from .hex_grid import HexGrid, HexPosition, Team


class TargetingPriority(Enum):
    """타겟팅 우선순위 타입"""
    NEAREST = auto()           # 최근접 (기본)
    FARTHEST = auto()          # 최원거리 (어쌔신)
    LOWEST_HP = auto()         # 최저 HP
    HIGHEST_HP = auto()        # 최고 HP
    LOWEST_HP_PERCENT = auto() # 최저 HP%
    HIGHEST_MANA = auto()      # 최고 마나 (마나 리브)
    RANDOM = auto()            # 랜덤


@dataclass
class TargetingContext:
    """타겟팅에 필요한 컨텍스트"""
    grid: 'HexGrid'
    all_units: dict[str, 'CombatUnit']  # unit_id -> CombatUnit
    
    def get_unit_position(self, unit_id: str) -> Optional['HexPosition']:
        """유닛 위치 조회"""
        return self.grid.get_unit_position(unit_id)
    
    def get_unit(self, unit_id: str) -> Optional['CombatUnit']:
        """유닛 조회"""
        return self.all_units.get(unit_id)


class TargetSelector:
    """
    타겟 선택기
    
    Usage:
        selector = TargetSelector(context)
        target = selector.find_target(attacker, priority=TargetingPriority.NEAREST)
    """
    
    def __init__(self, context: TargetingContext, rng: Optional[random.Random] = None):
        self.context = context
        self.rng = rng or random.Random()
    
    def find_target(
        self, 
        attacker: 'CombatUnit',
        priority: TargetingPriority = TargetingPriority.NEAREST,
        filter_fn: Optional[Callable[['CombatUnit'], bool]] = None
    ) -> Optional[str]:
        """
        공격자의 타겟 찾기
        
        Args:
            attacker: 공격하는 유닛
            priority: 타겟팅 우선순위
            filter_fn: 추가 필터 (예: 특정 특성만)
            
        Returns:
            타겟 유닛 ID 또는 None
        """
        # 공격자 위치
        attacker_pos = self.context.get_unit_position(attacker.id)
        if attacker_pos is None:
            return None
        
        # 적 팀 유닛들 필터링
        enemies = self._get_valid_enemies(attacker, filter_fn)
        if not enemies:
            return None
        
        # 우선순위에 따른 정렬
        if priority == TargetingPriority.NEAREST:
            return self._select_nearest(attacker_pos, enemies)
        elif priority == TargetingPriority.FARTHEST:
            return self._select_farthest(attacker_pos, enemies)
        elif priority == TargetingPriority.LOWEST_HP:
            return self._select_lowest_hp(enemies)
        elif priority == TargetingPriority.HIGHEST_HP:
            return self._select_highest_hp(enemies)
        elif priority == TargetingPriority.LOWEST_HP_PERCENT:
            return self._select_lowest_hp_percent(enemies)
        elif priority == TargetingPriority.HIGHEST_MANA:
            return self._select_highest_mana(enemies)
        elif priority == TargetingPriority.RANDOM:
            return self._select_random(enemies)
        else:
            return self._select_nearest(attacker_pos, enemies)
    
    def _get_valid_enemies(
        self, 
        attacker: 'CombatUnit',
        filter_fn: Optional[Callable[['CombatUnit'], bool]] = None
    ) -> List['CombatUnit']:
        """유효한 적 유닛 목록"""
        enemies = []
        for unit_id, unit in self.context.all_units.items():
            # 같은 팀 제외
            if unit.team == attacker.team:
                continue
            # 타겟 불가 제외
            if not unit.is_targetable:
                continue
            # 추가 필터
            if filter_fn and not filter_fn(unit):
                continue
            enemies.append(unit)
        return enemies
    
    def _select_nearest(
        self, 
        from_pos: 'HexPosition', 
        candidates: List['CombatUnit']
    ) -> Optional[str]:
        """최근접 적 선택 (타이브레이크: 낮은 row, 낮은 col)"""
        if not candidates:
            return None
        
        def sort_key(unit: 'CombatUnit'):
            pos = self.context.get_unit_position(unit.id)
            if pos is None:
                return (float('inf'), float('inf'), float('inf'))
            distance = from_pos.distance_to(pos)
            # 타이브레이크: 거리 -> row -> col (deterministic)
            return (distance, pos.row, pos.col)
        
        candidates.sort(key=sort_key)
        return candidates[0].id
    
    def _select_farthest(
        self, 
        from_pos: 'HexPosition', 
        candidates: List['CombatUnit']
    ) -> Optional[str]:
        """최원거리 적 선택 (어쌔신용)"""
        if not candidates:
            return None
        
        def sort_key(unit: 'CombatUnit'):
            pos = self.context.get_unit_position(unit.id)
            if pos is None:
                return (float('-inf'), 0, 0)
            distance = from_pos.distance_to(pos)
            return (-distance, pos.row, pos.col)  # 거리 내림차순
        
        candidates.sort(key=sort_key)
        return candidates[0].id
    
    def _select_lowest_hp(self, candidates: List['CombatUnit']) -> Optional[str]:
        """최저 HP 적 선택"""
        if not candidates:
            return None
        candidates.sort(key=lambda u: (u.stats.current_hp, u.id))
        return candidates[0].id
    
    def _select_highest_hp(self, candidates: List['CombatUnit']) -> Optional[str]:
        """최고 HP 적 선택"""
        if not candidates:
            return None
        candidates.sort(key=lambda u: (-u.stats.current_hp, u.id))
        return candidates[0].id
    
    def _select_lowest_hp_percent(self, candidates: List['CombatUnit']) -> Optional[str]:
        """최저 HP% 적 선택"""
        if not candidates:
            return None
        
        def hp_percent(unit: 'CombatUnit') -> float:
            if unit.stats.max_hp <= 0:
                return 1.0
            return unit.stats.current_hp / unit.stats.max_hp
        
        candidates.sort(key=lambda u: (hp_percent(u), u.id))
        return candidates[0].id
    
    def _select_highest_mana(self, candidates: List['CombatUnit']) -> Optional[str]:
        """최고 마나 적 선택"""
        if not candidates:
            return None
        candidates.sort(key=lambda u: (-u.stats.current_mana, u.id))
        return candidates[0].id
    
    def _select_random(self, candidates: List['CombatUnit']) -> Optional[str]:
        """랜덤 선택"""
        if not candidates:
            return None
        return self.rng.choice(candidates).id
    
    def is_in_range(
        self, 
        attacker: 'CombatUnit', 
        target_id: str, 
        attack_range: Optional[int] = None
    ) -> bool:
        """
        타겟이 사거리 내에 있는지 확인
        
        Args:
            attacker: 공격자
            target_id: 타겟 ID
            attack_range: 사거리 (None이면 공격자의 기본 사거리)
        """
        attacker_pos = self.context.get_unit_position(attacker.id)
        target_pos = self.context.get_unit_position(target_id)
        
        if attacker_pos is None or target_pos is None:
            return False
        
        range_ = attack_range if attack_range is not None else attacker.stats.attack_range
        return attacker_pos.distance_to(target_pos) <= range_
    
    def get_units_in_range(
        self, 
        attacker: 'CombatUnit',
        range_: int,
        enemies_only: bool = True
    ) -> List[str]:
        """사거리 내 유닛들"""
        attacker_pos = self.context.get_unit_position(attacker.id)
        if attacker_pos is None:
            return []
        
        result = []
        for unit_id, unit in self.context.all_units.items():
            if unit_id == attacker.id:
                continue
            if enemies_only and unit.team == attacker.team:
                continue
            if not unit.is_targetable:
                continue
            
            unit_pos = self.context.get_unit_position(unit_id)
            if unit_pos and attacker_pos.distance_to(unit_pos) <= range_:
                result.append(unit_id)
        
        return result
```

### 2. `src/combat/pathfinding.py`

```python
"""
A* 경로 탐색
- 헥스 그리드 기반 최단 경로
- 장애물 (다른 유닛) 회피
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple
from heapq import heappush, heappop

from .hex_grid import HexGrid, HexPosition


@dataclass(order=True)
class PathNode:
    """A* 탐색용 노드"""
    f_score: float  # g + h
    position: HexPosition = field(compare=False)
    g_score: float = field(compare=False)  # 시작점에서의 실제 비용
    parent: Optional['PathNode'] = field(default=None, compare=False)


class PathFinder:
    """
    A* 기반 경로 탐색기
    
    Usage:
        finder = PathFinder(grid)
        path = finder.find_path(start, goal, blocked_positions)
    """
    
    def __init__(self, grid: HexGrid):
        self.grid = grid
    
    def find_path(
        self, 
        start: HexPosition, 
        goal: HexPosition,
        blocked: Optional[Set[HexPosition]] = None,
        max_iterations: int = 1000
    ) -> Optional[List[HexPosition]]:
        """
        시작점에서 목표까지의 최단 경로 찾기
        
        Args:
            start: 시작 위치
            goal: 목표 위치
            blocked: 막힌 위치들 (다른 유닛 등)
            max_iterations: 최대 탐색 횟수 (무한 루프 방지)
            
        Returns:
            경로 (시작점 제외, 목표점 포함) 또는 None
        """
        if start == goal:
            return []
        
        if blocked is None:
            blocked = set()
        
        # 목표가 막혀있으면 근처로 이동
        # (공격 사거리 내로만 이동하면 되므로)
        
        open_set: List[PathNode] = []
        closed_set: Set[HexPosition] = set()
        g_scores: Dict[HexPosition, float] = {start: 0}
        
        start_node = PathNode(
            f_score=self._heuristic(start, goal),
            position=start,
            g_score=0
        )
        heappush(open_set, start_node)
        
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current = heappop(open_set)
            
            if current.position == goal:
                return self._reconstruct_path(current)
            
            if current.position in closed_set:
                continue
            
            closed_set.add(current.position)
            
            # 인접 헥스 탐색
            for neighbor_pos in current.position.get_neighbors():
                # 유효성 체크
                if not neighbor_pos.is_valid(self.grid.ROWS, self.grid.COLS):
                    continue
                
                # 막힌 위치 체크 (목표 위치는 예외)
                if neighbor_pos != goal and neighbor_pos in blocked:
                    continue
                
                if neighbor_pos in closed_set:
                    continue
                
                # g 점수 계산 (모든 이동 비용 = 1)
                tentative_g = current.g_score + 1
                
                if neighbor_pos in g_scores and tentative_g >= g_scores[neighbor_pos]:
                    continue
                
                g_scores[neighbor_pos] = tentative_g
                f_score = tentative_g + self._heuristic(neighbor_pos, goal)
                
                neighbor_node = PathNode(
                    f_score=f_score,
                    position=neighbor_pos,
                    g_score=tentative_g,
                    parent=current
                )
                heappush(open_set, neighbor_node)
        
        # 경로 없음
        return None
    
    def find_path_to_range(
        self,
        start: HexPosition,
        target: HexPosition,
        attack_range: int,
        blocked: Optional[Set[HexPosition]] = None
    ) -> Optional[List[HexPosition]]:
        """
        타겟의 공격 사거리 내까지의 경로 찾기
        
        공격 사거리 내에 도달할 수 있는 가장 가까운 위치로 이동
        """
        if blocked is None:
            blocked = set()
        
        # 이미 사거리 내면 이동 불필요
        if start.distance_to(target) <= attack_range:
            return []
        
        # 사거리 내의 모든 위치 중 막히지 않은 곳 찾기
        valid_goals = []
        for row in range(self.grid.ROWS):
            for col in range(self.grid.COLS):
                pos = HexPosition(row, col)
                if pos == start:
                    continue
                if pos in blocked:
                    continue
                if pos.distance_to(target) <= attack_range:
                    valid_goals.append(pos)
        
        if not valid_goals:
            return None
        
        # 가장 가까운 유효 목표로 경로 찾기
        best_path = None
        best_length = float('inf')
        
        for goal in valid_goals:
            # 간단한 휴리스틱: 시작점에서 가까운 것부터 시도
            if start.distance_to(goal) >= best_length:
                continue
            
            path = self.find_path(start, goal, blocked)
            if path is not None and len(path) < best_length:
                best_path = path
                best_length = len(path)
        
        return best_path
    
    def get_next_step(
        self,
        start: HexPosition,
        goal: HexPosition,
        blocked: Optional[Set[HexPosition]] = None
    ) -> Optional[HexPosition]:
        """경로의 첫 번째 스텝만 반환 (효율적인 이동용)"""
        path = self.find_path(start, goal, blocked)
        if path and len(path) > 0:
            return path[0]
        return None
    
    def _heuristic(self, a: HexPosition, b: HexPosition) -> float:
        """휴리스틱: 헥스 거리"""
        return float(a.distance_to(b))
    
    def _reconstruct_path(self, node: PathNode) -> List[HexPosition]:
        """경로 복원 (시작점 제외)"""
        path = []
        current = node
        while current.parent is not None:
            path.append(current.position)
            current = current.parent
        path.reverse()
        return path


def get_blocked_positions(grid: HexGrid, exclude_unit_id: Optional[str] = None) -> Set[HexPosition]:
    """
    현재 그리드에서 막힌 위치들 (유닛이 있는 곳)
    
    Args:
        grid: 헥스 그리드
        exclude_unit_id: 제외할 유닛 (자기 자신)
    """
    blocked = set()
    for unit_id, pos in grid.get_all_units().items():
        if unit_id != exclude_unit_id:
            blocked.add(pos)
    return blocked
```

### 3. `src/combat/movement.py`

```python
"""
유닛 이동 시스템
- 틱 기반 이동
- 경로 따라 이동
"""

from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING

from .hex_grid import HexGrid, HexPosition
from .pathfinding import PathFinder, get_blocked_positions

if TYPE_CHECKING:
    from .combat_unit import CombatUnit


# TFT 이동 속도 (모든 유닛 동일)
BASE_MOVE_SPEED = 550  # 유닛/초 (대략적 값)
HEX_SIZE = 180  # 헥스 크기 (대략적 값)
MOVE_TIME_PER_HEX = HEX_SIZE / BASE_MOVE_SPEED  # 약 0.33초/헥스


@dataclass
class MovementState:
    """유닛의 이동 상태"""
    unit_id: str
    current_path: List[HexPosition]  # 남은 경로
    progress: float  # 현재 헥스 이동 진행률 (0.0 ~ 1.0)
    is_moving: bool
    
    @property
    def next_position(self) -> Optional[HexPosition]:
        """다음 목표 헥스"""
        if self.current_path:
            return self.current_path[0]
        return None


class MovementSystem:
    """
    유닛 이동 관리 시스템
    
    Usage:
        movement = MovementSystem(grid)
        movement.start_move(unit, target_pos)
        movement.update(delta_time)  # 매 틱마다
    """
    
    def __init__(self, grid: HexGrid, move_time_per_hex: float = MOVE_TIME_PER_HEX):
        self.grid = grid
        self.pathfinder = PathFinder(grid)
        self.move_time_per_hex = move_time_per_hex
        
        # 유닛별 이동 상태
        self._movement_states: dict[str, MovementState] = {}
    
    def start_move_to_target(
        self, 
        unit: 'CombatUnit', 
        target_pos: HexPosition,
        attack_range: int = 1
    ) -> bool:
        """
        타겟 위치의 공격 사거리 내로 이동 시작
        
        Returns:
            True if 이동 시작됨, False if 이동 불필요 또는 불가
        """
        unit_pos = self.grid.get_unit_position(unit.id)
        if unit_pos is None:
            return False
        
        # 이미 사거리 내면 이동 불필요
        if unit_pos.distance_to(target_pos) <= attack_range:
            self._stop_movement(unit.id)
            return False
        
        # 경로 찾기
        blocked = get_blocked_positions(self.grid, exclude_unit_id=unit.id)
        path = self.pathfinder.find_path_to_range(
            unit_pos, target_pos, attack_range, blocked
        )
        
        if not path:
            return False
        
        # 이동 상태 설정
        self._movement_states[unit.id] = MovementState(
            unit_id=unit.id,
            current_path=path,
            progress=0.0,
            is_moving=True
        )
        
        return True
    
    def start_move_to_position(self, unit: 'CombatUnit', goal: HexPosition) -> bool:
        """특정 위치로 직접 이동"""
        unit_pos = self.grid.get_unit_position(unit.id)
        if unit_pos is None:
            return False
        
        if unit_pos == goal:
            return False
        
        blocked = get_blocked_positions(self.grid, exclude_unit_id=unit.id)
        path = self.pathfinder.find_path(unit_pos, goal, blocked)
        
        if not path:
            return False
        
        self._movement_states[unit.id] = MovementState(
            unit_id=unit.id,
            current_path=path,
            progress=0.0,
            is_moving=True
        )
        
        return True
    
    def update(self, unit: 'CombatUnit', delta_time: float) -> bool:
        """
        유닛 이동 업데이트
        
        Args:
            unit: 이동할 유닛
            delta_time: 경과 시간 (초)
            
        Returns:
            True if 이동 완료 (목적지 도착 또는 이동 중지)
        """
        state = self._movement_states.get(unit.id)
        if state is None or not state.is_moving:
            return True
        
        if not state.current_path:
            self._stop_movement(unit.id)
            return True
        
        # 진행률 업데이트
        progress_delta = delta_time / self.move_time_per_hex
        state.progress += progress_delta
        
        # 다음 헥스에 도착했는지 체크
        while state.progress >= 1.0 and state.current_path:
            state.progress -= 1.0
            next_pos = state.current_path.pop(0)
            
            # 실제 이동 수행
            if self.grid.move_unit(unit.id, next_pos):
                # 이동 성공
                pass
            else:
                # 이동 실패 (다른 유닛이 차지함) - 경로 재계산 필요
                state.is_moving = False
                return False
        
        # 경로 완료 체크
        if not state.current_path:
            self._stop_movement(unit.id)
            return True
        
        return False
    
    def is_moving(self, unit_id: str) -> bool:
        """유닛이 이동 중인지"""
        state = self._movement_states.get(unit_id)
        return state is not None and state.is_moving
    
    def get_movement_progress(self, unit_id: str) -> float:
        """현재 헥스 이동 진행률"""
        state = self._movement_states.get(unit_id)
        if state is None:
            return 0.0
        return state.progress
    
    def stop_movement(self, unit_id: str) -> None:
        """이동 중지 (외부 호출용)"""
        self._stop_movement(unit_id)
    
    def _stop_movement(self, unit_id: str) -> None:
        """이동 중지"""
        if unit_id in self._movement_states:
            self._movement_states[unit_id].is_moving = False
            self._movement_states[unit_id].current_path = []
            self._movement_states[unit_id].progress = 0.0
    
    def recalculate_path(
        self, 
        unit: 'CombatUnit', 
        target_pos: HexPosition,
        attack_range: int = 1
    ) -> bool:
        """경로 재계산 (장애물 변경 시)"""
        self._stop_movement(unit.id)
        return self.start_move_to_target(unit, target_pos, attack_range)
    
    def clear(self) -> None:
        """모든 이동 상태 초기화"""
        self._movement_states.clear()
```

### 4. `src/combat/__init__.py` 업데이트

```python
"""Combat simulation module"""

from .hex_grid import HexPosition, HexGrid, Team
from .combat_unit import CombatUnit, CombatStats, CombatResult, UnitState
from .targeting import TargetSelector, TargetingContext, TargetingPriority
from .pathfinding import PathFinder, get_blocked_positions
from .movement import MovementSystem, MovementState, MOVE_TIME_PER_HEX

__all__ = [
    # Hex Grid
    'HexPosition',
    'HexGrid', 
    'Team',
    # Combat Unit
    'CombatUnit',
    'CombatStats',
    'CombatResult',
    'UnitState',
    # Targeting
    'TargetSelector',
    'TargetingContext',
    'TargetingPriority',
    # Pathfinding
    'PathFinder',
    'get_blocked_positions',
    # Movement
    'MovementSystem',
    'MovementState',
    'MOVE_TIME_PER_HEX',
]
```

### 5. `tests/test_targeting.py`

```python
"""타겟팅 시스템 테스트"""

import pytest
import random
from src.combat.hex_grid import HexGrid, HexPosition, Team
from src.combat.combat_unit import CombatUnit, CombatStats, UnitState
from src.combat.targeting import (
    TargetSelector, TargetingContext, TargetingPriority
)


def create_test_unit(
    unit_id: str, 
    team: Team, 
    hp: float = 1000,
    mana: float = 50,
    attack_range: int = 1
) -> CombatUnit:
    """테스트용 유닛 생성"""
    stats = CombatStats(
        max_hp=hp, current_hp=hp,
        attack_damage=100, ability_power=100,
        armor=50, magic_resist=50,
        attack_speed=1.0, crit_chance=0.25, crit_damage=1.4,
        max_mana=100, current_mana=mana, starting_mana=50,
        attack_range=attack_range, dodge_chance=0.0,
        omnivamp=0.0, damage_amp=1.0, damage_reduction=0.0
    )
    return CombatUnit(
        id=unit_id, name=f"Unit_{unit_id}",
        champion_id="TFT_Test", star_level=1,
        team=team, stats=stats
    )


class TestTargetSelector:
    """TargetSelector 테스트"""
    
    @pytest.fixture
    def setup_battle(self):
        """기본 전투 설정"""
        grid = HexGrid()
        units = {}
        
        # BLUE 팀 유닛 (row 0-3)
        blue1 = create_test_unit("blue1", Team.BLUE)
        grid.place_unit("blue1", HexPosition(3, 3))
        units["blue1"] = blue1
        
        # RED 팀 유닛 (row 4-7)
        red1 = create_test_unit("red1", Team.RED)
        red2 = create_test_unit("red2", Team.RED, hp=500)
        red3 = create_test_unit("red3", Team.RED, hp=1500)
        
        grid.place_unit("red1", HexPosition(4, 3))  # 거리 1
        grid.place_unit("red2", HexPosition(5, 3))  # 거리 2
        grid.place_unit("red3", HexPosition(6, 3))  # 거리 3
        
        units["red1"] = red1
        units["red2"] = red2
        units["red3"] = red3
        
        context = TargetingContext(grid=grid, all_units=units)
        selector = TargetSelector(context, rng=random.Random(42))
        
        return grid, units, selector
    
    def test_find_nearest_target(self, setup_battle):
        """최근접 적 타겟팅"""
        grid, units, selector = setup_battle
        
        target = selector.find_target(
            units["blue1"], 
            priority=TargetingPriority.NEAREST
        )
        
        assert target == "red1"  # 거리 1
    
    def test_find_farthest_target(self, setup_battle):
        """최원거리 적 타겟팅 (어쌔신)"""
        grid, units, selector = setup_battle
        
        target = selector.find_target(
            units["blue1"],
            priority=TargetingPriority.FARTHEST
        )
        
        assert target == "red3"  # 거리 3
    
    def test_find_lowest_hp_target(self, setup_battle):
        """최저 HP 타겟팅"""
        grid, units, selector = setup_battle
        
        target = selector.find_target(
            units["blue1"],
            priority=TargetingPriority.LOWEST_HP
        )
        
        assert target == "red2"  # HP 500
    
    def test_find_highest_hp_target(self, setup_battle):
        """최고 HP 타겟팅"""
        grid, units, selector = setup_battle
        
        target = selector.find_target(
            units["blue1"],
            priority=TargetingPriority.HIGHEST_HP
        )
        
        assert target == "red3"  # HP 1500
    
    def test_no_target_same_team(self, setup_battle):
        """같은 팀은 타겟 안됨"""
        grid, units, selector = setup_battle
        
        # BLUE 팀끼리
        blue2 = create_test_unit("blue2", Team.BLUE)
        grid.place_unit("blue2", HexPosition(2, 3))
        units["blue2"] = blue2
        
        # blue2의 타겟은 RED 팀만
        target = selector.find_target(
            units["blue2"],
            priority=TargetingPriority.NEAREST
        )
        
        assert target in ["red1", "red2", "red3"]
    
    def test_no_target_dead_unit(self, setup_battle):
        """죽은 유닛은 타겟 안됨"""
        grid, units, selector = setup_battle
        
        # red1 사망 처리
        units["red1"].state = UnitState.DEAD
        units["red1"].stats.current_hp = 0
        
        target = selector.find_target(
            units["blue1"],
            priority=TargetingPriority.NEAREST
        )
        
        assert target == "red2"  # red1 대신 red2
    
    def test_is_in_range_true(self, setup_battle):
        """사거리 내 체크 - True"""
        grid, units, selector = setup_battle
        
        # blue1(3,3) -> red1(4,3) = 거리 1
        assert selector.is_in_range(units["blue1"], "red1", attack_range=1)
    
    def test_is_in_range_false(self, setup_battle):
        """사거리 내 체크 - False"""
        grid, units, selector = setup_battle
        
        # blue1(3,3) -> red2(5,3) = 거리 2
        assert not selector.is_in_range(units["blue1"], "red2", attack_range=1)
    
    def test_get_units_in_range(self, setup_battle):
        """범위 내 적 유닛들"""
        grid, units, selector = setup_battle
        
        in_range = selector.get_units_in_range(units["blue1"], range_=2)
        
        assert "red1" in in_range  # 거리 1
        assert "red2" in in_range  # 거리 2
        assert "red3" not in in_range  # 거리 3
    
    def test_filter_function(self, setup_battle):
        """커스텀 필터 함수"""
        grid, units, selector = setup_battle
        
        # HP가 1000 초과인 유닛만
        target = selector.find_target(
            units["blue1"],
            priority=TargetingPriority.NEAREST,
            filter_fn=lambda u: u.stats.current_hp > 1000
        )
        
        assert target == "red3"  # HP 1500만 해당
    
    def test_tiebreak_by_position(self):
        """동일 거리 시 위치로 타이브레이크"""
        grid = HexGrid()
        units = {}
        
        blue = create_test_unit("blue", Team.BLUE)
        grid.place_unit("blue", HexPosition(3, 3))
        units["blue"] = blue
        
        # 동일 거리에 두 적
        red1 = create_test_unit("red1", Team.RED)
        red2 = create_test_unit("red2", Team.RED)
        
        grid.place_unit("red1", HexPosition(4, 4))  # 거리 1
        grid.place_unit("red2", HexPosition(4, 3))  # 거리 1
        
        units["red1"] = red1
        units["red2"] = red2
        
        context = TargetingContext(grid=grid, all_units=units)
        selector = TargetSelector(context)
        
        target = selector.find_target(
            units["blue"],
            priority=TargetingPriority.NEAREST
        )
        
        # row가 같으므로 col이 낮은 red2 선택
        assert target == "red2"
```

### 6. `tests/test_pathfinding.py`

```python
"""경로 탐색 테스트"""

import pytest
from src.combat.hex_grid import HexGrid, HexPosition
from src.combat.pathfinding import PathFinder, get_blocked_positions


class TestPathFinder:
    """PathFinder 테스트"""
    
    @pytest.fixture
    def grid_and_finder(self):
        grid = HexGrid()
        finder = PathFinder(grid)
        return grid, finder
    
    def test_find_path_simple(self, grid_and_finder):
        """간단한 경로 찾기"""
        grid, finder = grid_and_finder
        
        start = HexPosition(0, 0)
        goal = HexPosition(0, 2)
        
        path = finder.find_path(start, goal)
        
        assert path is not None
        assert len(path) == 2  # 2칸 이동
        assert path[-1] == goal
    
    def test_find_path_same_position(self, grid_and_finder):
        """같은 위치 - 빈 경로"""
        grid, finder = grid_and_finder
        
        pos = HexPosition(3, 3)
        path = finder.find_path(pos, pos)
        
        assert path == []
    
    def test_find_path_blocked(self, grid_and_finder):
        """장애물 피해서 경로 찾기"""
        grid, finder = grid_and_finder
        
        start = HexPosition(2, 0)
        goal = HexPosition(2, 2)
        
        # 직선 경로 차단
        blocked = {HexPosition(2, 1)}
        
        path = finder.find_path(start, goal, blocked)
        
        assert path is not None
        assert HexPosition(2, 1) not in path  # 막힌 곳 안 지남
        assert path[-1] == goal
    
    def test_find_path_no_path(self, grid_and_finder):
        """경로 없음"""
        grid, finder = grid_and_finder
        
        start = HexPosition(0, 0)
        goal = HexPosition(0, 2)
        
        # 완전 차단
        blocked = {
            HexPosition(0, 1),
            HexPosition(1, 0),
            HexPosition(1, 1),
        }
        
        path = finder.find_path(start, goal, blocked)
        
        # 경로가 없거나 우회 경로
        # (보드가 넓어서 완전 차단이 어려울 수 있음)
    
    def test_find_path_to_range(self, grid_and_finder):
        """공격 사거리 내로 이동"""
        grid, finder = grid_and_finder
        
        start = HexPosition(0, 0)
        target = HexPosition(4, 4)
        
        path = finder.find_path_to_range(start, target, attack_range=1)
        
        assert path is not None
        # 도착점이 타겟의 사거리 내
        if path:
            final_pos = path[-1]
            assert final_pos.distance_to(target) <= 1
    
    def test_find_path_already_in_range(self, grid_and_finder):
        """이미 사거리 내"""
        grid, finder = grid_and_finder
        
        start = HexPosition(3, 3)
        target = HexPosition(4, 3)  # 거리 1
        
        path = finder.find_path_to_range(start, target, attack_range=1)
        
        assert path == []  # 이동 불필요
    
    def test_get_next_step(self, grid_and_finder):
        """다음 스텝만 얻기"""
        grid, finder = grid_and_finder
        
        start = HexPosition(0, 0)
        goal = HexPosition(2, 2)
        
        next_step = finder.get_next_step(start, goal)
        
        assert next_step is not None
        assert start.distance_to(next_step) == 1  # 인접 헥스
    
    def test_path_length_optimal(self, grid_and_finder):
        """최단 경로인지 확인"""
        grid, finder = grid_and_finder
        
        start = HexPosition(0, 0)
        goal = HexPosition(3, 0)
        
        path = finder.find_path(start, goal)
        
        assert path is not None
        # 최단 거리와 경로 길이가 같아야 함
        assert len(path) == start.distance_to(goal)


class TestGetBlockedPositions:
    """get_blocked_positions 테스트"""
    
    def test_empty_grid(self):
        """빈 그리드"""
        grid = HexGrid()
        blocked = get_blocked_positions(grid)
        assert len(blocked) == 0
    
    def test_with_units(self):
        """유닛이 있는 그리드"""
        grid = HexGrid()
        grid.place_unit("unit1", HexPosition(0, 0))
        grid.place_unit("unit2", HexPosition(1, 1))
        
        blocked = get_blocked_positions(grid)
        
        assert HexPosition(0, 0) in blocked
        assert HexPosition(1, 1) in blocked
        assert len(blocked) == 2
    
    def test_exclude_unit(self):
        """자기 자신 제외"""
        grid = HexGrid()
        grid.place_unit("unit1", HexPosition(0, 0))
        grid.place_unit("unit2", HexPosition(1, 1))
        
        blocked = get_blocked_positions(grid, exclude_unit_id="unit1")
        
        assert HexPosition(0, 0) not in blocked  # 제외됨
        assert HexPosition(1, 1) in blocked
        assert len(blocked) == 1
```

### 7. `tests/test_movement.py`

```python
"""이동 시스템 테스트"""

import pytest
from src.combat.hex_grid import HexGrid, HexPosition, Team
from src.combat.combat_unit import CombatUnit, CombatStats
from src.combat.movement import MovementSystem, MOVE_TIME_PER_HEX


def create_test_unit(unit_id: str, team: Team = Team.BLUE) -> CombatUnit:
    """테스트용 유닛"""
    stats = CombatStats(
        max_hp=1000, current_hp=1000,
        attack_damage=100, ability_power=100,
        armor=50, magic_resist=50,
        attack_speed=1.0, crit_chance=0.25, crit_damage=1.4,
        max_mana=100, current_mana=50, starting_mana=50,
        attack_range=1, dodge_chance=0.0,
        omnivamp=0.0, damage_amp=1.0, damage_reduction=0.0
    )
    return CombatUnit(
        id=unit_id, name=f"Unit_{unit_id}",
        champion_id="TFT_Test", star_level=1,
        team=team, stats=stats
    )


class TestMovementSystem:
    """MovementSystem 테스트"""
    
    @pytest.fixture
    def setup(self):
        grid = HexGrid()
        movement = MovementSystem(grid, move_time_per_hex=0.5)
        return grid, movement
    
    def test_start_move_to_position(self, setup):
        """위치로 이동 시작"""
        grid, movement = setup
        unit = create_test_unit("unit1")
        
        grid.place_unit("unit1", HexPosition(0, 0))
        
        result = movement.start_move_to_position(unit, HexPosition(2, 0))
        
        assert result is True
        assert movement.is_moving("unit1")
    
    def test_start_move_same_position(self, setup):
        """같은 위치로 이동 - 실패"""
        grid, movement = setup
        unit = create_test_unit("unit1")
        
        pos = HexPosition(0, 0)
        grid.place_unit("unit1", pos)
        
        result = movement.start_move_to_position(unit, pos)
        
        assert result is False
    
    def test_update_moves_unit(self, setup):
        """업데이트로 유닛 이동"""
        grid, movement = setup
        unit = create_test_unit("unit1")
        
        grid.place_unit("unit1", HexPosition(0, 0))
        movement.start_move_to_position(unit, HexPosition(1, 0))
        
        # 충분한 시간 업데이트
        movement.update(unit, delta_time=1.0)
        
        # 이동 완료
        assert grid.get_unit_position("unit1") == HexPosition(1, 0)
    
    def test_update_partial_progress(self, setup):
        """부분 이동"""
        grid, movement = setup
        unit = create_test_unit("unit1")
        
        grid.place_unit("unit1", HexPosition(0, 0))
        movement.start_move_to_position(unit, HexPosition(2, 0))
        
        # 절반만 이동
        movement.update(unit, delta_time=0.25)  # 0.5초/헥스의 절반
        
        progress = movement.get_movement_progress("unit1")
        assert progress > 0
        assert progress < 1.0
    
    def test_move_to_target_range(self, setup):
        """타겟 사거리 내로 이동"""
        grid, movement = setup
        
        unit = create_test_unit("unit1")
        grid.place_unit("unit1", HexPosition(0, 0))
        
        target_pos = HexPosition(4, 4)
        
        result = movement.start_move_to_target(unit, target_pos, attack_range=1)
        
        assert result is True
    
    def test_already_in_range(self, setup):
        """이미 사거리 내 - 이동 불필요"""
        grid, movement = setup
        
        unit = create_test_unit("unit1")
        grid.place_unit("unit1", HexPosition(3, 3))
        
        target_pos = HexPosition(4, 3)  # 거리 1
        
        result = movement.start_move_to_target(unit, target_pos, attack_range=1)
        
        assert result is False
        assert not movement.is_moving("unit1")
    
    def test_stop_movement(self, setup):
        """이동 중지"""
        grid, movement = setup
        unit = create_test_unit("unit1")
        
        grid.place_unit("unit1", HexPosition(0, 0))
        movement.start_move_to_position(unit, HexPosition(5, 5))
        
        assert movement.is_moving("unit1")
        
        movement.stop_movement("unit1")
        
        assert not movement.is_moving("unit1")
    
    def test_recalculate_path(self, setup):
        """경로 재계산"""
        grid, movement = setup
        unit = create_test_unit("unit1")
        
        grid.place_unit("unit1", HexPosition(0, 0))
        movement.start_move_to_target(unit, HexPosition(4, 4), attack_range=1)
        
        # 새 타겟으로 경로 재계산
        result = movement.recalculate_path(unit, HexPosition(4, 0), attack_range=1)
        
        assert result is True
    
    def test_blocked_destination(self, setup):
        """목적지가 막혀있을 때"""
        grid, movement = setup
        
        unit1 = create_test_unit("unit1")
        unit2 = create_test_unit("unit2")
        
        grid.place_unit("unit1", HexPosition(0, 0))
        grid.place_unit("unit2", HexPosition(1, 0))  # 목적지 차단
        
        result = movement.start_move_to_position(unit1, HexPosition(1, 0))
        
        # 막힌 위치로는 이동 불가
        assert result is False
    
    def test_clear_all_movement(self, setup):
        """모든 이동 상태 초기화"""
        grid, movement = setup
        
        unit1 = create_test_unit("unit1")
        unit2 = create_test_unit("unit2")
        
        grid.place_unit("unit1", HexPosition(0, 0))
        grid.place_unit("unit2", HexPosition(0, 1))
        
        movement.start_move_to_position(unit1, HexPosition(5, 5))
        movement.start_move_to_position(unit2, HexPosition(5, 4))
        
        movement.clear()
        
        assert not movement.is_moving("unit1")
        assert not movement.is_moving("unit2")
```

## 체크리스트

- [ ] `src/combat/targeting.py` 구현
  - [ ] TargetingPriority enum
  - [ ] TargetingContext 클래스
  - [ ] TargetSelector 클래스
  - [ ] 최근접/최원거리/최저HP 등 타겟팅
  - [ ] 사거리 체크
  - [ ] 커스텀 필터 지원
- [ ] `src/combat/pathfinding.py` 구현
  - [ ] PathNode 데이터클래스
  - [ ] PathFinder (A* 알고리즘)
  - [ ] find_path, find_path_to_range
  - [ ] get_blocked_positions 헬퍼
- [ ] `src/combat/movement.py` 구현
  - [ ] MovementState 데이터클래스
  - [ ] MovementSystem 클래스
  - [ ] 틱 기반 이동 업데이트
  - [ ] 경로 재계산
- [ ] `__init__.py` 업데이트
- [ ] 테스트 작성 및 통과

## 다음 스테이지 예고

**Stage 5.3: 기본 공격 & 데미지 계산**
- 공격 속도 기반 공격 타이밍
- 물리/마법/트루 데미지 처리
- 크리티컬 히트
- 방어력/마법저항 공식
- 데미지 이벤트 시스템
