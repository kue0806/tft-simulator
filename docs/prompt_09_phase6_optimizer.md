# Phase 6: Optimizer Module

## 목표
TFT 게임 내 의사결정을 최적화하는 추천 시스템을 구현합니다.
전투 시뮬레이션 결과를 활용하여 최적의 전략을 제안합니다.

## 모듈 구성

```
src/optimizer/
├── __init__.py
├── pick_advisor.py      # 상점 구매 추천
├── rolldown_planner.py  # 롤다운 타이밍/전략
├── comp_builder.py      # 조합 구성 추천
├── pivot_analyzer.py    # 피벗 분석
└── board_optimizer.py   # 포지셔닝 최적화
```

---

## 1. `src/optimizer/pick_advisor.py`

상점에서 어떤 챔피언을 구매할지 추천합니다.

```python
"""
상점 구매 추천 시스템
- 현재 조합에 맞는 챔피언 추천
- 업그레이드 우선순위
- 이코노미 고려
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum, auto

from src.core.game_state import PlayerState
from src.core.shop import Shop
from src.core.synergy_calculator import SynergyCalculator
from src.data.models.champion import Champion


class PickReason(Enum):
    """구매 추천 이유"""
    UPGRADE_2STAR = auto()      # 2성 업그레이드 가능
    UPGRADE_3STAR = auto()      # 3성 업그레이드 가능
    SYNERGY_ACTIVATE = auto()   # 시너지 활성화
    SYNERGY_UPGRADE = auto()    # 시너지 티어 업
    CORE_CARRY = auto()         # 핵심 캐리
    STRONG_UNIT = auto()        # 강한 유닛 (템포)
    ECONOMY_PAIR = auto()       # 페어 보유 (나중에 업그레이드)
    PIVOT_OPTION = auto()       # 피벗 옵션


@dataclass
class PickRecommendation:
    """구매 추천"""
    champion_id: str
    champion_name: str
    shop_index: int          # 상점 슬롯 (0-4)
    score: float             # 추천 점수 (높을수록 좋음)
    reasons: List[PickReason]
    synergy_delta: Dict[str, int]  # 시너지 변화
    cost: int
    
    # 업그레이드 정보
    copies_owned: int        # 현재 보유 수
    copies_needed: int       # 업그레이드까지 필요 수


@dataclass
class PickAdvice:
    """전체 구매 조언"""
    recommendations: List[PickRecommendation]
    should_refresh: bool     # 리롤 추천 여부
    refresh_reason: Optional[str]
    gold_to_save: int        # 유지해야 할 골드 (이자용)
    

class PickAdvisor:
    """
    상점 구매 추천기
    
    Usage:
        advisor = PickAdvisor()
        advice = advisor.analyze(player_state, shop)
        for rec in advice.recommendations:
            print(f"{rec.champion_name}: {rec.score:.1f}점 - {rec.reasons}")
    """
    
    def __init__(self, synergy_calculator: Optional[SynergyCalculator] = None):
        self.synergy_calc = synergy_calculator or SynergyCalculator()
        
        # 가중치 설정
        self.weights = {
            PickReason.UPGRADE_3STAR: 100,
            PickReason.UPGRADE_2STAR: 50,
            PickReason.CORE_CARRY: 40,
            PickReason.SYNERGY_ACTIVATE: 30,
            PickReason.SYNERGY_UPGRADE: 25,
            PickReason.STRONG_UNIT: 15,
            PickReason.ECONOMY_PAIR: 10,
            PickReason.PIVOT_OPTION: 5,
        }
    
    def analyze(
        self, 
        player: PlayerState, 
        shop: Shop,
        target_comp: Optional[List[str]] = None
    ) -> PickAdvice:
        """
        현재 상점 분석 및 추천
        
        Args:
            player: 플레이어 상태
            shop: 현재 상점
            target_comp: 목표 조합 (없으면 현재 보드 기반)
            
        Returns:
            구매 조언
        """
        recommendations = []
        
        for idx, slot in enumerate(shop.slots):
            if slot is None or slot.is_purchased:
                continue
            
            champion = slot.champion
            rec = self._evaluate_champion(
                champion, idx, player, target_comp
            )
            if rec:
                recommendations.append(rec)
        
        # 점수순 정렬
        recommendations.sort(key=lambda r: r.score, reverse=True)
        
        # 리롤 추천 여부
        should_refresh, refresh_reason = self._should_refresh(
            recommendations, player
        )
        
        # 이자용 골드
        gold_to_save = self._calculate_gold_to_save(player)
        
        return PickAdvice(
            recommendations=recommendations,
            should_refresh=should_refresh,
            refresh_reason=refresh_reason,
            gold_to_save=gold_to_save
        )
    
    def _evaluate_champion(
        self,
        champion: Champion,
        shop_index: int,
        player: PlayerState,
        target_comp: Optional[List[str]]
    ) -> Optional[PickRecommendation]:
        """개별 챔피언 평가"""
        reasons = []
        score = 0.0
        
        # 현재 보유 수 계산
        copies_owned = self._count_copies(player, champion.champion_id)
        
        # 1. 업그레이드 체크
        if copies_owned >= 6:  # 3성 가능
            reasons.append(PickReason.UPGRADE_3STAR)
            score += self.weights[PickReason.UPGRADE_3STAR]
        elif copies_owned >= 2:  # 2성 가능
            reasons.append(PickReason.UPGRADE_2STAR)
            score += self.weights[PickReason.UPGRADE_2STAR]
        elif copies_owned >= 1:  # 페어
            reasons.append(PickReason.ECONOMY_PAIR)
            score += self.weights[PickReason.ECONOMY_PAIR]
        
        # 2. 시너지 체크
        synergy_delta = self._calculate_synergy_delta(player, champion)
        for trait, delta in synergy_delta.items():
            if delta > 0:
                # 새 시너지 활성화 또는 티어 업
                if self._is_new_activation(player, trait, champion):
                    reasons.append(PickReason.SYNERGY_ACTIVATE)
                    score += self.weights[PickReason.SYNERGY_ACTIVATE]
                else:
                    reasons.append(PickReason.SYNERGY_UPGRADE)
                    score += self.weights[PickReason.SYNERGY_UPGRADE]
        
        # 3. 목표 조합 체크
        if target_comp and champion.champion_id in target_comp:
            reasons.append(PickReason.CORE_CARRY)
            score += self.weights[PickReason.CORE_CARRY]
        
        # 4. 강한 유닛 (높은 코스트 = 강함)
        if champion.cost >= 4:
            reasons.append(PickReason.STRONG_UNIT)
            score += self.weights[PickReason.STRONG_UNIT] * (champion.cost / 5)
        
        # 점수가 없으면 추천 안함
        if score <= 0:
            return None
        
        # 코스트 대비 효율 조정
        score = score / champion.cost
        
        return PickRecommendation(
            champion_id=champion.champion_id,
            champion_name=champion.name,
            shop_index=shop_index,
            score=score,
            reasons=reasons,
            synergy_delta=synergy_delta,
            cost=champion.cost,
            copies_owned=copies_owned,
            copies_needed=3 - (copies_owned % 3) if copies_owned < 9 else 0
        )
    
    def _count_copies(self, player: PlayerState, champion_id: str) -> int:
        """보유 챔피언 수 (벤치 + 보드)"""
        count = 0
        
        # 벤치
        for instance in player.units.bench:
            if instance and instance.champion.champion_id == champion_id:
                count += 3 ** (instance.star_level - 1)
        
        # 보드
        for instance in player.units.board.values():
            if instance.champion.champion_id == champion_id:
                count += 3 ** (instance.star_level - 1)
        
        return count
    
    def _calculate_synergy_delta(
        self, 
        player: PlayerState, 
        champion: Champion
    ) -> Dict[str, int]:
        """시너지 변화 계산"""
        # 현재 시너지
        current = player.units.get_active_synergies()
        
        # 챔피언 추가 후 시너지 (preview)
        preview = self.synergy_calc.preview_add_champion(
            player.units.board.values(),
            champion,
            star_level=1
        )
        
        delta = {}
        for trait_id, data in preview.items():
            current_count = current.get(trait_id, {}).get('count', 0)
            new_count = data.get('count', 0)
            if new_count > current_count:
                delta[trait_id] = new_count - current_count
        
        return delta
    
    def _is_new_activation(
        self, 
        player: PlayerState, 
        trait_id: str,
        champion: Champion
    ) -> bool:
        """새 시너지가 활성화되는지"""
        current = player.units.get_active_synergies()
        return trait_id not in current or not current[trait_id].get('is_active', False)
    
    def _should_refresh(
        self, 
        recommendations: List[PickRecommendation],
        player: PlayerState
    ) -> Tuple[bool, Optional[str]]:
        """리롤 추천 여부"""
        # 좋은 추천이 없으면 리롤
        if not recommendations:
            return True, "상점에 유용한 챔피언이 없습니다"
        
        # 최고 추천 점수가 낮으면 리롤
        if recommendations[0].score < 10:
            return True, "추천 점수가 낮습니다"
        
        # 골드가 충분하고 롤다운 중이면 리롤
        # (이 부분은 RolldownPlanner와 연동)
        
        return False, None
    
    def _calculate_gold_to_save(self, player: PlayerState) -> int:
        """이자용 유지 골드"""
        # 10, 20, 30, 40, 50 단위로 유지
        current_gold = player.gold
        interest_threshold = (current_gold // 10) * 10
        
        # 다음 이자 구간까지 남은 골드
        next_threshold = ((current_gold // 10) + 1) * 10
        if next_threshold - current_gold <= 3:
            return next_threshold
        
        return interest_threshold
```

---

## 2. `src/optimizer/rolldown_planner.py`

최적의 롤다운 타이밍과 전략을 제안합니다.

```python
"""
롤다운 전략 플래너
- 언제 롤다운할지
- 얼마나 롤할지
- 목표 레벨/유닛
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum, auto

from src.core.game_state import PlayerState, GameState
from src.core.economy import EconomyCalculator
from src.core.probability import ProbabilityCalculator
from src.core.constants import LEVEL_COSTS, SHOP_ODDS


class RolldownStrategy(Enum):
    """롤다운 전략"""
    FAST_8 = "fast_8"           # 빠른 8렙 후 롤다운
    FAST_9 = "fast_9"           # 9렙 도달 후 롤다운
    SLOW_ROLL_6 = "slow_roll_6" # 6렙에서 슬로우 롤 (1코 3성)
    SLOW_ROLL_7 = "slow_roll_7" # 7렙에서 슬로우 롤 (2코 3성)
    SLOW_ROLL_8 = "slow_roll_8" # 8렙에서 슬로우 롤 (3코 3성)
    ALL_IN = "all_in"           # 올인 (HP 위험)
    SAVE = "save"               # 저장 (이코노미 빌드)


@dataclass
class RolldownTiming:
    """롤다운 타이밍"""
    stage: str                  # "4-1", "4-2" 등
    level: int                  # 목표 레벨
    gold_threshold: int         # 롤다운 시작 골드
    description: str


@dataclass
class RolldownPlan:
    """롤다운 계획"""
    strategy: RolldownStrategy
    current_phase: str          # "leveling", "rolling", "stabilized"
    
    # 타이밍
    recommended_timing: RolldownTiming
    is_rolldown_now: bool       # 지금 롤다운 해야 하는지
    
    # 예산
    roll_budget: int            # 롤에 쓸 골드
    level_budget: int           # 레벨업에 쓸 골드
    save_amount: int            # 저장할 골드
    
    # 목표
    target_units: List[str]     # 찾을 유닛들
    target_star_levels: Dict[str, int]  # 목표 성급
    
    # 확률
    hit_probability: float      # 목표 히트 확률
    expected_rolls: int         # 예상 롤 횟수
    
    # 조언
    advice: List[str]


class RolldownPlanner:
    """
    롤다운 전략 플래너
    
    Usage:
        planner = RolldownPlanner()
        plan = planner.create_plan(player, game_state, target_units)
    """
    
    # 주요 롤다운 타이밍
    KEY_TIMINGS = {
        "3-2": RolldownTiming("3-2", 6, 30, "초반 안정화, 2코 캐리"),
        "4-1": RolldownTiming("4-1", 7, 50, "중반 파워스파이크"),
        "4-2": RolldownTiming("4-2", 8, 50, "4코 캐리 롤다운"),
        "4-5": RolldownTiming("4-5", 8, 30, "4코 완성/올인"),
        "5-1": RolldownTiming("5-1", 8, 50, "레이트 롤다운"),
        "5-2": RolldownTiming("5-2", 9, 50, "5코 캐리"),
    }
    
    def __init__(self):
        self.economy = EconomyCalculator()
        self.probability = ProbabilityCalculator()
    
    def create_plan(
        self,
        player: PlayerState,
        game: GameState,
        target_units: List[str],
        target_stars: Optional[Dict[str, int]] = None
    ) -> RolldownPlan:
        """
        롤다운 계획 생성
        
        Args:
            player: 플레이어 상태
            game: 게임 상태
            target_units: 목표 유닛들
            target_stars: 목표 성급 (기본 2성)
        """
        if target_stars is None:
            target_stars = {u: 2 for u in target_units}
        
        # 전략 결정
        strategy = self._determine_strategy(player, game, target_units)
        
        # 현재 페이즈
        current_phase = self._get_current_phase(player, strategy)
        
        # 타이밍 추천
        timing = self._recommend_timing(player, game, strategy)
        
        # 지금 롤다운해야 하는지
        is_rolldown_now = self._should_rolldown_now(player, game, strategy)
        
        # 예산 분배
        roll_budget, level_budget, save = self._allocate_budget(
            player, game, strategy, is_rolldown_now
        )
        
        # 히트 확률 계산
        hit_prob, expected_rolls = self._calculate_hit_probability(
            player, target_units, target_stars
        )
        
        # 조언 생성
        advice = self._generate_advice(
            player, game, strategy, is_rolldown_now, hit_prob
        )
        
        return RolldownPlan(
            strategy=strategy,
            current_phase=current_phase,
            recommended_timing=timing,
            is_rolldown_now=is_rolldown_now,
            roll_budget=roll_budget,
            level_budget=level_budget,
            save_amount=save,
            target_units=target_units,
            target_star_levels=target_stars,
            hit_probability=hit_prob,
            expected_rolls=expected_rolls,
            advice=advice
        )
    
    def _determine_strategy(
        self,
        player: PlayerState,
        game: GameState,
        target_units: List[str]
    ) -> RolldownStrategy:
        """전략 결정"""
        # HP가 위험하면 올인
        if player.hp <= 30:
            return RolldownStrategy.ALL_IN
        
        # 타겟 유닛 코스트 분석
        avg_cost = self._get_average_target_cost(target_units)
        
        # 1코 3성 노리면 슬로우롤 6
        if avg_cost <= 1.5:
            return RolldownStrategy.SLOW_ROLL_6
        
        # 2코 3성 노리면 슬로우롤 7
        if avg_cost <= 2.5:
            return RolldownStrategy.SLOW_ROLL_7
        
        # 3코 캐리면 슬로우롤 8 또는 패스트 8
        if avg_cost <= 3.5:
            if player.hp >= 70:
                return RolldownStrategy.FAST_8
            return RolldownStrategy.SLOW_ROLL_8
        
        # 4-5코 캐리면 패스트 8/9
        if player.hp >= 50 and player.gold >= 50:
            return RolldownStrategy.FAST_9
        
        return RolldownStrategy.FAST_8
    
    def _get_current_phase(
        self, 
        player: PlayerState, 
        strategy: RolldownStrategy
    ) -> str:
        """현재 페이즈"""
        target_level = self._get_target_level(strategy)
        
        if player.level < target_level:
            return "leveling"
        elif player.gold > 20:
            return "rolling"
        else:
            return "stabilized"
    
    def _recommend_timing(
        self,
        player: PlayerState,
        game: GameState,
        strategy: RolldownStrategy
    ) -> RolldownTiming:
        """타이밍 추천"""
        stage = game.stage_manager.get_stage_string()
        
        if strategy == RolldownStrategy.SLOW_ROLL_6:
            return RolldownTiming("3-2+", 6, 50, "50골드 이상 유지하며 롤")
        elif strategy == RolldownStrategy.SLOW_ROLL_7:
            return RolldownTiming("4-1+", 7, 50, "50골드 이상 유지하며 롤")
        elif strategy == RolldownStrategy.FAST_8:
            return self.KEY_TIMINGS.get("4-2", self.KEY_TIMINGS["4-1"])
        elif strategy == RolldownStrategy.FAST_9:
            return self.KEY_TIMINGS.get("5-2", self.KEY_TIMINGS["5-1"])
        elif strategy == RolldownStrategy.ALL_IN:
            return RolldownTiming(stage, player.level, 0, "지금 올인!")
        
        return self.KEY_TIMINGS.get("4-2", self.KEY_TIMINGS["4-1"])
    
    def _should_rolldown_now(
        self,
        player: PlayerState,
        game: GameState,
        strategy: RolldownStrategy
    ) -> bool:
        """지금 롤다운해야 하는지"""
        if strategy == RolldownStrategy.ALL_IN:
            return True
        
        if strategy in [RolldownStrategy.SLOW_ROLL_6, 
                       RolldownStrategy.SLOW_ROLL_7,
                       RolldownStrategy.SLOW_ROLL_8]:
            target_level = self._get_target_level(strategy)
            return player.level >= target_level and player.gold > 50
        
        stage = game.stage_manager.get_stage_string()
        target_level = self._get_target_level(strategy)
        
        # 목표 레벨 도달 + 적절한 스테이지
        if player.level >= target_level:
            key_stages = ["4-1", "4-2", "4-5", "5-1", "5-2"]
            return stage in key_stages
        
        return False
    
    def _allocate_budget(
        self,
        player: PlayerState,
        game: GameState,
        strategy: RolldownStrategy,
        is_rolldown: bool
    ) -> Tuple[int, int, int]:
        """예산 분배: (롤, 레벨업, 저장)"""
        gold = player.gold
        
        if strategy == RolldownStrategy.ALL_IN:
            return gold, 0, 0
        
        if strategy == RolldownStrategy.SAVE:
            return 0, 0, gold
        
        # 슬로우롤: 50골드 이상 유지
        if strategy in [RolldownStrategy.SLOW_ROLL_6,
                       RolldownStrategy.SLOW_ROLL_7,
                       RolldownStrategy.SLOW_ROLL_8]:
            roll_budget = max(0, gold - 50)
            return roll_budget, 0, 50
        
        # 레벨업 비용
        target_level = self._get_target_level(strategy)
        level_cost = 0
        if player.level < target_level:
            level_cost = self._calculate_level_cost(player, target_level)
        
        if is_rolldown:
            # 롤다운 시: 레벨업 후 남은 돈으로 롤
            remaining = gold - level_cost
            save = min(10, remaining)  # 최소 10골드 유지
            roll_budget = max(0, remaining - save)
            return roll_budget, level_cost, save
        else:
            # 저장 중: 이자 유지
            save = min(50, gold)
            level_budget = min(level_cost, gold - save)
            return 0, level_budget, save
    
    def _calculate_hit_probability(
        self,
        player: PlayerState,
        target_units: List[str],
        target_stars: Dict[str, int]
    ) -> Tuple[float, int]:
        """히트 확률 및 예상 롤 횟수"""
        # 간단한 근사치 계산
        # 실제로는 ProbabilityCalculator 사용
        
        level = player.level
        odds = SHOP_ODDS.get(level, {})
        
        # 타겟 유닛의 평균 코스트별 확률
        total_prob = 0.0
        for unit in target_units:
            cost = self._get_unit_cost(unit)
            cost_odds = odds.get(cost, 0) / 100
            # 해당 코스트 풀에서 특정 유닛 확률 (대략)
            unit_prob = cost_odds * 0.1  # 풀 크기에 따라 조정 필요
            total_prob += unit_prob
        
        # 5개 슬롯
        per_roll_prob = 1 - (1 - total_prob) ** 5
        
        if per_roll_prob > 0:
            expected_rolls = int(1 / per_roll_prob)
        else:
            expected_rolls = 999
        
        return per_roll_prob, expected_rolls
    
    def _generate_advice(
        self,
        player: PlayerState,
        game: GameState,
        strategy: RolldownStrategy,
        is_rolldown: bool,
        hit_prob: float
    ) -> List[str]:
        """조언 생성"""
        advice = []
        
        # 전략 설명
        strategy_desc = {
            RolldownStrategy.FAST_8: "8렙에서 4코스트 캐리를 찾으세요",
            RolldownStrategy.FAST_9: "9렙에서 5코스트를 찾으세요",
            RolldownStrategy.SLOW_ROLL_6: "6렙에서 50골드 이상 유지하며 롤하세요",
            RolldownStrategy.SLOW_ROLL_7: "7렙에서 50골드 이상 유지하며 롤하세요",
            RolldownStrategy.SLOW_ROLL_8: "8렙에서 50골드 이상 유지하며 롤하세요",
            RolldownStrategy.ALL_IN: "지금 모든 골드를 써서 덱을 완성하세요!",
            RolldownStrategy.SAVE: "이코노미를 모으세요",
        }
        advice.append(strategy_desc.get(strategy, ""))
        
        # HP 경고
        if player.hp <= 30:
            advice.append("⚠️ HP가 위험합니다! 빨리 덱을 완성하세요")
        elif player.hp <= 50:
            advice.append("HP가 낮습니다. 곧 롤다운이 필요합니다")
        
        # 히트 확률
        if hit_prob < 0.1:
            advice.append("히트 확률이 낮습니다. 레벨업을 고려하세요")
        elif hit_prob > 0.3:
            advice.append("히트 확률이 좋습니다!")
        
        # 이코노미
        if player.gold >= 50 and not is_rolldown:
            advice.append("50골드 이자를 유지하세요")
        
        return advice
    
    def _get_target_level(self, strategy: RolldownStrategy) -> int:
        """전략별 목표 레벨"""
        return {
            RolldownStrategy.SLOW_ROLL_6: 6,
            RolldownStrategy.SLOW_ROLL_7: 7,
            RolldownStrategy.SLOW_ROLL_8: 8,
            RolldownStrategy.FAST_8: 8,
            RolldownStrategy.FAST_9: 9,
            RolldownStrategy.ALL_IN: 8,
            RolldownStrategy.SAVE: 8,
        }.get(strategy, 8)
    
    def _calculate_level_cost(self, player: PlayerState, target: int) -> int:
        """목표 레벨까지 비용"""
        total = 0
        for lvl in range(player.level, target):
            xp_needed = LEVEL_COSTS.get(lvl + 1, 100) - player.xp
            total += (xp_needed + 3) // 4 * 4  # 4골드씩 구매
        return total
    
    def _get_average_target_cost(self, target_units: List[str]) -> float:
        """타겟 유닛 평균 코스트"""
        if not target_units:
            return 3.0
        costs = [self._get_unit_cost(u) for u in target_units]
        return sum(costs) / len(costs)
    
    def _get_unit_cost(self, champion_id: str) -> int:
        """유닛 코스트 조회"""
        # 실제로는 ChampionLoader 사용
        # 여기서는 간단히 처리
        return 3  # 기본값
```

---

## 3. `src/optimizer/comp_builder.py`

조합 구성을 추천합니다.

```python
"""
조합 빌더
- 메타 조합 추천
- 현재 보드에서 확장 가능한 조합
- 시너지 최적화
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from enum import Enum, auto

from src.core.game_state import PlayerState
from src.core.synergy_calculator import SynergyCalculator
from src.combat.simulation import CombatSimulator
from src.data.loaders.champion_loader import ChampionLoader
from src.data.loaders.trait_loader import TraitLoader


class CompStyle(Enum):
    """조합 스타일"""
    REROLL = "reroll"       # 리롤 (저코 3성)
    STANDARD = "standard"   # 표준 (4코 캐리)
    FAST_9 = "fast_9"       # 패스트 9 (5코 캐리)
    FLEX = "flex"           # 유동적


@dataclass
class CompTemplate:
    """조합 템플릿"""
    name: str
    style: CompStyle
    core_units: List[str]           # 필수 유닛
    flex_units: List[str]           # 선택 유닛
    carry: str                       # 메인 캐리
    items_priority: Dict[str, List[str]]  # 유닛별 아이템 우선순위
    
    # 시너지 목표
    target_synergies: Dict[str, int]  # trait_id -> 목표 수
    
    # 메타 정보
    tier: str                        # "S", "A", "B", "C"
    difficulty: str                  # "easy", "medium", "hard"
    description: str
    
    # 파워 스파이크
    power_spikes: List[str]          # "4-1", "4-5" 등


@dataclass
class CompRecommendation:
    """조합 추천"""
    template: CompTemplate
    match_score: float              # 현재 보드와의 일치도 (0-100)
    missing_units: List[str]        # 필요한 유닛
    current_units: List[str]        # 이미 있는 유닛
    transition_cost: int            # 전환 비용 (예상 롤 수)
    estimated_strength: float       # 예상 강도


class CompBuilder:
    """
    조합 빌더
    
    Usage:
        builder = CompBuilder()
        recommendations = builder.recommend(player_state)
    """
    
    def __init__(self):
        self.synergy_calc = SynergyCalculator()
        self.champion_loader = ChampionLoader()
        self.trait_loader = TraitLoader()
        
        # 메타 조합 템플릿 (실제로는 데이터 파일에서 로드)
        self.templates = self._load_templates()
    
    def recommend(
        self,
        player: PlayerState,
        top_n: int = 3,
        style_filter: Optional[CompStyle] = None
    ) -> List[CompRecommendation]:
        """
        현재 상태에서 추천 조합
        
        Args:
            player: 플레이어 상태
            top_n: 상위 N개 추천
            style_filter: 특정 스타일만
        """
        recommendations = []
        
        current_units = self._get_current_units(player)
        
        for template in self.templates:
            if style_filter and template.style != style_filter:
                continue
            
            rec = self._evaluate_template(template, player, current_units)
            recommendations.append(rec)
        
        # 매치 점수순 정렬
        recommendations.sort(key=lambda r: r.match_score, reverse=True)
        
        return recommendations[:top_n]
    
    def build_from_scratch(
        self,
        target_traits: Dict[str, int],
        level: int = 8
    ) -> List[str]:
        """
        시너지 목표로 조합 구성
        
        Args:
            target_traits: 목표 시너지
            level: 배치 가능 유닛 수
        """
        selected = []
        remaining_traits = target_traits.copy()
        
        # 그리디하게 유닛 선택
        all_champions = self.champion_loader.load_all()
        
        while len(selected) < level and remaining_traits:
            best_unit = None
            best_score = 0
            
            for champ in all_champions:
                if champ.champion_id in selected:
                    continue
                
                score = self._calculate_trait_coverage(
                    champ, remaining_traits
                )
                if score > best_score:
                    best_score = score
                    best_unit = champ
            
            if best_unit is None:
                break
            
            selected.append(best_unit.champion_id)
            # 커버된 특성 업데이트
            for trait in best_unit.traits:
                if trait in remaining_traits:
                    remaining_traits[trait] -= 1
                    if remaining_traits[trait] <= 0:
                        del remaining_traits[trait]
        
        return selected
    
    def suggest_additions(
        self,
        player: PlayerState,
        slots_available: int = 1
    ) -> List[str]:
        """
        현재 보드에 추가할 유닛 추천
        """
        current_synergies = player.units.get_active_synergies()
        suggestions = []
        
        all_champions = self.champion_loader.load_all()
        
        for champ in all_champions:
            # 이미 보드에 있으면 스킵
            if self._is_on_board(player, champ.champion_id):
                continue
            
            # 시너지 향상 점수
            score = self._calculate_addition_value(
                champ, current_synergies, player
            )
            suggestions.append((champ.champion_id, score))
        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in suggestions[:slots_available * 3]]
    
    def _evaluate_template(
        self,
        template: CompTemplate,
        player: PlayerState,
        current_units: Set[str]
    ) -> CompRecommendation:
        """템플릿 평가"""
        core_set = set(template.core_units)
        
        # 매치되는 유닛
        matched = current_units & core_set
        missing = core_set - current_units
        
        # 매치 점수
        match_score = len(matched) / len(core_set) * 100 if core_set else 0
        
        # 전환 비용 (대략적)
        transition_cost = len(missing) * 10  # 유닛당 10롤 가정
        
        # 예상 강도
        estimated_strength = self._estimate_comp_strength(template)
        
        return CompRecommendation(
            template=template,
            match_score=match_score,
            missing_units=list(missing),
            current_units=list(matched),
            transition_cost=transition_cost,
            estimated_strength=estimated_strength
        )
    
    def _get_current_units(self, player: PlayerState) -> Set[str]:
        """현재 보드 유닛"""
        units = set()
        for instance in player.units.board.values():
            units.add(instance.champion.champion_id)
        return units
    
    def _is_on_board(self, player: PlayerState, champion_id: str) -> bool:
        """보드에 있는지"""
        for instance in player.units.board.values():
            if instance.champion.champion_id == champion_id:
                return True
        return False
    
    def _calculate_trait_coverage(
        self,
        champion,
        remaining_traits: Dict[str, int]
    ) -> float:
        """특성 커버리지 점수"""
        score = 0
        for trait in champion.traits:
            if trait in remaining_traits:
                score += remaining_traits[trait]
        return score
    
    def _calculate_addition_value(
        self,
        champion,
        current_synergies: Dict,
        player: PlayerState
    ) -> float:
        """추가 가치 계산"""
        score = 0.0
        
        for trait in champion.traits:
            if trait in current_synergies:
                # 이미 활성화된 시너지 강화
                score += 10
            else:
                # 새 시너지
                score += 5
        
        # 코스트 효율
        score /= champion.cost
        
        return score
    
    def _estimate_comp_strength(self, template: CompTemplate) -> float:
        """조합 강도 추정"""
        tier_scores = {"S": 95, "A": 85, "B": 75, "C": 65}
        return tier_scores.get(template.tier, 70)
    
    def _load_templates(self) -> List[CompTemplate]:
        """메타 조합 템플릿 로드"""
        # 실제로는 JSON/데이터 파일에서 로드
        # 여기서는 예시 템플릿
        return [
            CompTemplate(
                name="Arcana Reroll",
                style=CompStyle.REROLL,
                core_units=["TFT16_Ahri", "TFT16_Twitch", "TFT16_Ziggs"],
                flex_units=["TFT16_Xerath", "TFT16_TahmKench"],
                carry="TFT16_Ahri",
                items_priority={
                    "TFT16_Ahri": ["JeweledGauntlet", "GiantSlayer", "Quicksilver"]
                },
                target_synergies={"Arcana": 6, "Mage": 2},
                tier="A",
                difficulty="medium",
                description="아리 3성을 메인 캐리로",
                power_spikes=["3-2", "4-1"]
            ),
            # 더 많은 템플릿...
        ]
```

---

## 4. `src/optimizer/pivot_analyzer.py`

피벗 타이밍과 방법을 분석합니다.

```python
"""
피벗 분석기
- 언제 피벗할지
- 어디로 피벗할지
- 피벗 비용 분석
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum, auto

from src.core.game_state import PlayerState, GameState
from .comp_builder import CompBuilder, CompTemplate, CompRecommendation


class PivotReason(Enum):
    """피벗 이유"""
    CONTESTED = auto()          # 다른 플레이어와 경합
    LOW_ROLLS = auto()          # 롤 운이 안 좋음
    HP_CRITICAL = auto()        # HP 위험
    BETTER_ITEMS = auto()       # 아이템이 다른 캐리에 적합
    HIGHROLL = auto()           # 고롤 (좋은 유닛 발견)
    LOBBY_READ = auto()         # 로비 상황 분석


@dataclass
class PivotOption:
    """피벗 옵션"""
    target_comp: CompTemplate
    from_comp: Optional[CompTemplate]
    
    # 전환 분석
    shared_units: List[str]     # 공유 유닛 (팔 필요 없음)
    units_to_sell: List[str]    # 팔아야 할 유닛
    units_to_buy: List[str]     # 사야 할 유닛
    
    # 비용
    gold_loss: int              # 판매로 인한 골드 손실
    roll_cost: int              # 예상 롤 비용
    total_cost: int             # 총 비용
    
    # 평가
    success_probability: float  # 성공 확률
    risk_level: str             # "low", "medium", "high"
    
    reasons: List[PivotReason]


@dataclass
class PivotAdvice:
    """피벗 조언"""
    should_pivot: bool
    urgency: str                # "immediate", "soon", "optional"
    current_comp_health: float  # 현재 조합 건강도 (0-100)
    options: List[PivotOption]
    recommendation: Optional[PivotOption]
    explanation: str


class PivotAnalyzer:
    """
    피벗 분석기
    
    Usage:
        analyzer = PivotAnalyzer()
        advice = analyzer.analyze(player, game, current_comp)
    """
    
    def __init__(self):
        self.comp_builder = CompBuilder()
    
    def analyze(
        self,
        player: PlayerState,
        game: GameState,
        current_comp: Optional[CompTemplate] = None,
        contested_units: Optional[List[str]] = None
    ) -> PivotAdvice:
        """
        피벗 분석
        
        Args:
            player: 플레이어 상태
            game: 게임 상태
            current_comp: 현재 목표 조합
            contested_units: 경합 중인 유닛들
        """
        # 현재 조합 건강도
        comp_health = self._evaluate_comp_health(
            player, current_comp, contested_units
        )
        
        # 피벗 필요 여부
        should_pivot, urgency, reasons = self._should_pivot(
            player, comp_health, contested_units
        )
        
        # 피벗 옵션 생성
        options = []
        if should_pivot:
            options = self._generate_pivot_options(
                player, current_comp, reasons
            )
        
        # 최선 추천
        recommendation = options[0] if options else None
        
        # 설명 생성
        explanation = self._generate_explanation(
            should_pivot, urgency, comp_health, reasons
        )
        
        return PivotAdvice(
            should_pivot=should_pivot,
            urgency=urgency,
            current_comp_health=comp_health,
            options=options,
            recommendation=recommendation,
            explanation=explanation
        )
    
    def calculate_pivot_cost(
        self,
        player: PlayerState,
        from_comp: CompTemplate,
        to_comp: CompTemplate
    ) -> PivotOption:
        """피벗 비용 계산"""
        current_units = set()
        for inst in player.units.board.values():
            current_units.add(inst.champion.champion_id)
        
        target_units = set(to_comp.core_units)
        
        shared = current_units & target_units
        to_sell = current_units - target_units
        to_buy = target_units - current_units
        
        # 골드 손실 (판매는 구매가의 일부만 회수)
        gold_loss = self._calculate_sell_loss(player, list(to_sell))
        
        # 롤 비용
        roll_cost = len(to_buy) * 15  # 유닛당 약 15골드
        
        # 성공 확률
        success_prob = self._estimate_success_probability(
            player, to_buy
        )
        
        # 리스크
        if success_prob > 0.7:
            risk = "low"
        elif success_prob > 0.4:
            risk = "medium"
        else:
            risk = "high"
        
        return PivotOption(
            target_comp=to_comp,
            from_comp=from_comp,
            shared_units=list(shared),
            units_to_sell=list(to_sell),
            units_to_buy=list(to_buy),
            gold_loss=gold_loss,
            roll_cost=roll_cost,
            total_cost=gold_loss + roll_cost,
            success_probability=success_prob,
            risk_level=risk,
            reasons=[]
        )
    
    def _evaluate_comp_health(
        self,
        player: PlayerState,
        current_comp: Optional[CompTemplate],
        contested_units: Optional[List[str]]
    ) -> float:
        """조합 건강도 평가 (0-100)"""
        if current_comp is None:
            return 50.0
        
        health = 100.0
        
        # 코어 유닛 보유율
        owned_cores = sum(
            1 for u in current_comp.core_units
            if self._owns_unit(player, u)
        )
        core_ratio = owned_cores / len(current_comp.core_units)
        health *= core_ratio
        
        # 경합도
        if contested_units:
            contested_cores = sum(
                1 for u in current_comp.core_units
                if u in contested_units
            )
            if contested_cores > 0:
                health -= contested_cores * 15
        
        # 2성/3성 진행도
        upgrade_progress = self._calculate_upgrade_progress(player)
        health += upgrade_progress * 20
        
        return max(0, min(100, health))
    
    def _should_pivot(
        self,
        player: PlayerState,
        comp_health: float,
        contested_units: Optional[List[str]]
    ) -> Tuple[bool, str, List[PivotReason]]:
        """피벗 필요 여부"""
        reasons = []
        
        # HP 위험
        if player.hp <= 30:
            reasons.append(PivotReason.HP_CRITICAL)
            return True, "immediate", reasons
        
        # 조합 건강도 낮음
        if comp_health < 40:
            reasons.append(PivotReason.LOW_ROLLS)
            return True, "soon", reasons
        
        # 심한 경합
        if contested_units and len(contested_units) >= 3:
            reasons.append(PivotReason.CONTESTED)
            return True, "soon", reasons
        
        # 건강하면 피벗 불필요
        if comp_health >= 70:
            return False, "none", []
        
        return False, "optional", reasons
    
    def _generate_pivot_options(
        self,
        player: PlayerState,
        current_comp: Optional[CompTemplate],
        reasons: List[PivotReason]
    ) -> List[PivotOption]:
        """피벗 옵션 생성"""
        # CompBuilder로 추천 조합 받기
        recommendations = self.comp_builder.recommend(player, top_n=5)
        
        options = []
        for rec in recommendations:
            if current_comp and rec.template.name == current_comp.name:
                continue  # 현재 조합 제외
            
            option = self.calculate_pivot_cost(
                player, current_comp, rec.template
            )
            option.reasons = reasons
            options.append(option)
        
        # 비용 대비 성공률로 정렬
        options.sort(
            key=lambda o: o.success_probability / (o.total_cost + 1),
            reverse=True
        )
        
        return options[:3]
    
    def _owns_unit(self, player: PlayerState, champion_id: str) -> bool:
        """유닛 보유 여부"""
        for inst in player.units.board.values():
            if inst.champion.champion_id == champion_id:
                return True
        for inst in player.units.bench:
            if inst and inst.champion.champion_id == champion_id:
                return True
        return False
    
    def _calculate_sell_loss(
        self,
        player: PlayerState,
        units_to_sell: List[str]
    ) -> int:
        """판매 손실 계산"""
        loss = 0
        for unit_id in units_to_sell:
            # 구매가 - 판매가
            # 실제로는 인스턴스의 성급을 고려해야 함
            loss += 1  # 간단히 1골드 손실 가정
        return loss
    
    def _estimate_success_probability(
        self,
        player: PlayerState,
        units_to_buy: List[str]
    ) -> float:
        """피벗 성공 확률"""
        if not units_to_buy:
            return 1.0
        
        # 골드와 남은 유닛 수로 추정
        gold = player.gold
        rolls_available = gold // 2
        
        # 유닛당 약 10롤 필요 가정
        needed_rolls = len(units_to_buy) * 10
        
        if rolls_available >= needed_rolls:
            return 0.8
        elif rolls_available >= needed_rolls * 0.5:
            return 0.5
        else:
            return 0.2
    
    def _calculate_upgrade_progress(self, player: PlayerState) -> float:
        """업그레이드 진행도 (0-1)"""
        total_stars = 0
        unit_count = 0
        
        for inst in player.units.board.values():
            total_stars += inst.star_level
            unit_count += 1
        
        if unit_count == 0:
            return 0.0
        
        avg_stars = total_stars / unit_count
        return (avg_stars - 1) / 2  # 1성=0, 2성=0.5, 3성=1
    
    def _generate_explanation(
        self,
        should_pivot: bool,
        urgency: str,
        comp_health: float,
        reasons: List[PivotReason]
    ) -> str:
        """설명 생성"""
        if not should_pivot:
            return f"현재 조합이 건강합니다 ({comp_health:.0f}%). 계속 진행하세요."
        
        reason_texts = {
            PivotReason.CONTESTED: "다른 플레이어들과 유닛 경합 중",
            PivotReason.LOW_ROLLS: "핵심 유닛을 찾지 못함",
            PivotReason.HP_CRITICAL: "HP가 위험",
            PivotReason.BETTER_ITEMS: "아이템이 다른 캐리에 적합",
            PivotReason.HIGHROLL: "더 좋은 유닛 발견",
            PivotReason.LOBBY_READ: "로비 상황상 유리",
        }
        
        reason_str = ", ".join(reason_texts.get(r, str(r)) for r in reasons)
        
        urgency_text = {
            "immediate": "즉시",
            "soon": "곧",
            "optional": "선택적으로"
        }
        
        return f"{urgency_text[urgency]} 피벗을 고려하세요. 이유: {reason_str}"
```

---

## 5. `src/optimizer/board_optimizer.py`

포지셔닝을 최적화합니다.

```python
"""
보드 포지셔닝 최적화
- 유닛 배치 최적화
- 상대 조합에 따른 대응 배치
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import random

from src.combat.hex_grid import HexGrid, HexPosition, Team
from src.combat.simulation import CombatSimulator, SimulationConfig
from src.core.game_state import PlayerState


@dataclass
class PositionScore:
    """포지션 점수"""
    position: HexPosition
    unit_id: str
    score: float
    reasons: List[str]


@dataclass
class BoardLayout:
    """보드 배치"""
    positions: Dict[str, HexPosition]  # unit_id -> position
    total_score: float
    win_rate: float
    description: str


class BoardOptimizer:
    """
    보드 포지셔닝 최적화
    
    Usage:
        optimizer = BoardOptimizer()
        layout = optimizer.optimize(player, enemy_comps)
    """
    
    # 포지션 타입
    FRONTLINE_ROWS = [3]        # 프론트라인
    MIDLINE_ROWS = [2]          # 미드라인
    BACKLINE_ROWS = [0, 1]      # 백라인
    
    def __init__(self, simulator: Optional[CombatSimulator] = None):
        self.simulator = simulator or CombatSimulator()
    
    def optimize(
        self,
        player: PlayerState,
        enemy_layouts: Optional[List[BoardLayout]] = None,
        iterations: int = 100
    ) -> BoardLayout:
        """
        포지셔닝 최적화
        
        Args:
            player: 플레이어 상태
            enemy_layouts: 적 보드 배치들 (없으면 기본 최적화)
            iterations: 최적화 반복 횟수
        """
        units = list(player.units.board.keys())
        instances = player.units.board
        
        if not units:
            return BoardLayout({}, 0, 0, "유닛 없음")
        
        # 유닛 역할 분류
        roles = self._classify_roles(instances)
        
        # 초기 배치 생성
        best_layout = self._generate_initial_layout(units, roles)
        best_score = self._evaluate_layout(best_layout, player, enemy_layouts)
        
        # 최적화 (시뮬레이티드 어닐링 또는 그리디)
        for _ in range(iterations):
            # 무작위 스왑
            new_layout = self._mutate_layout(best_layout, units)
            new_score = self._evaluate_layout(new_layout, player, enemy_layouts)
            
            if new_score > best_score:
                best_layout = new_layout
                best_score = new_score
        
        # 승률 계산
        win_rate = self._calculate_win_rate(
            best_layout, player, enemy_layouts
        )
        
        return BoardLayout(
            positions=best_layout,
            total_score=best_score,
            win_rate=win_rate,
            description=self._describe_layout(best_layout, roles)
        )
    
    def suggest_position(
        self,
        player: PlayerState,
        unit_id: str
    ) -> List[PositionScore]:
        """특정 유닛의 최적 위치 추천"""
        instance = player.units.board.get(unit_id)
        if not instance:
            return []
        
        role = self._get_unit_role(instance)
        suggestions = []
        
        # 역할별 권장 위치
        if role == "tank":
            target_rows = self.FRONTLINE_ROWS
        elif role == "assassin":
            target_rows = self.BACKLINE_ROWS  # 상대 백라인 도달용
        elif role == "carry":
            target_rows = self.BACKLINE_ROWS
        else:
            target_rows = self.MIDLINE_ROWS
        
        # 각 위치 점수 계산
        for row in target_rows:
            for col in range(7):
                pos = HexPosition(row, col)
                score, reasons = self._score_position(
                    pos, instance, player, role
                )
                suggestions.append(PositionScore(
                    position=pos,
                    unit_id=unit_id,
                    score=score,
                    reasons=reasons
                ))
        
        suggestions.sort(key=lambda s: s.score, reverse=True)
        return suggestions[:5]
    
    def counter_position(
        self,
        player: PlayerState,
        enemy_layout: BoardLayout
    ) -> BoardLayout:
        """상대 배치에 대한 카운터 포지셔닝"""
        # 상대 캐리 위치 파악
        # 우리 어쌔신/다이버를 해당 방향으로
        # 탱커를 상대 딜러 방향으로
        
        return self.optimize(player, [enemy_layout], iterations=50)
    
    def _classify_roles(
        self,
        instances: Dict[str, 'ChampionInstance']
    ) -> Dict[str, str]:
        """유닛 역할 분류"""
        roles = {}
        for unit_id, inst in instances.items():
            roles[unit_id] = self._get_unit_role(inst)
        return roles
    
    def _get_unit_role(self, instance: 'ChampionInstance') -> str:
        """유닛 역할 판단"""
        champion = instance.champion
        
        # 특성으로 판단
        traits = [t.lower() for t in champion.traits]
        
        if "tank" in traits or "guardian" in traits or "warden" in traits:
            return "tank"
        if "assassin" in traits:
            return "assassin"
        if champion.stats.attack_range >= 3:
            return "carry"
        if "support" in traits or "enchanter" in traits:
            return "support"
        
        # 스탯으로 판단
        if champion.stats.armor > 50 or champion.stats.hp > 900:
            return "tank"
        if champion.stats.attack_damage > 60:
            return "carry"
        
        return "flex"
    
    def _generate_initial_layout(
        self,
        units: List[str],
        roles: Dict[str, str]
    ) -> Dict[str, HexPosition]:
        """초기 배치 생성"""
        layout = {}
        
        # 역할별로 분류
        tanks = [u for u in units if roles.get(u) == "tank"]
        carries = [u for u in units if roles.get(u) == "carry"]
        assassins = [u for u in units if roles.get(u) == "assassin"]
        others = [u for u in units if roles.get(u) not in ["tank", "carry", "assassin"]]
        
        used_positions = set()
        
        # 탱커 -> 프론트
        for i, unit in enumerate(tanks):
            pos = HexPosition(3, i + 2)  # 중앙 프론트
            if pos not in used_positions:
                layout[unit] = pos
                used_positions.add(pos)
        
        # 캐리 -> 백라인
        for i, unit in enumerate(carries):
            pos = HexPosition(0, i + 2)  # 중앙 백라인
            if pos not in used_positions:
                layout[unit] = pos
                used_positions.add(pos)
        
        # 어쌔신 -> 코너
        corners = [HexPosition(0, 0), HexPosition(0, 6)]
        for i, unit in enumerate(assassins):
            if i < len(corners):
                pos = corners[i]
                if pos not in used_positions:
                    layout[unit] = pos
                    used_positions.add(pos)
        
        # 나머지 -> 빈 자리
        empty = []
        for row in range(4):
            for col in range(7):
                pos = HexPosition(row, col)
                if pos not in used_positions:
                    empty.append(pos)
        
        for unit in others + [u for u in units if u not in layout]:
            if empty:
                layout[unit] = empty.pop(0)
        
        return layout
    
    def _mutate_layout(
        self,
        layout: Dict[str, HexPosition],
        units: List[str]
    ) -> Dict[str, HexPosition]:
        """배치 변형 (두 유닛 스왑)"""
        new_layout = layout.copy()
        
        if len(units) < 2:
            return new_layout
        
        u1, u2 = random.sample(units, 2)
        if u1 in new_layout and u2 in new_layout:
            new_layout[u1], new_layout[u2] = new_layout[u2], new_layout[u1]
        
        return new_layout
    
    def _evaluate_layout(
        self,
        layout: Dict[str, HexPosition],
        player: PlayerState,
        enemy_layouts: Optional[List[BoardLayout]]
    ) -> float:
        """배치 점수 평가"""
        score = 0.0
        
        roles = self._classify_roles(player.units.board)
        
        for unit_id, pos in layout.items():
            role = roles.get(unit_id, "flex")
            pos_score, _ = self._score_position(
                pos, player.units.board.get(unit_id), player, role
            )
            score += pos_score
        
        return score
    
    def _score_position(
        self,
        position: HexPosition,
        instance: Optional['ChampionInstance'],
        player: PlayerState,
        role: str
    ) -> Tuple[float, List[str]]:
        """위치 점수"""
        if instance is None:
            return 0.0, []
        
        score = 0.0
        reasons = []
        
        # 역할별 적합도
        if role == "tank" and position.row == 3:
            score += 20
            reasons.append("프론트라인 탱커")
        elif role == "carry" and position.row in [0, 1]:
            score += 20
            reasons.append("백라인 캐리")
        elif role == "assassin" and position.col in [0, 6]:
            score += 15
            reasons.append("코너 어쌔신")
        
        # 중앙 보호 (캐리)
        if role == "carry" and 2 <= position.col <= 4:
            score += 10
            reasons.append("중앙 보호")
        
        return score, reasons
    
    def _calculate_win_rate(
        self,
        layout: Dict[str, HexPosition],
        player: PlayerState,
        enemy_layouts: Optional[List[BoardLayout]]
    ) -> float:
        """승률 계산 (시뮬레이션)"""
        if enemy_layouts is None or not enemy_layouts:
            return 0.5
        
        # 실제로는 CombatSimulator로 시뮬레이션
        # 여기서는 간단히 점수 기반 추정
        return 0.5
    
    def _describe_layout(
        self,
        layout: Dict[str, HexPosition],
        roles: Dict[str, str]
    ) -> str:
        """배치 설명"""
        front = sum(1 for u, p in layout.items() if p.row == 3)
        back = sum(1 for u, p in layout.items() if p.row in [0, 1])
        
        return f"프론트 {front}명, 백라인 {back}명"
```

---

## 6. `src/optimizer/__init__.py`

```python
"""Optimizer module - Decision making recommendations"""

from .pick_advisor import PickAdvisor, PickAdvice, PickRecommendation, PickReason
from .rolldown_planner import RolldownPlanner, RolldownPlan, RolldownStrategy, RolldownTiming
from .comp_builder import CompBuilder, CompTemplate, CompRecommendation, CompStyle
from .pivot_analyzer import PivotAnalyzer, PivotAdvice, PivotOption, PivotReason
from .board_optimizer import BoardOptimizer, BoardLayout, PositionScore

__all__ = [
    # Pick Advisor
    'PickAdvisor',
    'PickAdvice',
    'PickRecommendation',
    'PickReason',
    
    # Rolldown Planner
    'RolldownPlanner',
    'RolldownPlan',
    'RolldownStrategy',
    'RolldownTiming',
    
    # Comp Builder
    'CompBuilder',
    'CompTemplate',
    'CompRecommendation',
    'CompStyle',
    
    # Pivot Analyzer
    'PivotAnalyzer',
    'PivotAdvice',
    'PivotOption',
    'PivotReason',
    
    # Board Optimizer
    'BoardOptimizer',
    'BoardLayout',
    'PositionScore',
]
```

---

## 테스트 파일

### `tests/test_pick_advisor.py`

```python
"""PickAdvisor 테스트"""

import pytest
from src.optimizer.pick_advisor import PickAdvisor, PickReason

class TestPickAdvisor:
    def test_recommend_upgrade_2star(self):
        """2성 업그레이드 추천"""
        pass
    
    def test_recommend_synergy_activation(self):
        """시너지 활성화 추천"""
        pass
    
    def test_no_recommend_bad_shop(self):
        """나쁜 상점은 추천 없음"""
        pass
    
    def test_should_refresh(self):
        """리롤 추천"""
        pass
```

### `tests/test_rolldown_planner.py`

```python
"""RolldownPlanner 테스트"""

import pytest
from src.optimizer.rolldown_planner import RolldownPlanner, RolldownStrategy

class TestRolldownPlanner:
    def test_fast_8_strategy(self):
        """패스트 8 전략"""
        pass
    
    def test_slow_roll_strategy(self):
        """슬로우롤 전략"""
        pass
    
    def test_all_in_low_hp(self):
        """저 HP 올인"""
        pass
    
    def test_budget_allocation(self):
        """예산 분배"""
        pass
```

### `tests/test_comp_builder.py`, `tests/test_pivot_analyzer.py`, `tests/test_board_optimizer.py`

각각 해당 모듈의 핵심 기능 테스트

---

## 체크리스트

- [ ] `src/optimizer/` 디렉토리 생성
- [ ] `pick_advisor.py` 구현
- [ ] `rolldown_planner.py` 구현
- [ ] `comp_builder.py` 구현
- [ ] `pivot_analyzer.py` 구현
- [ ] `board_optimizer.py` 구현
- [ ] `__init__.py` exports
- [ ] 테스트 작성 및 통과

## 예상 테스트 수
- pick_advisor: ~15 tests
- rolldown_planner: ~12 tests
- comp_builder: ~10 tests
- pivot_analyzer: ~10 tests
- board_optimizer: ~10 tests
- **총: ~57 tests (누적 ~424 tests)**

## 다음 Phase 예고

**Phase 7: API Layer**
- FastAPI REST endpoints
- WebSocket for real-time
- Authentication & rate limiting
