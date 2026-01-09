# Phase 7: API Layer

## 목표
FastAPI 기반 REST API와 WebSocket을 구현하여 TFT 시뮬레이터를 외부에서 사용할 수 있게 합니다.

## 기술 스택
- **FastAPI**: REST API 프레임워크
- **Pydantic**: 요청/응답 스키마
- **WebSocket**: 실시간 시뮬레이션
- **uvicorn**: ASGI 서버

## 디렉토리 구조

```
src/api/
├── __init__.py
├── main.py              # FastAPI 앱 진입점
├── config.py            # API 설정
├── dependencies.py      # 의존성 주입
│
├── routes/
│   ├── __init__.py
│   ├── game.py          # 게임 상태 관리
│   ├── shop.py          # 상점 API
│   ├── combat.py        # 전투 시뮬레이션
│   ├── optimizer.py     # 최적화 추천
│   └── data.py          # 정적 데이터 (챔피언, 아이템 등)
│
├── schemas/
│   ├── __init__.py
│   ├── game.py          # 게임 관련 스키마
│   ├── combat.py        # 전투 관련 스키마
│   ├── optimizer.py     # 최적화 관련 스키마
│   └── common.py        # 공통 스키마
│
├── services/
│   ├── __init__.py
│   ├── game_service.py  # 게임 로직 서비스
│   ├── combat_service.py # 전투 서비스
│   └── optimizer_service.py # 최적화 서비스
│
└── websocket/
    ├── __init__.py
    └── handlers.py      # WebSocket 핸들러
```

---

## 1. `src/api/main.py`

```python
"""
FastAPI 메인 애플리케이션
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import game, shop, combat, optimizer, data

app = FastAPI(
    title="TFT Simulator API",
    description="Teamfight Tactics Set 16 시뮬레이터 및 최적화 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(game.router, prefix="/api/game", tags=["Game"])
app.include_router(shop.router, prefix="/api/shop", tags=["Shop"])
app.include_router(combat.router, prefix="/api/combat", tags=["Combat"])
app.include_router(optimizer.router, prefix="/api/optimizer", tags=["Optimizer"])
app.include_router(data.router, prefix="/api/data", tags=["Data"])


@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "status": "ok",
        "name": "TFT Simulator API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}
```

---

## 2. `src/api/config.py`

```python
"""
API 설정
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """API 설정"""
    
    # 서버
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # 시뮬레이션
    DEFAULT_SIMULATION_COUNT: int = 100
    MAX_SIMULATION_COUNT: int = 1000
    
    # 세션
    SESSION_TIMEOUT: int = 3600  # 1시간
    MAX_SESSIONS: int = 100
    
    class Config:
        env_file = ".env"


settings = Settings()
```

---

## 3. `src/api/schemas/common.py`

```python
"""
공통 스키마
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class BaseResponse(BaseModel):
    """기본 응답"""
    status: ResponseStatus = ResponseStatus.SUCCESS
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """에러 응답"""
    status: ResponseStatus = ResponseStatus.ERROR
    error: str
    detail: Optional[str] = None


class PaginatedResponse(BaseModel):
    """페이지네이션 응답"""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
```

---

## 4. `src/api/schemas/game.py`

```python
"""
게임 관련 스키마
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# === 요청 스키마 ===

class CreateGameRequest(BaseModel):
    """게임 생성 요청"""
    player_count: int = Field(default=8, ge=2, le=8)
    

class PlayerSetupRequest(BaseModel):
    """플레이어 설정"""
    level: int = Field(default=1, ge=1, le=10)
    gold: int = Field(default=0, ge=0)
    hp: int = Field(default=100, ge=0, le=100)


class PlaceUnitRequest(BaseModel):
    """유닛 배치 요청"""
    champion_id: str
    position: Dict[str, int]  # {"row": 0, "col": 3}
    

class MoveUnitRequest(BaseModel):
    """유닛 이동 요청"""
    unit_id: str
    new_position: Dict[str, int]


class EquipItemRequest(BaseModel):
    """아이템 장착 요청"""
    unit_id: str
    item_id: str


# === 응답 스키마 ===

class ChampionInstanceSchema(BaseModel):
    """챔피언 인스턴스"""
    id: str
    champion_id: str
    name: str
    cost: int
    star_level: int
    items: List[str]
    traits: List[str]
    position: Optional[Dict[str, int]]
    
    class Config:
        from_attributes = True


class PlayerStateSchema(BaseModel):
    """플레이어 상태"""
    player_id: int
    hp: int
    gold: int
    level: int
    xp: int
    streak: int
    board: List[ChampionInstanceSchema]
    bench: List[Optional[ChampionInstanceSchema]]
    
    class Config:
        from_attributes = True


class GameStateSchema(BaseModel):
    """게임 상태"""
    game_id: str
    stage: str
    round_type: str
    players: List[PlayerStateSchema]
    
    class Config:
        from_attributes = True


class SynergySchema(BaseModel):
    """시너지 정보"""
    trait_id: str
    name: str
    count: int
    breakpoints: List[int]
    active_breakpoint: Optional[int]
    is_active: bool
    style: str  # bronze, silver, gold, chromatic


class PlayerSynergiesResponse(BaseModel):
    """플레이어 시너지 응답"""
    synergies: List[SynergySchema]
    total_active: int
```

---

## 5. `src/api/schemas/combat.py`

```python
"""
전투 관련 스키마
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class TeamSetup(BaseModel):
    """팀 설정"""
    units: List[Dict[str, Any]]  # [{champion_id, star_level, items, position}]


class SimulateCombatRequest(BaseModel):
    """전투 시뮬레이션 요청"""
    team_blue: TeamSetup
    team_red: TeamSetup
    iterations: int = Field(default=100, ge=1, le=1000)


class UnitCombatStats(BaseModel):
    """유닛 전투 통계"""
    unit_id: str
    champion_name: str
    damage_dealt: float
    damage_taken: float
    healing_done: float
    kills: int
    survived: bool


class CombatResultSchema(BaseModel):
    """전투 결과"""
    winner: str  # "blue" or "red"
    units_remaining: int
    damage_to_loser: int
    rounds_taken: int
    unit_stats: List[UnitCombatStats]


class SimulationResultSchema(BaseModel):
    """시뮬레이션 결과"""
    iterations: int
    blue_wins: int
    red_wins: int
    blue_win_rate: float
    average_rounds: float
    average_damage: float
    
    # 상세 통계
    blue_unit_stats: List[Dict[str, float]]
    red_unit_stats: List[Dict[str, float]]


class QuickCombatRequest(BaseModel):
    """빠른 전투 요청 (현재 게임 상태 기반)"""
    player_id: int
    opponent_id: int
    iterations: int = Field(default=50, ge=1, le=500)
```

---

## 6. `src/api/schemas/optimizer.py`

```python
"""
최적화 관련 스키마
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum


# === Pick Advisor ===

class PickReasonEnum(str, Enum):
    UPGRADE_2STAR = "upgrade_2star"
    UPGRADE_3STAR = "upgrade_3star"
    SYNERGY_ACTIVATE = "synergy_activate"
    SYNERGY_UPGRADE = "synergy_upgrade"
    CORE_CARRY = "core_carry"
    STRONG_UNIT = "strong_unit"
    ECONOMY_PAIR = "economy_pair"


class PickRecommendationSchema(BaseModel):
    """구매 추천"""
    champion_id: str
    champion_name: str
    shop_index: int
    score: float
    reasons: List[PickReasonEnum]
    cost: int
    copies_owned: int
    copies_needed: int


class PickAdviceRequest(BaseModel):
    """구매 조언 요청"""
    player_id: int
    target_comp: Optional[List[str]] = None


class PickAdviceResponse(BaseModel):
    """구매 조언 응답"""
    recommendations: List[PickRecommendationSchema]
    should_refresh: bool
    refresh_reason: Optional[str]
    gold_to_save: int


# === Rolldown Planner ===

class RolldownStrategyEnum(str, Enum):
    FAST_8 = "fast_8"
    FAST_9 = "fast_9"
    SLOW_ROLL_6 = "slow_roll_6"
    SLOW_ROLL_7 = "slow_roll_7"
    SLOW_ROLL_8 = "slow_roll_8"
    ALL_IN = "all_in"
    SAVE = "save"


class RolldownPlanRequest(BaseModel):
    """롤다운 계획 요청"""
    player_id: int
    target_units: List[str]
    target_stars: Optional[Dict[str, int]] = None


class RolldownPlanResponse(BaseModel):
    """롤다운 계획 응답"""
    strategy: RolldownStrategyEnum
    current_phase: str
    is_rolldown_now: bool
    roll_budget: int
    level_budget: int
    save_amount: int
    hit_probability: float
    expected_rolls: int
    advice: List[str]


# === Comp Builder ===

class CompStyleEnum(str, Enum):
    REROLL = "reroll"
    STANDARD = "standard"
    FAST_9 = "fast_9"
    FLEX = "flex"


class CompTemplateSchema(BaseModel):
    """조합 템플릿"""
    name: str
    style: CompStyleEnum
    core_units: List[str]
    carry: str
    tier: str
    difficulty: str
    description: str


class CompRecommendationSchema(BaseModel):
    """조합 추천"""
    template: CompTemplateSchema
    match_score: float
    missing_units: List[str]
    current_units: List[str]
    transition_cost: int


class CompRecommendRequest(BaseModel):
    """조합 추천 요청"""
    player_id: int
    style_filter: Optional[CompStyleEnum] = None
    top_n: int = Field(default=3, ge=1, le=10)


# === Pivot Analyzer ===

class PivotReasonEnum(str, Enum):
    CONTESTED = "contested"
    LOW_ROLLS = "low_rolls"
    HP_CRITICAL = "hp_critical"
    BETTER_ITEMS = "better_items"
    HIGHROLL = "highroll"


class PivotOptionSchema(BaseModel):
    """피벗 옵션"""
    target_comp_name: str
    shared_units: List[str]
    units_to_sell: List[str]
    units_to_buy: List[str]
    total_cost: int
    success_probability: float
    risk_level: str


class PivotAdviceRequest(BaseModel):
    """피벗 조언 요청"""
    player_id: int
    current_comp_name: Optional[str] = None
    contested_units: Optional[List[str]] = None


class PivotAdviceResponse(BaseModel):
    """피벗 조언 응답"""
    should_pivot: bool
    urgency: str
    current_comp_health: float
    options: List[PivotOptionSchema]
    explanation: str


# === Board Optimizer ===

class PositionSchema(BaseModel):
    """위치"""
    row: int
    col: int


class PositionRecommendationSchema(BaseModel):
    """위치 추천"""
    unit_id: str
    position: PositionSchema
    score: float
    reasons: List[str]


class BoardLayoutSchema(BaseModel):
    """보드 배치"""
    positions: Dict[str, PositionSchema]
    total_score: float
    win_rate: float
    description: str


class OptimizeBoardRequest(BaseModel):
    """보드 최적화 요청"""
    player_id: int
    iterations: int = Field(default=100, ge=10, le=500)


class OptimizeBoardResponse(BaseModel):
    """보드 최적화 응답"""
    layout: BoardLayoutSchema
    unit_recommendations: List[PositionRecommendationSchema]
```

---

## 7. `src/api/routes/game.py`

```python
"""
게임 상태 관리 API
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List

from ..schemas.game import (
    CreateGameRequest, PlayerSetupRequest, PlaceUnitRequest,
    MoveUnitRequest, EquipItemRequest,
    GameStateSchema, PlayerStateSchema, PlayerSynergiesResponse
)
from ..schemas.common import BaseResponse
from ..services.game_service import GameService
from ..dependencies import get_game_service

router = APIRouter()


@router.post("/create", response_model=GameStateSchema)
async def create_game(
    request: CreateGameRequest,
    service: GameService = Depends(get_game_service)
):
    """새 게임 생성"""
    game = service.create_game(request.player_count)
    return game


@router.get("/{game_id}", response_model=GameStateSchema)
async def get_game(
    game_id: str,
    service: GameService = Depends(get_game_service)
):
    """게임 상태 조회"""
    game = service.get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


@router.get("/{game_id}/player/{player_id}", response_model=PlayerStateSchema)
async def get_player(
    game_id: str,
    player_id: int,
    service: GameService = Depends(get_game_service)
):
    """플레이어 상태 조회"""
    player = service.get_player(game_id, player_id)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return player


@router.post("/{game_id}/player/{player_id}/setup", response_model=PlayerStateSchema)
async def setup_player(
    game_id: str,
    player_id: int,
    request: PlayerSetupRequest,
    service: GameService = Depends(get_game_service)
):
    """플레이어 설정"""
    player = service.setup_player(game_id, player_id, request)
    return player


@router.post("/{game_id}/player/{player_id}/place", response_model=BaseResponse)
async def place_unit(
    game_id: str,
    player_id: int,
    request: PlaceUnitRequest,
    service: GameService = Depends(get_game_service)
):
    """유닛 배치"""
    success = service.place_unit(game_id, player_id, request)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to place unit")
    return BaseResponse(message="Unit placed successfully")


@router.post("/{game_id}/player/{player_id}/move", response_model=BaseResponse)
async def move_unit(
    game_id: str,
    player_id: int,
    request: MoveUnitRequest,
    service: GameService = Depends(get_game_service)
):
    """유닛 이동"""
    success = service.move_unit(game_id, player_id, request)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to move unit")
    return BaseResponse(message="Unit moved successfully")


@router.post("/{game_id}/player/{player_id}/equip", response_model=BaseResponse)
async def equip_item(
    game_id: str,
    player_id: int,
    request: EquipItemRequest,
    service: GameService = Depends(get_game_service)
):
    """아이템 장착"""
    success = service.equip_item(game_id, player_id, request)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to equip item")
    return BaseResponse(message="Item equipped successfully")


@router.get("/{game_id}/player/{player_id}/synergies", response_model=PlayerSynergiesResponse)
async def get_synergies(
    game_id: str,
    player_id: int,
    service: GameService = Depends(get_game_service)
):
    """플레이어 시너지 조회"""
    synergies = service.get_player_synergies(game_id, player_id)
    return synergies


@router.post("/{game_id}/next-round", response_model=GameStateSchema)
async def next_round(
    game_id: str,
    service: GameService = Depends(get_game_service)
):
    """다음 라운드로 진행"""
    game = service.advance_round(game_id)
    return game


@router.delete("/{game_id}", response_model=BaseResponse)
async def delete_game(
    game_id: str,
    service: GameService = Depends(get_game_service)
):
    """게임 삭제"""
    service.delete_game(game_id)
    return BaseResponse(message="Game deleted")
```

---

## 8. `src/api/routes/shop.py`

```python
"""
상점 API
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List

from ..schemas.game import ChampionInstanceSchema
from ..schemas.common import BaseResponse
from ..services.game_service import GameService
from ..dependencies import get_game_service

router = APIRouter()


class ShopSlotSchema(BaseModel):
    """상점 슬롯"""
    index: int
    champion_id: Optional[str]
    champion_name: Optional[str]
    cost: Optional[int]
    is_purchased: bool


class ShopStateSchema(BaseModel):
    """상점 상태"""
    slots: List[ShopSlotSchema]
    is_locked: bool
    refresh_cost: int


@router.get("/{game_id}/player/{player_id}", response_model=ShopStateSchema)
async def get_shop(
    game_id: str,
    player_id: int,
    service: GameService = Depends(get_game_service)
):
    """상점 상태 조회"""
    shop = service.get_shop(game_id, player_id)
    return shop


@router.post("/{game_id}/player/{player_id}/refresh", response_model=ShopStateSchema)
async def refresh_shop(
    game_id: str,
    player_id: int,
    service: GameService = Depends(get_game_service)
):
    """상점 새로고침 (2골드)"""
    shop = service.refresh_shop(game_id, player_id)
    return shop


@router.post("/{game_id}/player/{player_id}/buy/{slot_index}", response_model=BaseResponse)
async def buy_champion(
    game_id: str,
    player_id: int,
    slot_index: int,
    service: GameService = Depends(get_game_service)
):
    """챔피언 구매"""
    success = service.buy_champion(game_id, player_id, slot_index)
    if not success:
        raise HTTPException(status_code=400, detail="Purchase failed")
    return BaseResponse(message="Champion purchased")


@router.post("/{game_id}/player/{player_id}/sell/{unit_id}", response_model=BaseResponse)
async def sell_unit(
    game_id: str,
    player_id: int,
    unit_id: str,
    service: GameService = Depends(get_game_service)
):
    """유닛 판매"""
    success = service.sell_unit(game_id, player_id, unit_id)
    if not success:
        raise HTTPException(status_code=400, detail="Sell failed")
    return BaseResponse(message="Unit sold")


@router.post("/{game_id}/player/{player_id}/lock", response_model=BaseResponse)
async def toggle_lock(
    game_id: str,
    player_id: int,
    service: GameService = Depends(get_game_service)
):
    """상점 잠금 토글"""
    is_locked = service.toggle_shop_lock(game_id, player_id)
    return BaseResponse(message=f"Shop {'locked' if is_locked else 'unlocked'}")


@router.post("/{game_id}/player/{player_id}/levelup", response_model=BaseResponse)
async def buy_xp(
    game_id: str,
    player_id: int,
    service: GameService = Depends(get_game_service)
):
    """경험치 구매 (4골드 = 4XP)"""
    success = service.buy_xp(game_id, player_id)
    if not success:
        raise HTTPException(status_code=400, detail="Not enough gold")
    return BaseResponse(message="XP purchased")
```

---

## 9. `src/api/routes/combat.py`

```python
"""
전투 시뮬레이션 API
"""

from fastapi import APIRouter, HTTPException, Depends

from ..schemas.combat import (
    SimulateCombatRequest, SimulationResultSchema,
    QuickCombatRequest, CombatResultSchema
)
from ..services.combat_service import CombatService
from ..dependencies import get_combat_service

router = APIRouter()


@router.post("/simulate", response_model=SimulationResultSchema)
async def simulate_combat(
    request: SimulateCombatRequest,
    service: CombatService = Depends(get_combat_service)
):
    """
    전투 시뮬레이션 (커스텀 팀)
    
    두 팀의 유닛 구성을 받아 N회 시뮬레이션 후 결과 반환
    """
    result = service.simulate(
        team_blue=request.team_blue,
        team_red=request.team_red,
        iterations=request.iterations
    )
    return result


@router.post("/quick/{game_id}", response_model=SimulationResultSchema)
async def quick_combat(
    game_id: str,
    request: QuickCombatRequest,
    service: CombatService = Depends(get_combat_service)
):
    """
    빠른 전투 시뮬레이션 (게임 내 플레이어들)
    
    현재 게임 상태의 두 플레이어 간 전투 시뮬레이션
    """
    result = service.simulate_players(
        game_id=game_id,
        player_id=request.player_id,
        opponent_id=request.opponent_id,
        iterations=request.iterations
    )
    return result


@router.post("/single", response_model=CombatResultSchema)
async def single_combat(
    request: SimulateCombatRequest,
    service: CombatService = Depends(get_combat_service)
):
    """단일 전투 실행 (1회)"""
    result = service.simulate(
        team_blue=request.team_blue,
        team_red=request.team_red,
        iterations=1
    )
    # 단일 결과 반환
    return service.get_last_combat_result()


@router.get("/stats/{game_id}/player/{player_id}")
async def get_combat_stats(
    game_id: str,
    player_id: int,
    service: CombatService = Depends(get_combat_service)
):
    """플레이어 전투 통계"""
    stats = service.get_player_combat_stats(game_id, player_id)
    return stats
```

---

## 10. `src/api/routes/optimizer.py`

```python
"""
최적화 추천 API
"""

from fastapi import APIRouter, HTTPException, Depends

from ..schemas.optimizer import (
    PickAdviceRequest, PickAdviceResponse,
    RolldownPlanRequest, RolldownPlanResponse,
    CompRecommendRequest, CompRecommendationSchema,
    PivotAdviceRequest, PivotAdviceResponse,
    OptimizeBoardRequest, OptimizeBoardResponse
)
from ..services.optimizer_service import OptimizerService
from ..dependencies import get_optimizer_service

router = APIRouter()


# === Pick Advisor ===

@router.post("/pick/{game_id}", response_model=PickAdviceResponse)
async def get_pick_advice(
    game_id: str,
    request: PickAdviceRequest,
    service: OptimizerService = Depends(get_optimizer_service)
):
    """상점 구매 추천"""
    advice = service.get_pick_advice(
        game_id=game_id,
        player_id=request.player_id,
        target_comp=request.target_comp
    )
    return advice


# === Rolldown Planner ===

@router.post("/rolldown/{game_id}", response_model=RolldownPlanResponse)
async def get_rolldown_plan(
    game_id: str,
    request: RolldownPlanRequest,
    service: OptimizerService = Depends(get_optimizer_service)
):
    """롤다운 계획 추천"""
    plan = service.get_rolldown_plan(
        game_id=game_id,
        player_id=request.player_id,
        target_units=request.target_units,
        target_stars=request.target_stars
    )
    return plan


# === Comp Builder ===

@router.post("/comp/{game_id}", response_model=list[CompRecommendationSchema])
async def get_comp_recommendations(
    game_id: str,
    request: CompRecommendRequest,
    service: OptimizerService = Depends(get_optimizer_service)
):
    """조합 추천"""
    recommendations = service.get_comp_recommendations(
        game_id=game_id,
        player_id=request.player_id,
        style_filter=request.style_filter,
        top_n=request.top_n
    )
    return recommendations


@router.get("/comp/templates", response_model=list[CompTemplateSchema])
async def get_comp_templates(
    service: OptimizerService = Depends(get_optimizer_service)
):
    """모든 조합 템플릿 조회"""
    return service.get_all_templates()


# === Pivot Analyzer ===

@router.post("/pivot/{game_id}", response_model=PivotAdviceResponse)
async def get_pivot_advice(
    game_id: str,
    request: PivotAdviceRequest,
    service: OptimizerService = Depends(get_optimizer_service)
):
    """피벗 조언"""
    advice = service.get_pivot_advice(
        game_id=game_id,
        player_id=request.player_id,
        current_comp_name=request.current_comp_name,
        contested_units=request.contested_units
    )
    return advice


# === Board Optimizer ===

@router.post("/board/{game_id}", response_model=OptimizeBoardResponse)
async def optimize_board(
    game_id: str,
    request: OptimizeBoardRequest,
    service: OptimizerService = Depends(get_optimizer_service)
):
    """보드 포지셔닝 최적화"""
    result = service.optimize_board(
        game_id=game_id,
        player_id=request.player_id,
        iterations=request.iterations
    )
    return result


@router.get("/board/{game_id}/player/{player_id}/suggest/{unit_id}")
async def suggest_position(
    game_id: str,
    player_id: int,
    unit_id: str,
    service: OptimizerService = Depends(get_optimizer_service)
):
    """특정 유닛 위치 추천"""
    suggestions = service.suggest_unit_position(
        game_id=game_id,
        player_id=player_id,
        unit_id=unit_id
    )
    return suggestions
```

---

## 11. `src/api/routes/data.py`

```python
"""
정적 데이터 API
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional

from src.data.loaders.champion_loader import ChampionLoader
from src.data.loaders.trait_loader import TraitLoader
from src.data.loaders.item_loader import ItemLoader

router = APIRouter()

# 로더 초기화
champion_loader = ChampionLoader()
trait_loader = TraitLoader()
item_loader = ItemLoader()


# === Champions ===

@router.get("/champions")
async def get_all_champions(cost: Optional[int] = None):
    """모든 챔피언 조회"""
    champions = champion_loader.load_all()
    if cost is not None:
        champions = [c for c in champions if c.cost == cost]
    return [c.model_dump() for c in champions]


@router.get("/champions/{champion_id}")
async def get_champion(champion_id: str):
    """특정 챔피언 조회"""
    champion = champion_loader.load_by_id(champion_id)
    if not champion:
        raise HTTPException(status_code=404, detail="Champion not found")
    return champion.model_dump()


@router.get("/champions/by-trait/{trait_id}")
async def get_champions_by_trait(trait_id: str):
    """특성별 챔피언 조회"""
    champions = champion_loader.load_by_trait(trait_id)
    return [c.model_dump() for c in champions]


# === Traits ===

@router.get("/traits")
async def get_all_traits():
    """모든 특성 조회"""
    traits = trait_loader.load_all()
    return [t.model_dump() for t in traits]


@router.get("/traits/{trait_id}")
async def get_trait(trait_id: str):
    """특정 특성 조회"""
    trait = trait_loader.load_by_id(trait_id)
    if not trait:
        raise HTTPException(status_code=404, detail="Trait not found")
    return trait.model_dump()


@router.get("/traits/origins")
async def get_origins():
    """오리진 특성 조회"""
    return [t.model_dump() for t in trait_loader.load_origins()]


@router.get("/traits/classes")
async def get_classes():
    """클래스 특성 조회"""
    return [t.model_dump() for t in trait_loader.load_classes()]


# === Items ===

@router.get("/items")
async def get_all_items():
    """모든 아이템 조회"""
    items = item_loader.load_all()
    return [i.model_dump() for i in items]


@router.get("/items/{item_id}")
async def get_item(item_id: str):
    """특정 아이템 조회"""
    item = item_loader.load_by_id(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item.model_dump()


@router.get("/items/components")
async def get_components():
    """기본 아이템 조회"""
    return [i.model_dump() for i in item_loader.load_components()]


@router.get("/items/combined")
async def get_combined_items():
    """조합 아이템 조회"""
    return [i.model_dump() for i in item_loader.load_combined()]


@router.get("/items/recipe/{item_id}")
async def get_recipe(item_id: str):
    """아이템 레시피 조회"""
    recipe = item_loader.get_recipe(item_id)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return recipe


# === Constants ===

@router.get("/constants/shop-odds")
async def get_shop_odds():
    """레벨별 상점 확률"""
    from src.core.constants import SHOP_ODDS
    return SHOP_ODDS


@router.get("/constants/pool-sizes")
async def get_pool_sizes():
    """챔피언 풀 크기"""
    from src.core.constants import POOL_SIZES
    return POOL_SIZES


@router.get("/constants/level-costs")
async def get_level_costs():
    """레벨업 비용"""
    from src.core.constants import LEVEL_COSTS
    return LEVEL_COSTS
```

---

## 12. `src/api/services/game_service.py`

```python
"""
게임 로직 서비스
"""

from typing import Dict, Optional
import uuid

from src.core.game_state import GameState, PlayerState
from src.core.shop import Shop
from src.core.champion_pool import ChampionPool
from src.core.synergy_calculator import SynergyCalculator

from ..schemas.game import (
    PlayerSetupRequest, PlaceUnitRequest, MoveUnitRequest, EquipItemRequest
)


class GameService:
    """게임 관리 서비스"""
    
    def __init__(self):
        self._games: Dict[str, GameState] = {}
        self._shops: Dict[str, Dict[int, Shop]] = {}  # game_id -> {player_id -> Shop}
        self._pools: Dict[str, ChampionPool] = {}
        self.synergy_calc = SynergyCalculator()
    
    def create_game(self, player_count: int) -> GameState:
        """게임 생성"""
        game_id = str(uuid.uuid4())
        
        # 챔피언 풀 생성
        pool = ChampionPool()
        self._pools[game_id] = pool
        
        # 게임 상태 생성
        game = GameState(game_id=game_id, player_count=player_count)
        self._games[game_id] = game
        
        # 각 플레이어 상점 생성
        self._shops[game_id] = {}
        for i in range(player_count):
            shop = Shop(pool=pool, player_level=1)
            self._shops[game_id][i] = shop
        
        return game
    
    def get_game(self, game_id: str) -> Optional[GameState]:
        """게임 조회"""
        return self._games.get(game_id)
    
    def get_player(self, game_id: str, player_id: int) -> Optional[PlayerState]:
        """플레이어 조회"""
        game = self.get_game(game_id)
        if game and 0 <= player_id < len(game.players):
            return game.players[player_id]
        return None
    
    def setup_player(
        self, 
        game_id: str, 
        player_id: int, 
        request: PlayerSetupRequest
    ) -> PlayerState:
        """플레이어 설정"""
        player = self.get_player(game_id, player_id)
        if not player:
            raise ValueError("Player not found")
        
        player.level = request.level
        player.gold = request.gold
        player.hp = request.hp
        
        # 상점 레벨 업데이트
        if game_id in self._shops and player_id in self._shops[game_id]:
            self._shops[game_id][player_id].player_level = request.level
        
        return player
    
    def place_unit(
        self, 
        game_id: str, 
        player_id: int, 
        request: PlaceUnitRequest
    ) -> bool:
        """유닛 배치"""
        player = self.get_player(game_id, player_id)
        if not player:
            return False
        
        position = (request.position["row"], request.position["col"])
        return player.units.place_on_board(request.champion_id, position)
    
    def move_unit(
        self, 
        game_id: str, 
        player_id: int, 
        request: MoveUnitRequest
    ) -> bool:
        """유닛 이동"""
        player = self.get_player(game_id, player_id)
        if not player:
            return False
        
        new_position = (request.new_position["row"], request.new_position["col"])
        return player.units.move_unit(request.unit_id, new_position)
    
    def equip_item(
        self, 
        game_id: str, 
        player_id: int, 
        request: EquipItemRequest
    ) -> bool:
        """아이템 장착"""
        player = self.get_player(game_id, player_id)
        if not player:
            return False
        
        return player.units.equip_item(request.unit_id, request.item_id)
    
    def get_player_synergies(self, game_id: str, player_id: int) -> dict:
        """플레이어 시너지 조회"""
        player = self.get_player(game_id, player_id)
        if not player:
            return {"synergies": [], "total_active": 0}
        
        synergies = player.units.get_active_synergies()
        return {
            "synergies": list(synergies.values()),
            "total_active": sum(1 for s in synergies.values() if s.get("is_active"))
        }
    
    def get_shop(self, game_id: str, player_id: int) -> dict:
        """상점 조회"""
        if game_id not in self._shops or player_id not in self._shops[game_id]:
            return {"slots": [], "is_locked": False, "refresh_cost": 2}
        
        shop = self._shops[game_id][player_id]
        return {
            "slots": [
                {
                    "index": i,
                    "champion_id": slot.champion.champion_id if slot and not slot.is_purchased else None,
                    "champion_name": slot.champion.name if slot and not slot.is_purchased else None,
                    "cost": slot.champion.cost if slot and not slot.is_purchased else None,
                    "is_purchased": slot.is_purchased if slot else True
                }
                for i, slot in enumerate(shop.slots)
            ],
            "is_locked": shop.is_locked,
            "refresh_cost": 2
        }
    
    def refresh_shop(self, game_id: str, player_id: int) -> dict:
        """상점 새로고침"""
        player = self.get_player(game_id, player_id)
        shop = self._shops.get(game_id, {}).get(player_id)
        
        if not player or not shop:
            raise ValueError("Game or player not found")
        
        if player.gold < 2:
            raise ValueError("Not enough gold")
        
        player.gold -= 2
        shop.refresh()
        
        return self.get_shop(game_id, player_id)
    
    def buy_champion(self, game_id: str, player_id: int, slot_index: int) -> bool:
        """챔피언 구매"""
        player = self.get_player(game_id, player_id)
        shop = self._shops.get(game_id, {}).get(player_id)
        
        if not player or not shop:
            return False
        
        return shop.purchase(slot_index, player)
    
    def sell_unit(self, game_id: str, player_id: int, unit_id: str) -> bool:
        """유닛 판매"""
        player = self.get_player(game_id, player_id)
        pool = self._pools.get(game_id)
        
        if not player or not pool:
            return False
        
        return player.units.sell_unit(unit_id, pool)
    
    def toggle_shop_lock(self, game_id: str, player_id: int) -> bool:
        """상점 잠금 토글"""
        shop = self._shops.get(game_id, {}).get(player_id)
        if not shop:
            return False
        
        shop.toggle_lock()
        return shop.is_locked
    
    def buy_xp(self, game_id: str, player_id: int) -> bool:
        """경험치 구매"""
        player = self.get_player(game_id, player_id)
        if not player or player.gold < 4:
            return False
        
        player.gold -= 4
        player.add_xp(4)
        
        # 상점 레벨 업데이트
        shop = self._shops.get(game_id, {}).get(player_id)
        if shop:
            shop.player_level = player.level
        
        return True
    
    def advance_round(self, game_id: str) -> GameState:
        """다음 라운드 진행"""
        game = self.get_game(game_id)
        if not game:
            raise ValueError("Game not found")
        
        game.stage_manager.advance_round()
        
        # 각 플레이어 골드 지급, 상점 리프레시 등
        for i, player in enumerate(game.players):
            income = game.calculate_income(player)
            player.gold += income
            
            shop = self._shops.get(game_id, {}).get(i)
            if shop and not shop.is_locked:
                shop.refresh()
        
        return game
    
    def delete_game(self, game_id: str) -> None:
        """게임 삭제"""
        self._games.pop(game_id, None)
        self._shops.pop(game_id, None)
        self._pools.pop(game_id, None)
```

---

## 13. `src/api/dependencies.py`

```python
"""
의존성 주입
"""

from functools import lru_cache

from .services.game_service import GameService
from .services.combat_service import CombatService
from .services.optimizer_service import OptimizerService


@lru_cache()
def get_game_service() -> GameService:
    """GameService 싱글톤"""
    return GameService()


@lru_cache()
def get_combat_service() -> CombatService:
    """CombatService 싱글톤"""
    return CombatService(get_game_service())


@lru_cache()
def get_optimizer_service() -> OptimizerService:
    """OptimizerService 싱글톤"""
    return OptimizerService(get_game_service())
```

---

## 14. 테스트

### `tests/api/test_game_routes.py`

```python
"""게임 API 테스트"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


class TestGameRoutes:
    def test_create_game(self):
        """게임 생성"""
        response = client.post("/api/game/create", json={"player_count": 8})
        assert response.status_code == 200
        data = response.json()
        assert "game_id" in data
    
    def test_get_game(self):
        """게임 조회"""
        # 먼저 게임 생성
        create_response = client.post("/api/game/create", json={"player_count": 8})
        game_id = create_response.json()["game_id"]
        
        response = client.get(f"/api/game/{game_id}")
        assert response.status_code == 200
    
    def test_get_nonexistent_game(self):
        """존재하지 않는 게임"""
        response = client.get("/api/game/nonexistent")
        assert response.status_code == 404


class TestShopRoutes:
    def test_get_shop(self):
        """상점 조회"""
        pass
    
    def test_refresh_shop(self):
        """상점 새로고침"""
        pass
    
    def test_buy_champion(self):
        """챔피언 구매"""
        pass


class TestCombatRoutes:
    def test_simulate_combat(self):
        """전투 시뮬레이션"""
        pass


class TestOptimizerRoutes:
    def test_pick_advice(self):
        """구매 추천"""
        pass
    
    def test_rolldown_plan(self):
        """롤다운 계획"""
        pass


class TestDataRoutes:
    def test_get_champions(self):
        """챔피언 조회"""
        response = client.get("/api/data/champions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_get_traits(self):
        """특성 조회"""
        response = client.get("/api/data/traits")
        assert response.status_code == 200
    
    def test_get_items(self):
        """아이템 조회"""
        response = client.get("/api/data/items")
        assert response.status_code == 200
```

---

## 15. 서버 실행

### `run.py`

```python
"""서버 실행 스크립트"""

import uvicorn
from src.api.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
```

### 실행 명령

```bash
# 개발 모드
python run.py

# 또는
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 체크리스트

- [ ] `src/api/` 디렉토리 구조 생성
- [ ] `main.py` - FastAPI 앱 설정
- [ ] `config.py` - 설정
- [ ] `dependencies.py` - 의존성 주입
- [ ] `schemas/` - Pydantic 스키마들
- [ ] `routes/game.py` - 게임 상태 API
- [ ] `routes/shop.py` - 상점 API
- [ ] `routes/combat.py` - 전투 시뮬레이션 API
- [ ] `routes/optimizer.py` - 최적화 추천 API
- [ ] `routes/data.py` - 정적 데이터 API
- [ ] `services/game_service.py` - 게임 서비스
- [ ] `services/combat_service.py` - 전투 서비스
- [ ] `services/optimizer_service.py` - 최적화 서비스
- [ ] 테스트 작성 및 통과
- [ ] API 문서 확인 (`/docs`)

## 예상 테스트 수
- API routes: ~40 tests
- Services: ~20 tests
- **총: ~60 tests (누적 ~525 tests)**

## API 엔드포인트 요약

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/game/create` | 게임 생성 |
| GET | `/api/game/{id}` | 게임 상태 조회 |
| POST | `/api/shop/{game_id}/player/{id}/refresh` | 상점 새로고침 |
| POST | `/api/shop/{game_id}/player/{id}/buy/{slot}` | 챔피언 구매 |
| POST | `/api/combat/simulate` | 전투 시뮬레이션 |
| POST | `/api/optimizer/pick/{game_id}` | 구매 추천 |
| POST | `/api/optimizer/rolldown/{game_id}` | 롤다운 계획 |
| POST | `/api/optimizer/board/{game_id}` | 보드 최적화 |
| GET | `/api/data/champions` | 챔피언 목록 |
| GET | `/api/data/traits` | 특성 목록 |
| GET | `/api/data/items` | 아이템 목록 |
