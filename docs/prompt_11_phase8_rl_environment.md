# Phase 8: TFT 강화학습 환경 (Gym Environment)

## 목표
TFT 시뮬레이터를 기반으로 OpenAI Gym 호환 강화학습 환경을 구현합니다.

## 기술 스택
- **Gymnasium**: OpenAI Gym 후속 (환경 인터페이스)
- **Stable-Baselines3**: RL 알고리즘 (PPO, DQN 등)
- **PyTorch**: 커스텀 네트워크
- **NumPy**: 상태/행동 공간

## 디렉토리 구조

```
src/rl/
├── __init__.py
├── env/
│   ├── __init__.py
│   ├── tft_env.py           # 메인 Gym 환경
│   ├── state_encoder.py     # 상태 인코딩
│   ├── action_space.py      # 행동 공간 정의
│   └── reward_calculator.py # 보상 계산
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py        # 기본 에이전트 인터페이스
│   ├── random_agent.py      # 랜덤 에이전트 (베이스라인)
│   ├── heuristic_agent.py   # 휴리스틱 에이전트
│   └── ppo_agent.py         # PPO 에이전트
│
├── networks/
│   ├── __init__.py
│   ├── feature_extractor.py # 상태 특성 추출
│   ├── policy_network.py    # 정책 네트워크
│   └── value_network.py     # 가치 네트워크
│
├── training/
│   ├── __init__.py
│   ├── trainer.py           # 학습 루프
│   ├── self_play.py         # 셀프 플레이
│   └── curriculum.py        # 커리큘럼 학습
│
└── utils/
    ├── __init__.py
    ├── replay_buffer.py     # 경험 리플레이
    └── logger.py            # 학습 로깅
```

---

## 1. State Space (상태 공간)

### 1.1 상태 구성 요소

```python
"""
TFT 상태 공간 설계

총 벡터 크기: ~2000+ 차원 (단순화 가능)
"""

# === 1. 내 보드 상태 ===
# 28개 헥스 × (챔피언 원핫 100 + 성급 3 + 아이템 46×3) = 28 × 241 = 6,748
# 단순화: 28 × (챔피언 ID 임베딩 32 + 성급 3 + 아이템 임베딩 16) = 28 × 51 = 1,428

# === 2. 내 벤치 상태 ===
# 9개 슬롯 × 51 = 459

# === 3. 상점 상태 ===
# 5개 슬롯 × (챔피언 임베딩 32 + 코스트 5) = 5 × 37 = 185

# === 4. 경제 상태 ===
# 골드 (정규화), HP (정규화), 레벨, XP, 연승/패 = ~10

# === 5. 시너지 상태 ===
# 44개 특성 × (카운트 + 활성 여부) = 88

# === 6. 스테이지 정보 ===
# 스테이지, 라운드, 라운드 타입 = ~10

# === 7. 다른 플레이어 (공개 정보) ===
# 7명 × (HP + 레벨 + 대략적 조합 강도) = 7 × 10 = 70

# === 총계 (단순화 버전) ===
# ~1,500 - 2,500 차원
```

### 1.2 State Encoder

```python
# src/rl/env/state_encoder.py

"""
상태 인코딩: 게임 상태 → 신경망 입력 벡터
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.core.game_state import PlayerState, GameState
from src.data.loaders.champion_loader import ChampionLoader


@dataclass
class EncoderConfig:
    """인코더 설정"""
    # 임베딩 차원
    champion_embed_dim: int = 32
    item_embed_dim: int = 16
    trait_embed_dim: int = 8
    
    # 보드/벤치
    board_size: int = 28  # 4x7 헥스
    bench_size: int = 9
    shop_size: int = 5
    
    # 게임 상수
    num_champions: int = 100
    num_items: int = 46
    num_traits: int = 44
    max_players: int = 8
    
    # 정규화
    max_gold: int = 100
    max_hp: int = 100
    max_level: int = 10


class StateEncoder:
    """
    게임 상태를 신경망 입력으로 인코딩
    
    Usage:
        encoder = StateEncoder()
        state_vector = encoder.encode(player_state, game_state)
    """
    
    def __init__(self, config: Optional[EncoderConfig] = None):
        self.config = config or EncoderConfig()
        self.champion_loader = ChampionLoader()
        
        # 챔피언 ID -> 인덱스 매핑
        self._champion_to_idx: Dict[str, int] = {}
        self._build_mappings()
        
        # 상태 차원 계산
        self.state_dim = self._calculate_state_dim()
    
    def _build_mappings(self):
        """ID -> 인덱스 매핑 구축"""
        champions = self.champion_loader.load_all()
        for idx, champ in enumerate(champions):
            self._champion_to_idx[champ.champion_id] = idx
    
    def _calculate_state_dim(self) -> int:
        """총 상태 차원 계산"""
        c = self.config
        
        # 단순화된 버전
        unit_dim = c.champion_embed_dim + 3 + c.item_embed_dim  # 임베딩 + 성급 + 아이템
        
        board_dim = c.board_size * unit_dim
        bench_dim = c.bench_size * unit_dim
        shop_dim = c.shop_size * (c.champion_embed_dim + 5)  # 임베딩 + 코스트 원핫
        economy_dim = 10  # 골드, HP, 레벨, XP, 연승/패 등
        synergy_dim = c.num_traits * 2  # 카운트 + 활성 여부
        stage_dim = 10
        other_players_dim = (c.max_players - 1) * 5  # HP, 레벨, 강도 등
        
        return board_dim + bench_dim + shop_dim + economy_dim + synergy_dim + stage_dim + other_players_dim
    
    def encode(
        self, 
        player: PlayerState, 
        game: GameState,
        player_idx: int = 0
    ) -> np.ndarray:
        """
        전체 상태 인코딩
        
        Returns:
            np.ndarray: 상태 벡터 (float32)
        """
        parts = []
        
        # 1. 보드 인코딩
        parts.append(self._encode_board(player))
        
        # 2. 벤치 인코딩
        parts.append(self._encode_bench(player))
        
        # 3. 상점 인코딩
        parts.append(self._encode_shop(player, game))
        
        # 4. 경제 상태
        parts.append(self._encode_economy(player))
        
        # 5. 시너지
        parts.append(self._encode_synergies(player))
        
        # 6. 스테이지
        parts.append(self._encode_stage(game))
        
        # 7. 다른 플레이어
        parts.append(self._encode_other_players(game, player_idx))
        
        return np.concatenate(parts).astype(np.float32)
    
    def _encode_board(self, player: PlayerState) -> np.ndarray:
        """보드 유닛 인코딩"""
        c = self.config
        unit_dim = c.champion_embed_dim + 3 + c.item_embed_dim
        board = np.zeros(c.board_size * unit_dim, dtype=np.float32)
        
        for pos, instance in player.units.board.items():
            # 위치 -> 인덱스 (row * 7 + col)
            idx = pos[0] * 7 + pos[1]
            if idx >= c.board_size:
                continue
            
            start = idx * unit_dim
            
            # 챔피언 (원핫 또는 임베딩 인덱스)
            champ_idx = self._champion_to_idx.get(instance.champion.champion_id, 0)
            board[start:start + c.champion_embed_dim] = self._get_champion_embedding(champ_idx)
            
            # 성급 (원핫)
            star = min(instance.star_level - 1, 2)
            board[start + c.champion_embed_dim + star] = 1.0
            
            # 아이템 (간단히 처리)
            item_start = start + c.champion_embed_dim + 3
            board[item_start:item_start + c.item_embed_dim] = self._encode_items(instance.items)
        
        return board
    
    def _encode_bench(self, player: PlayerState) -> np.ndarray:
        """벤치 유닛 인코딩"""
        c = self.config
        unit_dim = c.champion_embed_dim + 3 + c.item_embed_dim
        bench = np.zeros(c.bench_size * unit_dim, dtype=np.float32)
        
        for idx, instance in enumerate(player.units.bench):
            if instance is None or idx >= c.bench_size:
                continue
            
            start = idx * unit_dim
            
            champ_idx = self._champion_to_idx.get(instance.champion.champion_id, 0)
            bench[start:start + c.champion_embed_dim] = self._get_champion_embedding(champ_idx)
            
            star = min(instance.star_level - 1, 2)
            bench[start + c.champion_embed_dim + star] = 1.0
            
            item_start = start + c.champion_embed_dim + 3
            bench[item_start:item_start + c.item_embed_dim] = self._encode_items(instance.items)
        
        return bench
    
    def _encode_shop(self, player: PlayerState, game: GameState) -> np.ndarray:
        """상점 인코딩"""
        c = self.config
        slot_dim = c.champion_embed_dim + 5  # 임베딩 + 코스트 원핫 (1-5)
        shop = np.zeros(c.shop_size * slot_dim, dtype=np.float32)
        
        # 상점 정보 가져오기 (game_state에서)
        shop_state = game.get_shop_for_player(player.player_id)
        if shop_state is None:
            return shop
        
        for idx, slot in enumerate(shop_state.slots[:c.shop_size]):
            if slot is None or slot.is_purchased:
                continue
            
            start = idx * slot_dim
            
            champ_idx = self._champion_to_idx.get(slot.champion.champion_id, 0)
            shop[start:start + c.champion_embed_dim] = self._get_champion_embedding(champ_idx)
            
            # 코스트 원핫 (1-5)
            cost = min(slot.champion.cost - 1, 4)
            shop[start + c.champion_embed_dim + cost] = 1.0
        
        return shop
    
    def _encode_economy(self, player: PlayerState) -> np.ndarray:
        """경제 상태 인코딩"""
        c = self.config
        economy = np.zeros(10, dtype=np.float32)
        
        economy[0] = player.gold / c.max_gold  # 골드 (정규화)
        economy[1] = player.hp / c.max_hp  # HP (정규화)
        economy[2] = player.level / c.max_level  # 레벨 (정규화)
        economy[3] = player.xp / 100  # XP (정규화)
        economy[4] = min(player.streak, 5) / 5  # 연승 (정규화)
        economy[5] = min(-player.streak if player.streak < 0 else 0, 5) / 5  # 연패
        economy[6] = 1.0 if player.streak > 0 else 0.0  # 연승 중?
        economy[7] = len(player.units.board) / 10  # 보드 유닛 수
        economy[8] = len([b for b in player.units.bench if b]) / 9  # 벤치 유닛 수
        economy[9] = player.level  # 레벨 (정수)
        
        return economy
    
    def _encode_synergies(self, player: PlayerState) -> np.ndarray:
        """시너지 인코딩"""
        c = self.config
        synergies = np.zeros(c.num_traits * 2, dtype=np.float32)
        
        active_synergies = player.units.get_active_synergies()
        
        for trait_id, data in active_synergies.items():
            # 특성 인덱스 (간단히 해시)
            trait_idx = hash(trait_id) % c.num_traits
            
            synergies[trait_idx * 2] = data.get('count', 0) / 10  # 카운트 (정규화)
            synergies[trait_idx * 2 + 1] = 1.0 if data.get('is_active') else 0.0
        
        return synergies
    
    def _encode_stage(self, game: GameState) -> np.ndarray:
        """스테이지 정보 인코딩"""
        stage = np.zeros(10, dtype=np.float32)
        
        sm = game.stage_manager
        stage[0] = sm.stage / 7  # 스테이지 (정규화)
        stage[1] = sm.round / 7  # 라운드 (정규화)
        stage[2] = 1.0 if sm.is_pvp_round() else 0.0
        stage[3] = 1.0 if sm.is_carousel_round() else 0.0
        stage[4] = 1.0 if sm.is_augment_round() else 0.0
        stage[5] = sm.get_total_rounds() / 50  # 총 라운드 수 (정규화)
        
        return stage
    
    def _encode_other_players(self, game: GameState, my_idx: int) -> np.ndarray:
        """다른 플레이어 정보 인코딩"""
        c = self.config
        others = np.zeros((c.max_players - 1) * 5, dtype=np.float32)
        
        other_idx = 0
        for idx, player in enumerate(game.players):
            if idx == my_idx or not player.is_alive:
                continue
            if other_idx >= c.max_players - 1:
                break
            
            start = other_idx * 5
            others[start] = player.hp / c.max_hp
            others[start + 1] = player.level / c.max_level
            others[start + 2] = len(player.units.board) / 10
            others[start + 3] = player.gold / c.max_gold
            # 대략적인 보드 강도 (시너지 수 등)
            others[start + 4] = len(player.units.get_active_synergies()) / 10
            
            other_idx += 1
        
        return others
    
    def _get_champion_embedding(self, champ_idx: int) -> np.ndarray:
        """챔피언 임베딩 (학습 가능하게 하려면 nn.Embedding 사용)"""
        # 간단히 원핫의 압축 버전
        embed = np.zeros(self.config.champion_embed_dim, dtype=np.float32)
        embed[champ_idx % self.config.champion_embed_dim] = 1.0
        return embed
    
    def _encode_items(self, items: List) -> np.ndarray:
        """아이템 인코딩"""
        embed = np.zeros(self.config.item_embed_dim, dtype=np.float32)
        for item in items[:3]:
            item_idx = hash(item.item_id) % self.config.item_embed_dim
            embed[item_idx] = 1.0
        return embed
```

---

## 2. Action Space (행동 공간)

### 2.1 행동 정의

```python
# src/rl/env/action_space.py

"""
TFT 행동 공간 정의

행동 타입:
1. 구매 (buy): 상점 슬롯 0-4에서 구매
2. 판매 (sell): 벤치/보드 유닛 판매
3. 배치 (place): 벤치 -> 보드 배치
4. 이동 (move): 보드 내 유닛 이동
5. 리롤 (refresh): 상점 새로고침
6. 레벨업 (buy_xp): 경험치 구매
7. 패스 (pass): 아무것도 안함
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum
import gymnasium as gym
from gymnasium import spaces


class ActionType(IntEnum):
    """행동 타입"""
    PASS = 0
    BUY = 1          # buy_slot_idx
    SELL_BENCH = 2   # sell_bench_idx
    SELL_BOARD = 3   # sell_board_pos
    PLACE = 4        # place_bench_idx_to_board_pos
    MOVE = 5         # move_from_pos_to_pos
    REFRESH = 6      # 상점 리롤
    BUY_XP = 7       # 경험치 구매


@dataclass
class ActionConfig:
    """행동 공간 설정"""
    shop_size: int = 5
    bench_size: int = 9
    board_rows: int = 4
    board_cols: int = 7
    
    @property
    def board_size(self) -> int:
        return self.board_rows * self.board_cols


class ActionSpace:
    """
    TFT 행동 공간
    
    Discrete 또는 MultiDiscrete로 표현 가능
    """
    
    def __init__(self, config: Optional[ActionConfig] = None):
        self.config = config or ActionConfig()
        
        # 행동 인덱스 매핑 구축
        self._build_action_mapping()
        
        # Gym 행동 공간
        self.gym_space = spaces.Discrete(self.num_actions)
    
    def _build_action_mapping(self):
        """행동 인덱스 -> (타입, 파라미터) 매핑"""
        c = self.config
        self._idx_to_action: List[Tuple[ActionType, Any]] = []
        self._action_to_idx: Dict[Tuple, int] = {}
        
        idx = 0
        
        # 0: PASS
        self._idx_to_action.append((ActionType.PASS, None))
        self._action_to_idx[(ActionType.PASS, None)] = idx
        idx += 1
        
        # 1-5: BUY (상점 슬롯 0-4)
        for slot in range(c.shop_size):
            self._idx_to_action.append((ActionType.BUY, slot))
            self._action_to_idx[(ActionType.BUY, slot)] = idx
            idx += 1
        
        # 6-14: SELL_BENCH (벤치 슬롯 0-8)
        for bench_idx in range(c.bench_size):
            self._idx_to_action.append((ActionType.SELL_BENCH, bench_idx))
            self._action_to_idx[(ActionType.SELL_BENCH, bench_idx)] = idx
            idx += 1
        
        # 15-42: SELL_BOARD (보드 위치 0-27)
        for board_pos in range(c.board_size):
            self._idx_to_action.append((ActionType.SELL_BOARD, board_pos))
            self._action_to_idx[(ActionType.SELL_BOARD, board_pos)] = idx
            idx += 1
        
        # 43-294: PLACE (벤치 9 × 보드 28 = 252)
        for bench_idx in range(c.bench_size):
            for board_pos in range(c.board_size):
                self._idx_to_action.append((ActionType.PLACE, (bench_idx, board_pos)))
                self._action_to_idx[(ActionType.PLACE, (bench_idx, board_pos))] = idx
                idx += 1
        
        # 295: REFRESH
        self._idx_to_action.append((ActionType.REFRESH, None))
        self._action_to_idx[(ActionType.REFRESH, None)] = idx
        idx += 1
        
        # 296: BUY_XP
        self._idx_to_action.append((ActionType.BUY_XP, None))
        self._action_to_idx[(ActionType.BUY_XP, None)] = idx
        idx += 1
        
        self.num_actions = idx
    
    def decode_action(self, action_idx: int) -> Tuple[ActionType, Any]:
        """행동 인덱스 -> (타입, 파라미터)"""
        if 0 <= action_idx < len(self._idx_to_action):
            return self._idx_to_action[action_idx]
        return (ActionType.PASS, None)
    
    def encode_action(self, action_type: ActionType, params: Any = None) -> int:
        """(타입, 파라미터) -> 행동 인덱스"""
        key = (action_type, params)
        return self._action_to_idx.get(key, 0)
    
    def get_valid_actions(self, player: 'PlayerState', game: 'GameState') -> np.ndarray:
        """
        현재 상태에서 유효한 행동 마스크
        
        Returns:
            np.ndarray: shape (num_actions,), 1이면 유효, 0이면 무효
        """
        mask = np.zeros(self.num_actions, dtype=np.float32)
        c = self.config
        
        # PASS는 항상 유효
        mask[0] = 1.0
        
        # BUY: 골드 충분 + 벤치 공간 + 상점에 유닛 있음
        shop = game.get_shop_for_player(player.player_id)
        bench_space = sum(1 for b in player.units.bench if b is None)
        
        if shop and bench_space > 0:
            for slot_idx, slot in enumerate(shop.slots[:c.shop_size]):
                if slot and not slot.is_purchased:
                    if player.gold >= slot.champion.cost:
                        action_idx = self.encode_action(ActionType.BUY, slot_idx)
                        mask[action_idx] = 1.0
        
        # SELL_BENCH: 벤치에 유닛 있음
        for bench_idx, unit in enumerate(player.units.bench[:c.bench_size]):
            if unit is not None:
                action_idx = self.encode_action(ActionType.SELL_BENCH, bench_idx)
                mask[action_idx] = 1.0
        
        # SELL_BOARD: 보드에 유닛 있음
        for pos, unit in player.units.board.items():
            board_pos = pos[0] * c.board_cols + pos[1]
            if board_pos < c.board_size:
                action_idx = self.encode_action(ActionType.SELL_BOARD, board_pos)
                mask[action_idx] = 1.0
        
        # PLACE: 벤치 유닛 -> 빈 보드 위치
        board_positions = set(pos[0] * c.board_cols + pos[1] for pos in player.units.board.keys())
        max_units = player.level  # 레벨 = 최대 배치 유닛
        
        if len(player.units.board) < max_units:
            for bench_idx, unit in enumerate(player.units.bench[:c.bench_size]):
                if unit is not None:
                    for board_pos in range(c.board_size):
                        if board_pos not in board_positions:
                            action_idx = self.encode_action(ActionType.PLACE, (bench_idx, board_pos))
                            mask[action_idx] = 1.0
        
        # REFRESH: 골드 >= 2
        if player.gold >= 2:
            action_idx = self.encode_action(ActionType.REFRESH, None)
            mask[action_idx] = 1.0
        
        # BUY_XP: 골드 >= 4 + 레벨 < 10
        if player.gold >= 4 and player.level < 10:
            action_idx = self.encode_action(ActionType.BUY_XP, None)
            mask[action_idx] = 1.0
        
        return mask
    
    def sample_valid_action(self, mask: np.ndarray) -> int:
        """유효한 행동 중 랜덤 샘플"""
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) == 0:
            return 0  # PASS
        return np.random.choice(valid_indices)
    
    def pos_to_idx(self, row: int, col: int) -> int:
        """(row, col) -> board position index"""
        return row * self.config.board_cols + col
    
    def idx_to_pos(self, idx: int) -> Tuple[int, int]:
        """board position index -> (row, col)"""
        return (idx // self.config.board_cols, idx % self.config.board_cols)
```

---

## 3. Reward Function (보상 함수)

### 3.1 보상 설계

```python
# src/rl/env/reward_calculator.py

"""
TFT 보상 함수

보상 구성:
1. 최종 보상: 순위 기반
2. 라운드 보상: 승패, HP 변화
3. 보상 쉐이핑: 시너지, 업그레이드, 경제 등
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    """보상 설정"""
    # 최종 보상 (순위별)
    placement_rewards: Dict[int, float] = field(default_factory=lambda: {
        1: 10.0,   # 1등
        2: 6.0,    # 2등
        3: 3.0,    # 3등
        4: 1.0,    # 4등
        5: -1.0,   # 5등
        6: -3.0,   # 6등
        7: -6.0,   # 7등
        8: -10.0,  # 8등
    })
    
    # 라운드 보상
    win_reward: float = 0.5
    lose_reward: float = -0.2
    hp_loss_penalty: float = -0.05  # HP 1당
    
    # 쉐이핑 보상
    synergy_activate_reward: float = 0.3
    synergy_upgrade_reward: float = 0.2
    unit_upgrade_2star: float = 0.5
    unit_upgrade_3star: float = 2.0
    
    # 경제 보상
    interest_reward: float = 0.1  # 이자 골드당
    streak_reward: float = 0.1   # 연승/패 보너스
    
    # 패널티
    bench_full_penalty: float = -0.1
    invalid_action_penalty: float = -0.5
    
    # 할인율 (중간 보상 vs 최종 보상)
    shaping_weight: float = 0.1  # 쉐이핑 보상 가중치


class RewardCalculator:
    """
    보상 계산기
    
    Usage:
        calc = RewardCalculator()
        reward = calc.calculate(prev_state, action, new_state, done, info)
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        
        # 이전 상태 추적 (쉐이핑용)
        self._prev_synergies: Dict[str, int] = {}
        self._prev_hp: int = 100
        self._prev_gold: int = 0
        self._prev_star_levels: Dict[str, int] = {}
    
    def calculate(
        self,
        player: 'PlayerState',
        action_type: 'ActionType',
        action_valid: bool,
        round_result: Optional[Dict[str, Any]] = None,
        done: bool = False,
        placement: Optional[int] = None
    ) -> float:
        """
        보상 계산
        
        Args:
            player: 현재 플레이어 상태
            action_type: 수행한 행동 타입
            action_valid: 행동 유효 여부
            round_result: 라운드 결과 (승패, HP 변화 등)
            done: 게임 종료 여부
            placement: 최종 순위 (게임 종료 시)
            
        Returns:
            float: 총 보상
        """
        c = self.config
        reward = 0.0
        
        # 1. 무효 행동 패널티
        if not action_valid:
            reward += c.invalid_action_penalty
            return reward
        
        # 2. 최종 보상 (게임 종료 시)
        if done and placement is not None:
            reward += c.placement_rewards.get(placement, 0)
            return reward
        
        # 3. 라운드 보상
        if round_result:
            reward += self._calculate_round_reward(round_result)
        
        # 4. 쉐이핑 보상
        reward += self._calculate_shaping_reward(player) * c.shaping_weight
        
        # 이전 상태 업데이트
        self._update_prev_state(player)
        
        return reward
    
    def _calculate_round_reward(self, result: Dict[str, Any]) -> float:
        """라운드 승패 보상"""
        c = self.config
        reward = 0.0
        
        if result.get('won', False):
            reward += c.win_reward
        else:
            reward += c.lose_reward
            hp_lost = result.get('hp_lost', 0)
            reward += hp_lost * c.hp_loss_penalty
        
        return reward
    
    def _calculate_shaping_reward(self, player: 'PlayerState') -> float:
        """보상 쉐이핑"""
        c = self.config
        reward = 0.0
        
        # 시너지 변화
        current_synergies = player.units.get_active_synergies()
        for trait_id, data in current_synergies.items():
            prev_count = self._prev_synergies.get(trait_id, 0)
            curr_count = data.get('count', 0)
            
            if curr_count > prev_count:
                if data.get('is_active') and trait_id not in self._prev_synergies:
                    reward += c.synergy_activate_reward
                else:
                    reward += c.synergy_upgrade_reward
        
        # 유닛 업그레이드
        for pos, unit in player.units.board.items():
            unit_key = f"board_{pos}"
            prev_star = self._prev_star_levels.get(unit_key, 1)
            if unit.star_level > prev_star:
                if unit.star_level == 2:
                    reward += c.unit_upgrade_2star
                elif unit.star_level == 3:
                    reward += c.unit_upgrade_3star
        
        for idx, unit in enumerate(player.units.bench):
            if unit:
                unit_key = f"bench_{idx}"
                prev_star = self._prev_star_levels.get(unit_key, 1)
                if unit.star_level > prev_star:
                    if unit.star_level == 2:
                        reward += c.unit_upgrade_2star
                    elif unit.star_level == 3:
                        reward += c.unit_upgrade_3star
        
        # 이자 보상
        interest = min(player.gold // 10, 5)
        reward += interest * c.interest_reward
        
        return reward
    
    def _update_prev_state(self, player: 'PlayerState'):
        """이전 상태 저장"""
        self._prev_synergies = {
            trait_id: data.get('count', 0)
            for trait_id, data in player.units.get_active_synergies().items()
        }
        self._prev_hp = player.hp
        self._prev_gold = player.gold
        
        self._prev_star_levels = {}
        for pos, unit in player.units.board.items():
            self._prev_star_levels[f"board_{pos}"] = unit.star_level
        for idx, unit in enumerate(player.units.bench):
            if unit:
                self._prev_star_levels[f"bench_{idx}"] = unit.star_level
    
    def reset(self):
        """에피소드 리셋"""
        self._prev_synergies = {}
        self._prev_hp = 100
        self._prev_gold = 0
        self._prev_star_levels = {}
```

---

## 4. Gym Environment

### 4.1 메인 환경

```python
# src/rl/env/tft_env.py

"""
TFT Gymnasium 환경
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import gymnasium as gym
from gymnasium import spaces

from src.core.game_state import GameState, PlayerState
from src.core.champion_pool import ChampionPool
from src.core.shop import Shop

from .state_encoder import StateEncoder, EncoderConfig
from .action_space import ActionSpace, ActionType, ActionConfig
from .reward_calculator import RewardCalculator, RewardConfig


class TFTEnv(gym.Env):
    """
    TFT Gymnasium 환경
    
    Single-agent 환경 (다른 플레이어는 봇/휴리스틱)
    
    Usage:
        env = TFTEnv()
        obs, info = env.reset()
        
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}
    
    def __init__(
        self,
        num_players: int = 8,
        agent_player_idx: int = 0,
        max_rounds: int = 50,
        render_mode: Optional[str] = None,
        encoder_config: Optional[EncoderConfig] = None,
        action_config: Optional[ActionConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ):
        super().__init__()
        
        self.num_players = num_players
        self.agent_player_idx = agent_player_idx
        self.max_rounds = max_rounds
        self.render_mode = render_mode
        
        # 컴포넌트 초기화
        self.state_encoder = StateEncoder(encoder_config)
        self.action_space_handler = ActionSpace(action_config)
        self.reward_calculator = RewardCalculator(reward_config)
        
        # Gym 공간 정의
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_encoder.state_dim,),
            dtype=np.float32
        )
        self.action_space = self.action_space_handler.gym_space
        
        # 게임 상태 (reset에서 초기화)
        self.game: Optional[GameState] = None
        self.pool: Optional[ChampionPool] = None
        self.shops: Dict[int, Shop] = {}
        
        # 에피소드 추적
        self.current_round = 0
        self.done = False
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """환경 리셋"""
        super().reset(seed=seed)
        
        # 게임 초기화
        self.pool = ChampionPool()
        self.game = GameState(
            game_id="rl_game",
            player_count=self.num_players
        )
        
        # 각 플레이어 상점 초기화
        self.shops = {}
        for i in range(self.num_players):
            self.shops[i] = Shop(pool=self.pool, player_level=1)
        
        # 보상 계산기 리셋
        self.reward_calculator.reset()
        
        # 에피소드 상태
        self.current_round = 0
        self.done = False
        
        # 초기 관찰
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        행동 수행
        
        Args:
            action: 행동 인덱스
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.done:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        player = self.game.players[self.agent_player_idx]
        
        # 행동 디코딩 및 실행
        action_type, params = self.action_space_handler.decode_action(action)
        action_valid = self._execute_action(player, action_type, params)
        
        # 라운드 종료 체크 (일정 행동 후 또는 PASS)
        round_result = None
        if action_type == ActionType.PASS or self._should_end_round():
            round_result = self._simulate_round()
            self.current_round += 1
        
        # 게임 종료 체크
        terminated = self._check_game_over()
        truncated = self.current_round >= self.max_rounds
        self.done = terminated or truncated
        
        # 순위 계산 (게임 종료 시)
        placement = None
        if self.done:
            placement = self._calculate_placement()
        
        # 보상 계산
        reward = self.reward_calculator.calculate(
            player=player,
            action_type=action_type,
            action_valid=action_valid,
            round_result=round_result,
            done=self.done,
            placement=placement
        )
        
        # 관찰 및 정보
        obs = self._get_observation()
        info = self._get_info()
        info['action_valid'] = action_valid
        info['round_result'] = round_result
        info['placement'] = placement
        
        return obs, reward, terminated, truncated, info
    
    def _execute_action(
        self,
        player: PlayerState,
        action_type: ActionType,
        params: Any
    ) -> bool:
        """행동 실행"""
        c = self.action_space_handler.config
        shop = self.shops.get(player.player_id)
        
        if action_type == ActionType.PASS:
            return True
        
        elif action_type == ActionType.BUY:
            slot_idx = params
            if shop and 0 <= slot_idx < c.shop_size:
                return shop.purchase(slot_idx, player)
            return False
        
        elif action_type == ActionType.SELL_BENCH:
            bench_idx = params
            if 0 <= bench_idx < c.bench_size:
                return player.units.sell_bench(bench_idx, self.pool)
            return False
        
        elif action_type == ActionType.SELL_BOARD:
            board_pos = params
            pos = self.action_space_handler.idx_to_pos(board_pos)
            return player.units.sell_board(pos, self.pool)
        
        elif action_type == ActionType.PLACE:
            bench_idx, board_pos = params
            pos = self.action_space_handler.idx_to_pos(board_pos)
            return player.units.place_from_bench(bench_idx, pos)
        
        elif action_type == ActionType.REFRESH:
            if player.gold >= 2:
                player.gold -= 2
                shop.refresh()
                return True
            return False
        
        elif action_type == ActionType.BUY_XP:
            if player.gold >= 4 and player.level < 10:
                player.gold -= 4
                player.add_xp(4)
                shop.player_level = player.level
                return True
            return False
        
        return False
    
    def _simulate_round(self) -> Dict[str, Any]:
        """라운드 시뮬레이션 (전투)"""
        # 에이전트 vs 랜덤 상대
        agent = self.game.players[self.agent_player_idx]
        
        # 살아있는 상대 중 랜덤 선택
        alive_opponents = [
            p for i, p in enumerate(self.game.players)
            if i != self.agent_player_idx and p.is_alive
        ]
        
        if not alive_opponents:
            return {'won': True, 'hp_lost': 0}
        
        opponent = np.random.choice(alive_opponents)
        
        # 전투 시뮬레이션 (간단한 버전)
        # 실제로는 CombatSimulator 사용
        agent_power = self._estimate_board_power(agent)
        opponent_power = self._estimate_board_power(opponent)
        
        won = agent_power > opponent_power + np.random.randn() * 10
        hp_lost = 0
        
        if not won:
            # HP 손실 계산
            units_remaining = max(1, len(opponent.units.board) - len(agent.units.board) // 2)
            hp_lost = units_remaining + self.current_round // 3
            agent.hp -= hp_lost
            
            if agent.hp <= 0:
                agent.hp = 0
                agent.is_alive = False
        
        # 봇 플레이어들도 업데이트
        self._update_bot_players()
        
        # 골드 지급
        income = self.game.calculate_income(agent)
        agent.gold += income
        
        # 상점 리프레시
        shop = self.shops.get(agent.player_id)
        if shop and not shop.is_locked:
            shop.refresh()
        
        return {'won': won, 'hp_lost': hp_lost}
    
    def _estimate_board_power(self, player: PlayerState) -> float:
        """보드 파워 추정 (간단한 휴리스틱)"""
        power = 0.0
        
        for unit in player.units.board.values():
            # 기본 파워: 코스트 × 성급 보너스
            star_mult = {1: 1.0, 2: 1.8, 3: 3.0}.get(unit.star_level, 1.0)
            power += unit.champion.cost * star_mult * 10
            
            # 아이템 보너스
            power += len(unit.items) * 5
        
        # 시너지 보너스
        synergies = player.units.get_active_synergies()
        active_count = sum(1 for s in synergies.values() if s.get('is_active'))
        power += active_count * 15
        
        return power
    
    def _update_bot_players(self):
        """봇 플레이어 업데이트 (간단한 휴리스틱)"""
        for i, player in enumerate(self.game.players):
            if i == self.agent_player_idx or not player.is_alive:
                continue
            
            # 간단한 봇 로직
            shop = self.shops.get(i)
            if not shop:
                continue
            
            # 구매 가능하면 랜덤 구매
            for slot_idx, slot in enumerate(shop.slots):
                if slot and not slot.is_purchased:
                    if player.gold >= slot.champion.cost:
                        if np.random.random() < 0.7:  # 70% 확률로 구매
                            shop.purchase(slot_idx, player)
            
            # 레벨업 (확률적)
            if player.gold >= 4 and player.level < 10:
                if np.random.random() < 0.3:
                    player.gold -= 4
                    player.add_xp(4)
                    shop.player_level = player.level
            
            # 골드 지급
            income = self.game.calculate_income(player)
            player.gold += income
            
            # 상점 리프레시
            if not shop.is_locked:
                shop.refresh()
    
    def _should_end_round(self) -> bool:
        """라운드 종료 조건"""
        # 간단히: 매 행동 후 라운드 종료 체크 없이
        # PASS 시에만 라운드 종료
        return False
    
    def _check_game_over(self) -> bool:
        """게임 종료 체크"""
        agent = self.game.players[self.agent_player_idx]
        
        # 에이전트 사망
        if not agent.is_alive:
            return True
        
        # 1명만 생존
        alive_count = sum(1 for p in self.game.players if p.is_alive)
        return alive_count <= 1
    
    def _calculate_placement(self) -> int:
        """최종 순위 계산"""
        agent = self.game.players[self.agent_player_idx]
        
        # HP 기준 순위
        sorted_players = sorted(
            self.game.players,
            key=lambda p: (p.is_alive, p.hp),
            reverse=True
        )
        
        for rank, player in enumerate(sorted_players, 1):
            if player.player_id == agent.player_id:
                return rank
        
        return self.num_players
    
    def _get_observation(self) -> np.ndarray:
        """현재 관찰 반환"""
        player = self.game.players[self.agent_player_idx]
        return self.state_encoder.encode(player, self.game, self.agent_player_idx)
    
    def _get_info(self) -> Dict[str, Any]:
        """추가 정보 반환"""
        player = self.game.players[self.agent_player_idx]
        
        return {
            'round': self.current_round,
            'hp': player.hp,
            'gold': player.gold,
            'level': player.level,
            'board_size': len(player.units.board),
            'valid_action_mask': self.action_space_handler.get_valid_actions(player, self.game),
            'alive_players': sum(1 for p in self.game.players if p.is_alive),
        }
    
    def render(self) -> Optional[str]:
        """환경 렌더링"""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
        return None
    
    def _render_ansi(self) -> str:
        """텍스트 렌더링"""
        player = self.game.players[self.agent_player_idx]
        
        lines = [
            f"=== Round {self.current_round} ===",
            f"HP: {player.hp} | Gold: {player.gold} | Level: {player.level}",
            f"Board: {len(player.units.board)} units",
            f"Bench: {sum(1 for b in player.units.bench if b)} units",
            f"Alive players: {sum(1 for p in self.game.players if p.is_alive)}",
        ]
        
        return "\n".join(lines)
    
    def get_valid_action_mask(self) -> np.ndarray:
        """유효 행동 마스크 (외부 접근용)"""
        player = self.game.players[self.agent_player_idx]
        return self.action_space_handler.get_valid_actions(player, self.game)


# 환경 등록
gym.register(
    id='TFT-v0',
    entry_point='src.rl.env.tft_env:TFTEnv',
    max_episode_steps=1000,
)
```

---

## 5. Training Pipeline

### 5.1 PPO 학습

```python
# src/rl/training/trainer.py

"""
TFT 에이전트 학습
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import json
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gymnasium as gym
import torch.nn as nn

from src.rl.env.tft_env import TFTEnv


class TFTFeaturesExtractor(BaseFeaturesExtractor):
    """
    커스텀 특성 추출기
    
    TFT 상태의 구조적 특성을 활용
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class MaskablePPO:
    """
    Invalid Action Masking이 적용된 PPO
    
    유효하지 않은 행동에 큰 음수 로짓 적용
    """
    
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        device: str = "auto",
        verbose: int = 1,
    ):
        policy_kwargs = dict(
            features_extractor_class=TFTFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
        )
        
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=verbose,
        )
    
    def predict(self, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
        """마스킹 적용 예측"""
        action, _ = self.model.predict(obs, deterministic=False)
        
        if mask is not None:
            # 유효하지 않은 행동이면 유효한 것 중 랜덤 선택
            if mask[action] == 0:
                valid_actions = np.where(mask > 0)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    action = 0  # PASS
        
        return action
    
    def learn(self, total_timesteps: int, callback: Optional[BaseCallback] = None):
        """학습"""
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
    
    def save(self, path: str):
        """모델 저장"""
        self.model.save(path)
    
    def load(self, path: str):
        """모델 로드"""
        self.model = PPO.load(path)


class TrainingCallback(BaseCallback):
    """학습 콜백"""
    
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.placements = []
    
    def _on_step(self) -> bool:
        # 에피소드 종료 시 로깅
        for info in self.locals.get('infos', []):
            if 'placement' in info and info['placement'] is not None:
                self.placements.append(info['placement'])
        
        return True
    
    def _on_rollout_end(self) -> None:
        # 롤아웃 종료 시 통계
        if self.placements:
            avg_placement = np.mean(self.placements[-100:])
            self.logger.record("tft/avg_placement", avg_placement)
            
            top4_rate = sum(1 for p in self.placements[-100:] if p <= 4) / min(len(self.placements), 100)
            self.logger.record("tft/top4_rate", top4_rate)


def train_agent(
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    save_dir: str = "models/tft_agent",
    log_dir: str = "logs/tft",
):
    """
    TFT 에이전트 학습
    
    Args:
        total_timesteps: 총 학습 스텝
        n_envs: 병렬 환경 수
        learning_rate: 학습률
        save_dir: 모델 저장 경로
        log_dir: 로그 저장 경로
    """
    # 환경 생성
    def make_env():
        return TFTEnv(render_mode=None)
    
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # 에이전트 생성
    agent = MaskablePPO(
        env=env,
        learning_rate=learning_rate,
        verbose=1,
    )
    
    # 콜백
    callback = TrainingCallback(log_dir)
    
    # 학습
    print(f"Starting training for {total_timesteps} timesteps...")
    agent.learn(total_timesteps=total_timesteps, callback=callback)
    
    # 저장
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = save_path / f"tft_ppo_{timestamp}"
    agent.save(str(model_path))
    
    print(f"Model saved to {model_path}")
    
    return agent


def evaluate_agent(
    model_path: str,
    n_episodes: int = 100,
):
    """
    에이전트 평가
    
    Args:
        model_path: 모델 경로
        n_episodes: 평가 에피소드 수
    """
    env = TFTEnv(render_mode=None)
    agent = MaskablePPO(env=env)
    agent.load(model_path)
    
    placements = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            mask = info.get('valid_action_mask')
            action = agent.predict(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        placement = info.get('placement', 8)
        placements.append(placement)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}: Placement {placement}")
    
    # 결과
    avg_placement = np.mean(placements)
    top4_rate = sum(1 for p in placements if p <= 4) / len(placements)
    win_rate = sum(1 for p in placements if p == 1) / len(placements)
    
    print(f"\n=== Evaluation Results ({n_episodes} episodes) ===")
    print(f"Average Placement: {avg_placement:.2f}")
    print(f"Top 4 Rate: {top4_rate * 100:.1f}%")
    print(f"Win Rate: {win_rate * 100:.1f}%")
    
    return {
        'avg_placement': avg_placement,
        'top4_rate': top4_rate,
        'win_rate': win_rate,
        'placements': placements,
    }


if __name__ == "__main__":
    # 학습 실행
    agent = train_agent(
        total_timesteps=500_000,
        n_envs=4,
    )
    
    # 평가
    evaluate_agent("models/tft_agent/tft_ppo_latest", n_episodes=50)
```

---

## 6. 설치 및 실행

### 6.1 추가 의존성

```bash
pip install gymnasium stable-baselines3 tensorboard torch
```

### 6.2 실행 스크립트

```python
# train_tft.py

from src.rl.training.trainer import train_agent, evaluate_agent

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train agent")
    parser.add_argument("--eval", type=str, help="Evaluate model at path")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--episodes", type=int, default=100)
    
    args = parser.parse_args()
    
    if args.train:
        train_agent(total_timesteps=args.timesteps)
    
    if args.eval:
        evaluate_agent(args.eval, n_episodes=args.episodes)
```

```bash
# 학습
python train_tft.py --train --timesteps 1000000

# 평가
python train_tft.py --eval models/tft_agent/tft_ppo_latest --episodes 100
```

---

## 체크리스트

- [ ] `src/rl/env/` 디렉토리 생성
- [ ] `state_encoder.py` - 상태 인코딩
- [ ] `action_space.py` - 행동 공간 정의
- [ ] `reward_calculator.py` - 보상 계산
- [ ] `tft_env.py` - Gym 환경
- [ ] `src/rl/training/trainer.py` - 학습 파이프라인
- [ ] 테스트 작성
- [ ] 학습 실행 및 평가

## 예상 테스트 수
- env 테스트: ~30 tests
- training 테스트: ~10 tests
- **총: ~40 tests (누적 ~538 tests)**

## 학습 팁

1. **단순화된 환경부터**: 처음에는 2-4명 플레이어로 시작
2. **Curriculum Learning**: 쉬운 봇 → 어려운 봇 순서로
3. **Self-Play**: 일정 수준 도달 후 자기 자신과 대전
4. **Reward Shaping 조절**: 처음에는 강하게 → 점점 최종 보상 위주로
5. **Invalid Action Masking**: 유효한 행동만 선택하도록 강제
