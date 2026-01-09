# TFT 시뮬레이터 & 최적화 툴 프로젝트 명세서

## 📋 프로젝트 개요

### 목표
Teamfight Tactics (TFT) Set 16 "Lore & Legends" 기반의 완전한 게임 시뮬레이터와 실시간 의사결정 최적화 도구 개발

### 대상 시즌
- **Set 16: Lore & Legends** (2025년 12월 3일 출시)
- 100개 챔피언 (60개 기본 + 40개 언락 가능)
- 새로운 언락 시스템 포함

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        TFT Simulator Core                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Champion Pool│  │  Shop System │  │  Synergy Calculator  │   │
│  │   Manager    │  │              │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │    Item      │  │   Economy    │  │  Combat Simulator    │   │
│  │   System     │  │   System     │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Unlockable Champion System                   │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      Optimization Engine                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Pick Advisor │  │ Rolldown     │  │  Comp Builder        │   │
│  │              │  │ Timer        │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Pivot Analyzer                         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 모듈 1: 챔피언 풀 & 상점 시스템

### 1.1 챔피언 풀 (Champion Pool)

#### 데이터 구조
```typescript
interface Champion {
  id: string;
  name: string;
  cost: 1 | 2 | 3 | 4 | 5 | 6 | 7;  // Set 16은 7코스트까지 존재
  traits: Trait[];
  stats: ChampionStats;
  ability: Ability;
  isUnlockable: boolean;
  unlockCondition?: UnlockCondition;
}

interface ChampionStats {
  health: number[];      // [1성, 2성, 3성]
  attackDamage: number[];
  attackSpeed: number;
  armor: number;
  magicResist: number;
  range: number;
  mana: [number, number]; // [시작, 최대]
}

interface ChampionPool {
  // 코스트별 총 유닛 수 (8인 게임 기준)
  poolSize: {
    1: 30,  // 각 1코스트 챔피언당 30개
    2: 25,
    3: 18,
    4: 10,
    5: 9,
    6: 8,   // 6코스트 (언락 전용)
    7: 6    // 7코스트 (언락 전용)
  }
}
```

#### 핵심 기능
- 전체 챔피언 풀 추적
- 플레이어별 보유 챔피언 관리
- 남은 챔피언 수 계산
- 확률 계산 (특정 챔피언을 뽑을 확률)

### 1.2 상점 시스템 (Shop System)

#### 레벨별 상점 확률
```typescript
const SHOP_ODDS: Record<number, number[]> = {
  // [1코스트, 2코스트, 3코스트, 4코스트, 5코스트]
  1:  [100, 0,   0,   0,   0],
  2:  [100, 0,   0,   0,   0],
  3:  [75,  25,  0,   0,   0],
  4:  [55,  30,  15,  0,   0],
  5:  [45,  33,  20,  2,   0],
  6:  [30,  40,  25,  5,   0],
  7:  [19,  30,  40,  10,  1],
  8:  [18,  25,  32,  22,  3],
  9:  [10,  20,  25,  35,  10],
  10: [5,   10,  20,  40,  25],
};
```

#### 상점 기능
```typescript
interface ShopSystem {
  // 상점 새로고침 (5개 슬롯)
  refresh(playerState: PlayerState): Champion[];
  
  // 특정 챔피언 구매
  purchase(championId: string, playerState: PlayerState): boolean;
  
  // 판매 (챔피언 풀로 반환)
  sell(champion: Champion, playerState: PlayerState): void;
  
  // 언락된 챔피언 상점에 추가 (Set 16 특수 기능)
  addUnlockedToShop(champion: Champion): void;
}
```

### 1.3 언락 시스템 (Set 16 전용)

```typescript
interface UnlockCondition {
  type: 'trait_level' | 'champion_star' | 'gold_amount' | 'win_streak' | 'custom';
  requirement: any;
  description: string;
}

// 예시: Galio 언락 조건
const galioUnlock: UnlockCondition = {
  type: 'trait_level',
  requirement: { trait: 'Demacia', level: 12 },
  description: '데마시아 특성 12레벨 달성'
};
```

---

## 📦 모듈 2: 시너지/특성 계산기

### 2.1 특성 데이터 구조

```typescript
interface Trait {
  id: string;
  name: string;
  type: 'origin' | 'class';
  breakpoints: TraitBreakpoint[];
  description: string;
}

interface TraitBreakpoint {
  count: number;
  effect: TraitEffect;
}

interface TraitEffect {
  type: 'stat_bonus' | 'special' | 'summon';
  values: Record<string, number>;
  description: string;
}
```

### 2.2 Set 16 주요 지역(Origin) 특성

```typescript
const SET16_ORIGINS = {
  Bilgewater: {
    breakpoints: [2, 4, 6],
    mechanic: '은화 뱀 획득, 블랙마켓 이용'
  },
  Demacia: {
    breakpoints: [2, 4, 6, 8],
    mechanic: '팀이 체력을 잃으면 랠리 보너스'
  },
  Ionia: {
    breakpoints: [2, 3, 4, 5],
    mechanic: '각 게임마다 다른 경로 선택'
  },
  Noxus: {
    breakpoints: [3, 5, 7, 9],
    mechanic: '아타칸 소환'
  },
  PiltoverZaun: {
    breakpoints: [2, 4, 6],
    mechanic: '발명품 제작'
  },
  ShadowIsles: {
    breakpoints: [2, 4, 6],
    mechanic: '영혼 수집'
  },
  Shurima: {
    breakpoints: [2, 4, 6],
    mechanic: '공격속도 및 체력 회복'
  },
  Targon: {
    breakpoints: [2, 3, 4],
    mechanic: '신성한 축복'
  },
  Void: {
    breakpoints: [3, 5, 7],
    mechanic: '공허 균열'
  },
  // ... 기타 지역
};
```

### 2.3 시너지 계산 엔진

```typescript
class SynergyCalculator {
  // 현재 보드의 모든 활성 시너지 계산
  calculateActiveSynergies(board: Champion[]): ActiveSynergy[];
  
  // 챔피언 추가 시 시너지 변화 프리뷰
  previewSynergyChange(board: Champion[], newChamp: Champion): SynergyDelta;
  
  // 최적 시너지 조합 제안
  suggestOptimalSynergies(availableChamps: Champion[], slots: number): Champion[];
  
  // 엠블럼/스패츌라 효과 계산
  applyEmblem(champion: Champion, trait: Trait): void;
}
```

---

## 📦 모듈 3: 아이템 시스템

### 3.1 기본 아이템 (Components)

```typescript
const BASE_ITEMS = {
  BFSword: { ad: 10 },
  RecurveBow: { as: 10 },
  NeedlesslyLargeRod: { ap: 10 },
  TearOfTheGoddess: { mana: 15 },
  ChainVest: { armor: 20 },
  NegatronCloak: { mr: 20 },
  GiantsBelt: { hp: 150 },
  Spatula: { special: true },
  FryingPan: { special: true }  // Set 16 신규
};
```

### 3.2 조합 아이템 (Completed Items)

```typescript
interface CompletedItem {
  id: string;
  name: string;
  components: [string, string];
  stats: ItemStats;
  effect: ItemEffect;
  unique?: boolean;  // 중복 착용 불가 여부
}

// 조합 매트릭스
const ITEM_RECIPES: Record<string, Record<string, string>> = {
  BFSword: {
    BFSword: 'Deathblade',
    RecurveBow: 'GiantSlayer',
    NeedlesslyLargeRod: 'HextechGunblade',
    // ...
  },
  // ...
};
```

### 3.3 아이템 최적화 엔진

```typescript
class ItemOptimizer {
  // 챔피언에 최적 아이템 추천
  recommendItems(champion: Champion, availableItems: Item[]): Item[];
  
  // 현재 컴포넌트로 가능한 조합 아이템 목록
  getAvailableRecipes(components: Item[]): CompletedItem[];
  
  // 특정 아이템을 만들기 위해 필요한 컴포넌트
  getRequiredComponents(item: CompletedItem): Item[];
  
  // BiS (Best in Slot) 계산
  calculateBiS(champion: Champion): CompletedItem[];
}
```

---

## 📦 모듈 4: 경제 시스템

### 4.1 골드 메커니즘

```typescript
interface EconomySystem {
  // 기본 수입
  baseIncome: 5;
  
  // 이자 (10골드당 1골드, 최대 5골드)
  calculateInterest(gold: number): number;
  
  // 연승/연패 보너스
  streakBonus: {
    2: 1,
    3: 1,
    4: 2,
    5: 3,  // 5+ 연승/연패
  };
  
  // 레벨업 비용
  levelUpCost: {
    2: 2,   // 2XP 필요
    3: 6,
    4: 10,
    5: 20,
    6: 36,
    7: 56,
    8: 80,
    9: 84,
    10: 100,
  };
  
  // 리롤 비용
  rerollCost: 2;
}
```

### 4.2 경제 시뮬레이터

```typescript
class EconomySimulator {
  // N 라운드 후 예상 골드
  predictGold(currentState: PlayerState, rounds: number): number;
  
  // 특정 레벨 도달에 필요한 라운드/골드
  calculateLevelUpPlan(currentState: PlayerState, targetLevel: number): LevelUpPlan;
  
  // 롤다운 예산 계산 (남겨야 할 최소 골드)
  calculateRolldownBudget(currentState: PlayerState, stage: string): number;
  
  // 이코노미 전략 추천 (하이롤/로우롤/표준)
  recommendEconomyStrategy(gameState: GameState): EconomyStrategy;
}
```

---

## 📦 모듈 5: 전투 시뮬레이션

### 5.1 전투 시스템 개요

```typescript
interface CombatSystem {
  // 전투 시뮬레이션 실행
  simulate(
    playerBoard: Board,
    enemyBoard: Board,
    iterations?: number
  ): CombatResult;
  
  // 데미지 계산
  calculateDamage(attacker: Unit, defender: Unit, abilityDamage?: number): number;
  
  // 타겟팅 로직
  findTarget(unit: Unit, enemyUnits: Unit[]): Unit;
  
  // 마나 획득
  gainMana(unit: Unit, source: 'attack' | 'damage_taken' | 'ability'): void;
}
```

### 5.2 데미지 공식

```typescript
// 물리 데미지
function calculatePhysicalDamage(
  rawDamage: number,
  attackerAD: number,
  defenderArmor: number,
  critChance: number,
  critDamage: number
): number {
  const armorReduction = 100 / (100 + defenderArmor);
  const baseDamage = rawDamage * armorReduction;
  
  if (Math.random() < critChance) {
    return baseDamage * critDamage;
  }
  return baseDamage;
}

// 마법 데미지
function calculateMagicDamage(
  rawDamage: number,
  attackerAP: number,
  defenderMR: number
): number {
  const apMultiplier = 1 + (attackerAP / 100);
  const mrReduction = 100 / (100 + defenderMR);
  return rawDamage * apMultiplier * mrReduction;
}
```

### 5.3 전투 AI

```typescript
class CombatAI {
  // 유닛 행동 결정
  decideAction(unit: Unit, gameState: CombatState): Action;
  
  // 타겟 우선순위 계산
  calculateTargetPriority(unit: Unit, targets: Unit[]): Unit[];
  
  // 포지셔닝 평가
  evaluatePositioning(board: Board): PositioningScore;
  
  // 스킬 사용 타이밍
  shouldCastAbility(unit: Unit, gameState: CombatState): boolean;
}
```

### 5.4 몬테카를로 시뮬레이션

```typescript
class MonteCarloSimulator {
  // 승률 계산 (N회 시뮬레이션)
  calculateWinRate(
    playerBoard: Board,
    enemyBoard: Board,
    iterations: number = 1000
  ): WinRateResult;
  
  // 예상 데미지
  calculateExpectedDamage(
    playerBoard: Board,
    enemyBoard: Board
  ): DamageEstimate;
  
  // 포지셔닝 최적화
  optimizePositioning(
    board: Board,
    enemyBoard: Board
  ): OptimalPositioning;
}
```

---

## 📦 모듈 6: 최적화 엔진

### 6.1 픽 추천 시스템

```typescript
class PickAdvisor {
  // 현재 상점에서 최적의 선택 추천
  recommendPurchase(
    shop: Champion[],
    playerState: PlayerState,
    gameState: GameState
  ): PickRecommendation[];
  
  interface PickRecommendation {
    champion: Champion;
    priority: 'must_buy' | 'should_buy' | 'consider' | 'skip';
    reasons: string[];
    synergyImpact: SynergyDelta;
    economicImpact: number;
  }
}
```

### 6.2 롤다운 타이밍 분석기

```typescript
class RolldownAnalyzer {
  // 최적의 롤다운 타이밍 계산
  calculateOptimalTiming(
    playerState: PlayerState,
    targetComp: Composition
  ): RolldownPlan;
  
  interface RolldownPlan {
    recommendedStage: string;  // e.g., "4-2", "5-1"
    targetLevel: number;
    goldToSave: number;
    expectedRolls: number;
    hitProbability: number;    // 원하는 유닛 찾을 확률
    alternatives: RolldownPlan[];
  }
}
```

### 6.3 덱 빌더 & 피벗 분석기

```typescript
class CompBuilder {
  // 현재 유닛으로 최적의 조합 제안
  suggestComposition(
    currentUnits: Champion[],
    availableItems: Item[],
    stage: string
  ): CompositionSuggestion[];
  
  // 피벗 필요성 분석
  analyzePivotNeed(
    currentState: PlayerState,
    gameState: GameState
  ): PivotAnalysis;
  
  interface PivotAnalysis {
    shouldPivot: boolean;
    currentCompViability: number;  // 0-100
    alternativeComps: AlternativeComp[];
    pivotCost: number;  // 예상 골드 손실
    pivotTiming: string;
  }
}
```

### 6.4 종합 의사결정 엔진

```typescript
class DecisionEngine {
  // 현재 상황에서 최선의 행동 추천
  recommend(gameState: GameState): Decision {
    // 1. 경제 상태 분석
    const economyAnalysis = this.analyzeEconomy(gameState);
    
    // 2. 보드 강도 분석
    const boardStrength = this.analyzeBoardStrength(gameState);
    
    // 3. 상점 분석
    const shopAnalysis = this.analyzeShop(gameState);
    
    // 4. 경쟁자 분석
    const competitorAnalysis = this.analyzeCompetitors(gameState);
    
    // 5. 언락 진행도 분석 (Set 16)
    const unlockProgress = this.analyzeUnlockProgress(gameState);
    
    // 종합 판단
    return this.synthesizeDecision({
      economyAnalysis,
      boardStrength,
      shopAnalysis,
      competitorAnalysis,
      unlockProgress
    });
  }
  
  interface Decision {
    action: 'level_up' | 'roll' | 'save' | 'buy_specific' | 'sell_and_pivot';
    confidence: number;
    reasoning: string[];
    alternatives: Decision[];
  }
}
```

---

## 🔧 기술 스택 추천

### Backend
```
- Language: TypeScript/Node.js 또는 Python
- Framework: FastAPI (Python) 또는 Express.js (Node)
- Database: PostgreSQL (게임 데이터) + Redis (캐싱)
- 시뮬레이션: WebAssembly (성능 최적화 필요시)
```

### Frontend
```
- Framework: React + TypeScript
- State Management: Zustand 또는 Redux Toolkit
- UI Library: Tailwind CSS + Headless UI
- 차트/시각화: D3.js 또는 Recharts
```

### 데이터 파이프라인
```
- TFT API: Riot Games Data Dragon
- 데이터 업데이트: Cron job으로 패치마다 자동 갱신
- 크롤러: 메타 데이터 수집 (tftactics, lolchess 등)
```

### 아키텍처
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Frontend   │────▶│   API Layer  │────▶│  Simulator   │
│   (React)    │     │  (FastAPI)   │     │   Engine     │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                     ┌──────┴──────┐
                     ▼             ▼
              ┌──────────┐  ┌──────────┐
              │PostgreSQL│  │  Redis   │
              └──────────┘  └──────────┘
```

---

## 📊 데이터 수집 전략

### 필요 데이터

1. **정적 데이터** (패치마다 업데이트)
   - 챔피언 스탯, 스킬, 특성
   - 아이템 레시피, 효과
   - 시너지 브레이크포인트
   - 언락 조건

2. **동적 데이터** (실시간/일간)
   - 메타 덱 티어 리스트
   - 챔피언별 승률
   - 아이템 조합 승률
   - 포지셔닝 히트맵

### 데이터 소스

```typescript
const DATA_SOURCES = {
  official: {
    dataDragon: 'https://ddragon.leagueoflegends.com/',
    riotAPI: 'https://developer.riotgames.com/'
  },
  community: {
    tftactics: 'https://tftactics.gg/',
    lolchess: 'https://lolchess.gg/',
    metatft: 'https://www.metatft.com/'
  }
};
```

---

## 🚀 개발 로드맵

### Phase 1: 기초 시뮬레이터 (2-3주)
- [ ] 챔피언/특성 데이터 모델링
- [ ] 챔피언 풀 시스템 구현
- [ ] 상점 시스템 구현
- [ ] 시너지 계산기 구현

### Phase 2: 경제 & 아이템 (1-2주)
- [ ] 경제 시스템 구현
- [ ] 아이템 시스템 구현
- [ ] 아이템 최적화 로직

### Phase 3: 전투 시뮬레이션 (3-4주)
- [ ] 기본 전투 로직
- [ ] 데미지 계산 공식
- [ ] 스킬 시스템
- [ ] 몬테카를로 시뮬레이션

### Phase 4: 최적화 엔진 (2-3주)
- [ ] 픽 추천 시스템
- [ ] 롤다운 타이밍 분석기
- [ ] 덱 빌더
- [ ] 피벗 분석기

### Phase 5: UI/UX (2-3주)
- [ ] 메인 대시보드
- [ ] 시뮬레이터 인터페이스
- [ ] 실시간 추천 패널
- [ ] 통계 시각화

---

## 📝 구현 우선순위

### Must Have (핵심)
1. 챔피언 풀 & 상점 시스템
2. 시너지 계산기
3. 경제 시스템
4. 픽 추천 시스템

### Should Have (중요)
1. 아이템 최적화
2. 롤다운 타이밍 분석
3. 기본 전투 시뮬레이션
4. 덱 빌더

### Nice to Have (부가)
1. 고급 전투 시뮬레이션 (몬테카를로)
2. 피벗 분석기
3. 실시간 게임 오버레이
4. 경쟁자 추적

---

## ⚠️ 주의사항

1. **Riot API 정책 준수**: Rate limit, ToS 확인
2. **패치 대응**: 2주마다 패치가 있으므로 데이터 업데이트 자동화 필수
3. **성능 최적화**: 전투 시뮬레이션은 연산 집약적이므로 웹워커/WASM 고려
4. **Set 전환 대비**: Set 17 출시 시 데이터 구조 마이그레이션 계획

---

## 🔗 참고 자료

- [TFT Set 16 공식 페이지](https://teamfighttactics.leagueoflegends.com/)
- [Riot Developer Portal](https://developer.riotgames.com/)
- [TFTactics](https://tftactics.gg/)
- [MetaTFT](https://www.metatft.com/)
