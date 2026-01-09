"""
Optimizer recommendation service.
"""

from typing import List, Dict, Optional, Any

from src.optimizer import (
    PickAdvisor,
    RolldownPlanner,
    CompBuilder,
    PivotAnalyzer,
    BoardOptimizer,
)
from src.optimizer.pick_advisor import PickReason

from ..schemas.optimizer import (
    PickAdviceResponse,
    PickRecommendationSchema,
    PickReasonEnum,
    RolldownPlanResponse,
    RolldownStrategyEnum,
    CompRecommendationSchema,
    CompTemplateSchema,
    CompStyleEnum,
    PivotAdviceResponse,
    PivotOptionSchema,
    OptimizeBoardResponse,
    BoardLayoutSchema,
    PositionSchema,
    PositionRecommendationSchema,
)
from .game_service import GameService


class OptimizerService:
    """Optimizer recommendation service."""

    # Mapping from PickReason to PickReasonEnum
    _PICK_REASON_MAP = {
        PickReason.UPGRADE_2STAR: PickReasonEnum.UPGRADE_2STAR,
        PickReason.UPGRADE_3STAR: PickReasonEnum.UPGRADE_3STAR,
        PickReason.SYNERGY_ACTIVATE: PickReasonEnum.SYNERGY_ACTIVATE,
        PickReason.SYNERGY_UPGRADE: PickReasonEnum.SYNERGY_UPGRADE,
        PickReason.CORE_CARRY: PickReasonEnum.CORE_CARRY,
        PickReason.STRONG_UNIT: PickReasonEnum.STRONG_UNIT,
        PickReason.ECONOMY_PAIR: PickReasonEnum.ECONOMY_PAIR,
    }

    def __init__(self, game_service: GameService):
        self.game_service = game_service
        self.pick_advisor = PickAdvisor()
        self.rolldown_planner = RolldownPlanner()
        self.comp_builder = CompBuilder()
        self.pivot_analyzer = PivotAnalyzer()
        self.board_optimizer = BoardOptimizer()

    def get_pick_advice(
        self,
        game_id: str,
        player_id: int,
        target_comp: Optional[List[str]] = None,
    ) -> PickAdviceResponse:
        """Get shop purchase recommendations."""
        player = self.game_service.get_player_raw(game_id, player_id)
        if not player:
            raise ValueError("Player not found")

        # Create mock player state for advisor
        mock_player = self._create_mock_player_for_advisor(player)
        shop = self.game_service._shops.get(game_id, {}).get(player_id)

        if shop:
            # Set up shop slots
            mock_player.shop = self._create_mock_shop(shop)

        advice = self.pick_advisor.analyze(mock_player, target_comp=target_comp)

        # Convert to response schema
        recommendations = []
        for rec in advice.recommendations:
            recommendations.append(
                PickRecommendationSchema(
                    champion_id=rec.champion_id,
                    champion_name=rec.champion_name,
                    shop_index=rec.shop_index,
                    score=rec.score,
                    reasons=[self._PICK_REASON_MAP.get(r, PickReasonEnum.STRONG_UNIT) for r in rec.reasons],
                    cost=rec.cost,
                    copies_owned=rec.copies_owned,
                    copies_needed=rec.copies_needed,
                )
            )

        return PickAdviceResponse(
            recommendations=recommendations,
            should_refresh=advice.should_refresh,
            refresh_reason=advice.refresh_reason,
            gold_to_save=advice.gold_to_save,
        )

    def get_rolldown_plan(
        self,
        game_id: str,
        player_id: int,
        target_units: List[str],
        target_stars: Optional[Dict[str, int]] = None,
    ) -> RolldownPlanResponse:
        """Get rolldown strategy plan."""
        player = self.game_service.get_player_raw(game_id, player_id)
        game = self.game_service._games.get(game_id)

        if not player or not game:
            raise ValueError("Player or game not found")

        mock_player = self._create_mock_player_for_advisor(player)
        mock_game = self._create_mock_game_state(game)

        plan = self.rolldown_planner.create_plan(
            mock_player, mock_game, target_units, target_stars
        )

        return RolldownPlanResponse(
            strategy=RolldownStrategyEnum(plan.strategy.value),
            current_phase=plan.current_phase,
            is_rolldown_now=plan.is_rolldown_now,
            roll_budget=plan.roll_budget,
            level_budget=plan.level_budget,
            save_amount=plan.save_amount,
            hit_probability=plan.hit_probability,
            expected_rolls=plan.expected_rolls,
            advice=plan.advice,
        )

    def get_comp_recommendations(
        self,
        game_id: str,
        player_id: int,
        style_filter: Optional[CompStyleEnum] = None,
        top_n: int = 3,
    ) -> List[CompRecommendationSchema]:
        """Get composition recommendations."""
        player = self.game_service.get_player_raw(game_id, player_id)
        if not player:
            raise ValueError("Player not found")

        mock_player = self._create_mock_player_for_advisor(player)

        # Convert style filter
        from src.optimizer.comp_builder import CompStyle

        internal_style = None
        if style_filter:
            internal_style = CompStyle(style_filter.value)

        recommendations = self.comp_builder.recommend(
            mock_player, top_n=top_n, style_filter=internal_style
        )

        result = []
        for rec in recommendations:
            template = rec.template
            result.append(
                CompRecommendationSchema(
                    template=CompTemplateSchema(
                        name=template.name,
                        style=CompStyleEnum(template.style.value),
                        core_units=template.core_units,
                        carry=template.carry,
                        tier=template.tier,
                        difficulty=template.difficulty,
                        description=template.description,
                    ),
                    match_score=rec.match_score,
                    missing_units=rec.missing_units,
                    current_units=rec.current_units,
                    transition_cost=rec.transition_cost,
                )
            )

        return result

    def get_all_templates(self) -> List[CompTemplateSchema]:
        """Get all composition templates."""
        templates = []
        for t in self.comp_builder.templates:
            templates.append(
                CompTemplateSchema(
                    name=t.name,
                    style=CompStyleEnum(t.style.value),
                    core_units=t.core_units,
                    carry=t.carry,
                    tier=t.tier,
                    difficulty=t.difficulty,
                    description=t.description,
                )
            )
        return templates

    def get_pivot_advice(
        self,
        game_id: str,
        player_id: int,
        current_comp_name: Optional[str] = None,
        contested_units: Optional[List[str]] = None,
    ) -> PivotAdviceResponse:
        """Get pivot analysis."""
        player = self.game_service.get_player_raw(game_id, player_id)
        game = self.game_service._games.get(game_id)

        if not player or not game:
            raise ValueError("Player or game not found")

        mock_player = self._create_mock_player_for_advisor(player)
        mock_game = self._create_mock_game_state(game)

        # Find current comp template
        current_comp = None
        if current_comp_name:
            for t in self.comp_builder.templates:
                if t.name == current_comp_name:
                    current_comp = t
                    break

        advice = self.pivot_analyzer.analyze(
            mock_player, mock_game, current_comp, contested_units
        )

        options = []
        for opt in advice.options:
            options.append(
                PivotOptionSchema(
                    target_comp_name=opt.target_comp.name if opt.target_comp else "Unknown",
                    shared_units=opt.shared_units,
                    units_to_sell=opt.units_to_sell,
                    units_to_buy=opt.units_to_buy,
                    total_cost=opt.total_cost,
                    success_probability=opt.success_probability,
                    risk_level=opt.risk_level,
                )
            )

        return PivotAdviceResponse(
            should_pivot=advice.should_pivot,
            urgency=advice.urgency,
            current_comp_health=advice.current_comp_health,
            options=options,
            explanation=advice.explanation,
        )

    def optimize_board(
        self,
        game_id: str,
        player_id: int,
        iterations: int = 100,
    ) -> OptimizeBoardResponse:
        """Optimize board positioning."""
        player = self.game_service.get_player_raw(game_id, player_id)
        if not player:
            raise ValueError("Player not found")

        mock_player = self._create_mock_player_for_advisor(player)

        layout = self.board_optimizer.optimize(mock_player, iterations=iterations)

        # Convert positions
        positions = {}
        for unit_id, pos in layout.positions.items():
            positions[unit_id] = PositionSchema(row=pos.row, col=pos.col)

        # Get recommendations for each unit
        recommendations = []
        for unit_id in player.units.board.keys():
            suggestions = self.board_optimizer.suggest_position(mock_player, unit_id)
            if suggestions:
                best = suggestions[0]
                recommendations.append(
                    PositionRecommendationSchema(
                        unit_id=unit_id,
                        position=PositionSchema(
                            row=best.position.row, col=best.position.col
                        ),
                        score=best.score,
                        reasons=best.reasons,
                    )
                )

        return OptimizeBoardResponse(
            layout=BoardLayoutSchema(
                positions=positions,
                total_score=layout.total_score,
                win_rate=layout.win_rate,
                description=layout.description,
            ),
            unit_recommendations=recommendations,
        )

    def suggest_unit_position(
        self,
        game_id: str,
        player_id: int,
        unit_id: str,
    ) -> List[PositionRecommendationSchema]:
        """Get position suggestions for a specific unit."""
        player = self.game_service.get_player_raw(game_id, player_id)
        if not player:
            raise ValueError("Player not found")

        mock_player = self._create_mock_player_for_advisor(player)
        suggestions = self.board_optimizer.suggest_position(mock_player, unit_id)

        result = []
        for s in suggestions:
            result.append(
                PositionRecommendationSchema(
                    unit_id=unit_id,
                    position=PositionSchema(row=s.position.row, col=s.position.col),
                    score=s.score,
                    reasons=s.reasons,
                )
            )

        return result

    def _create_mock_player_for_advisor(self, player) -> Any:
        """Create mock player state compatible with optimizer modules."""

        class MockUnits:
            def __init__(self, board, bench):
                self.board = board
                self.bench = bench

        class MockPlayer:
            def __init__(self, p):
                self.gold = p.gold
                self.level = p.level
                self.health = p.hp
                self.xp = p.xp
                self.units = MockUnits(
                    {uid: unit for uid, unit in p.units.board.items()},
                    [u for u in p.units.bench if u],
                )
                self.shop = None

        return MockPlayer(player)

    def _create_mock_shop(self, shop) -> Any:
        """Create mock shop for advisor."""

        class MockShop:
            def __init__(self, s):
                self.slots = []
                for slot in s.slots:
                    if slot and not slot["is_purchased"]:
                        self.slots.append(slot["champion"])
                    else:
                        self.slots.append(None)

        return MockShop(shop)

    def _create_mock_game_state(self, game) -> Any:
        """Create mock game state."""

        class MockStageManager:
            def __init__(self, sm):
                self._stage = sm.get_stage_string()

            def get_stage_string(self):
                return self._stage

        class MockGame:
            def __init__(self, g):
                self.stage_manager = MockStageManager(g.stage_manager)

        return MockGame(game)
