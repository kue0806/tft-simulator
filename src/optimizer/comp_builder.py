"""Composition Builder.

Provides recommendations for:
- Meta composition templates
- Composition building from current board
- Synergy optimization
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, TYPE_CHECKING
from enum import Enum, auto

from src.core.synergy_calculator import SynergyCalculator
from src.data.loaders import load_champions, load_traits

if TYPE_CHECKING:
    from src.core.game_state import PlayerState
    from src.core.player_units import ChampionInstance
    from src.data.models.champion import Champion


class CompStyle(Enum):
    """Composition play style."""

    REROLL = "reroll"  # Reroll for 3-star low costs
    STANDARD = "standard"  # Standard 4-cost carry
    FAST_9 = "fast_9"  # Fast 9 for 5-cost carries
    FLEX = "flex"  # Flexible/adaptive


@dataclass
class CompTemplate:
    """A composition template/blueprint."""

    name: str
    style: CompStyle
    core_units: List[str]  # Required champion IDs
    flex_units: List[str]  # Optional/flexible champion IDs
    carry: str  # Main carry champion ID
    items_priority: Dict[str, List[str]]  # Champion ID -> item priority list

    # Synergy targets
    target_synergies: Dict[str, int]  # trait_id -> target count

    # Meta information
    tier: str  # "S", "A", "B", "C"
    difficulty: str  # "easy", "medium", "hard"
    description: str

    # Power spike timings
    power_spikes: List[str]  # "4-1", "4-5", etc.


@dataclass
class CompRecommendation:
    """A composition recommendation."""

    template: CompTemplate
    match_score: float  # Match with current board (0-100)
    missing_units: List[str]  # Units still needed
    current_units: List[str]  # Units already owned
    transition_cost: int  # Estimated roll cost
    estimated_strength: float  # Estimated power level


class CompBuilder:
    """
    Composition builder and recommender.

    Provides meta composition recommendations and helps optimize
    synergies for the current board.

    Usage:
        builder = CompBuilder()
        recommendations = builder.recommend(player_state)
    """

    def __init__(self):
        """Initialize composition builder."""
        self.synergy_calc = SynergyCalculator()
        self._champions: Optional[Dict[str, "Champion"]] = None
        self._traits = None

        # Load meta templates
        self.templates = self._load_templates()

    @property
    def champions(self) -> Dict[str, "Champion"]:
        """Lazy load champions."""
        if self._champions is None:
            all_champs = load_champions()
            self._champions = {c.id: c for c in all_champs}
        return self._champions

    def recommend(
        self,
        player: "PlayerState",
        top_n: int = 3,
        style_filter: Optional[CompStyle] = None,
    ) -> List[CompRecommendation]:
        """
        Recommend compositions based on current state.

        Args:
            player: Player state.
            top_n: Number of recommendations to return.
            style_filter: Filter by specific style.

        Returns:
            List of composition recommendations.
        """
        recommendations = []
        current_units = self._get_current_units(player)

        for template in self.templates:
            if style_filter and template.style != style_filter:
                continue

            rec = self._evaluate_template(template, player, current_units)
            recommendations.append(rec)

        # Sort by match score
        recommendations.sort(key=lambda r: r.match_score, reverse=True)

        return recommendations[:top_n]

    def build_from_scratch(
        self, target_traits: Dict[str, int], level: int = 8
    ) -> List[str]:
        """
        Build optimal composition for target synergies.

        Args:
            target_traits: Target synergies {trait_id: count}.
            level: Max units to place.

        Returns:
            List of recommended champion IDs.
        """
        selected: List[str] = []
        remaining_traits = target_traits.copy()

        # Greedy selection
        while len(selected) < level and remaining_traits:
            best_unit = None
            best_score = 0

            for champ_id, champ in self.champions.items():
                if champ_id in selected:
                    continue

                score = self._calculate_trait_coverage(champ, remaining_traits)
                if score > best_score:
                    best_score = score
                    best_unit = champ

            if best_unit is None:
                break

            selected.append(best_unit.id)

            # Update remaining traits
            for trait in best_unit.traits:
                if trait in remaining_traits:
                    remaining_traits[trait] -= 1
                    if remaining_traits[trait] <= 0:
                        del remaining_traits[trait]

        return selected

    def suggest_additions(
        self, player: "PlayerState", slots_available: int = 1
    ) -> List[str]:
        """
        Suggest units to add to current board.

        Args:
            player: Player state.
            slots_available: Number of board slots available.

        Returns:
            List of recommended champion IDs.
        """
        board_units = list(player.units.board.values())
        current_synergies = self.synergy_calc.calculate_synergies(board_units)

        suggestions: List[tuple] = []

        for champ_id, champ in self.champions.items():
            # Skip if already on board
            if self._is_on_board(player, champ_id):
                continue

            # Calculate addition value
            score = self._calculate_addition_value(champ, current_synergies, player)
            suggestions.append((champ_id, score))

        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in suggestions[: slots_available * 3]]

    def find_best_flex(
        self, player: "PlayerState", slot_count: int = 1
    ) -> List[str]:
        """
        Find best flexible units for remaining slots.

        Args:
            player: Player state.
            slot_count: Number of slots to fill.

        Returns:
            List of recommended champion IDs.
        """
        return self.suggest_additions(player, slot_count)

    def _evaluate_template(
        self,
        template: CompTemplate,
        player: "PlayerState",
        current_units: Set[str],
    ) -> CompRecommendation:
        """
        Evaluate how well a template matches current state.

        Args:
            template: Composition template.
            player: Player state.
            current_units: Currently owned unit IDs.

        Returns:
            CompRecommendation with evaluation.
        """
        core_set = set(template.core_units)

        # Calculate matches
        matched = current_units & core_set
        missing = core_set - current_units

        # Match score
        match_score = (len(matched) / len(core_set) * 100) if core_set else 0

        # Transition cost estimate
        transition_cost = len(missing) * 10  # ~10 rolls per unit

        # Estimated strength
        estimated_strength = self._estimate_comp_strength(template)

        return CompRecommendation(
            template=template,
            match_score=match_score,
            missing_units=list(missing),
            current_units=list(matched),
            transition_cost=transition_cost,
            estimated_strength=estimated_strength,
        )

    def _get_current_units(self, player: "PlayerState") -> Set[str]:
        """Get all unique champion IDs on board and bench."""
        units = set()

        # Board units
        for instance in player.units.board.values():
            units.add(instance.champion.id)

        # Bench units
        for instance in player.units.bench:
            if instance:
                units.add(instance.champion.id)

        return units

    def _is_on_board(self, player: "PlayerState", champion_id: str) -> bool:
        """Check if champion is on board."""
        for instance in player.units.board.values():
            if instance.champion.id == champion_id:
                return True
        return False

    def _calculate_trait_coverage(
        self, champion: "Champion", remaining_traits: Dict[str, int]
    ) -> float:
        """Calculate how well champion covers remaining trait requirements."""
        score = 0.0
        for trait in champion.traits:
            if trait in remaining_traits:
                score += remaining_traits[trait]
        return score

    def _calculate_addition_value(
        self,
        champion: "Champion",
        current_synergies: Dict,
        player: "PlayerState",
    ) -> float:
        """Calculate value of adding a champion to board."""
        score = 0.0

        for trait in champion.traits:
            if trait in current_synergies:
                # Strengthens existing synergy
                active_trait = current_synergies[trait]
                if active_trait.is_active:
                    score += 10
                else:
                    # Could activate it
                    score += 15
            else:
                # New synergy (less valuable usually)
                score += 5

        # Adjust for cost efficiency
        score /= champion.cost

        return score

    def _estimate_comp_strength(self, template: CompTemplate) -> float:
        """Estimate composition strength based on tier."""
        tier_scores = {"S": 95, "A": 85, "B": 75, "C": 65}
        return tier_scores.get(template.tier, 70)

    def get_synergy_breakdown(
        self, champion_ids: List[str]
    ) -> Dict[str, int]:
        """
        Get synergy breakdown for a list of champions.

        Args:
            champion_ids: List of champion IDs.

        Returns:
            Dict mapping trait_id to count.
        """
        from src.core.player_units import ChampionInstance

        # Create temporary instances
        instances = []
        for champ_id in champion_ids:
            if champ_id in self.champions:
                inst = ChampionInstance(champion=self.champions[champ_id])
                instances.append(inst)

        # Calculate synergies
        synergies = self.synergy_calc.calculate_synergies(instances)

        return {
            trait_id: active.count
            for trait_id, active in synergies.items()
        }

    def _load_templates(self) -> List[CompTemplate]:
        """
        Load meta composition templates.

        In a full implementation, this would load from a data file.
        Here we provide example templates.
        """
        return [
            CompTemplate(
                name="Arcana Ahri Reroll",
                style=CompStyle.REROLL,
                core_units=[
                    "TFT16_Ahri",
                    "TFT16_Twitch",
                    "TFT16_Ziggs",
                    "TFT16_Xerath",
                ],
                flex_units=["TFT16_TahmKench", "TFT16_Lulu"],
                carry="TFT16_Ahri",
                items_priority={
                    "TFT16_Ahri": [
                        "JeweledGauntlet",
                        "GiantSlayer",
                        "Quicksilver",
                    ]
                },
                target_synergies={"Arcana": 6, "Mage": 2},
                tier="A",
                difficulty="medium",
                description="3-star Ahri as main carry",
                power_spikes=["3-2", "4-1"],
            ),
            CompTemplate(
                name="Demacia Vertical",
                style=CompStyle.STANDARD,
                core_units=[
                    "TFT16_Garen",
                    "TFT16_Jarvan",
                    "TFT16_Lux",
                    "TFT16_Galio",
                    "TFT16_Sona",
                ],
                flex_units=["TFT16_Kayle", "TFT16_Morgana"],
                carry="TFT16_Lux",
                items_priority={
                    "TFT16_Lux": [
                        "JeweledGauntlet",
                        "HextechGunblade",
                        "Morellonomicon",
                    ]
                },
                target_synergies={"Demacia": 6, "Mage": 2},
                tier="A",
                difficulty="easy",
                description="Demacia vertical with Lux carry",
                power_spikes=["4-2", "4-5"],
            ),
            CompTemplate(
                name="Fast 9 Legendaries",
                style=CompStyle.FAST_9,
                core_units=[
                    "TFT16_Aurelion_Sol",
                    "TFT16_Smolder",
                    "TFT16_Briar",
                ],
                flex_units=["TFT16_Xerath", "TFT16_Diana", "TFT16_Morgana"],
                carry="TFT16_Aurelion_Sol",
                items_priority={
                    "TFT16_Aurelion_Sol": [
                        "JeweledGauntlet",
                        "HextechGunblade",
                        "BlueBuff",
                    ]
                },
                target_synergies={"Dragon": 2, "Mage": 4},
                tier="S",
                difficulty="hard",
                description="Fast 9 with legendary carries",
                power_spikes=["5-1", "5-2"],
            ),
            CompTemplate(
                name="Flex AD Carry",
                style=CompStyle.FLEX,
                core_units=[
                    "TFT16_Jinx",
                    "TFT16_Caitlyn",
                    "TFT16_Miss_Fortune",
                ],
                flex_units=[
                    "TFT16_Leona",
                    "TFT16_Vi",
                    "TFT16_Braum",
                ],
                carry="TFT16_Jinx",
                items_priority={
                    "TFT16_Jinx": [
                        "InfinityEdge",
                        "LastWhisper",
                        "GiantSlayer",
                    ]
                },
                target_synergies={"Gunner": 4, "Hextech": 2},
                tier="B",
                difficulty="medium",
                description="Flexible AD carry composition",
                power_spikes=["4-1", "4-2"],
            ),
        ]

    def add_custom_template(self, template: CompTemplate) -> None:
        """Add a custom composition template."""
        self.templates.append(template)

    def get_templates_by_style(self, style: CompStyle) -> List[CompTemplate]:
        """Get all templates of a specific style."""
        return [t for t in self.templates if t.style == style]

    def get_templates_by_tier(self, tier: str) -> List[CompTemplate]:
        """Get all templates of a specific tier."""
        return [t for t in self.templates if t.tier == tier]
