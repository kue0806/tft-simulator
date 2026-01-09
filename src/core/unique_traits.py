"""Unique Trait Handlers for TFT Set 16.

Handle special/unique traits that have complex mechanics.
"""

import random
from typing import Optional, Any
from abc import ABC, abstractmethod

from src.core.player_units import ChampionInstance


class UniqueTraitHandler(ABC):
    """Base class for unique trait handlers."""

    @abstractmethod
    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        """Apply the trait effect."""
        raise NotImplementedError

    @abstractmethod
    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        """Get the stat bonuses from this trait."""
        raise NotImplementedError

    def get_bonuses(
        self,
        champion: ChampionInstance,
        active_trait: Any,
    ) -> dict[str, float]:
        """
        Get bonuses for a specific champion from this trait.

        This is called by StatCalculator._apply_special_effects() for
        each champion that has this trait active.

        Args:
            champion: The champion instance to get bonuses for
            active_trait: The ActiveTrait object with current level info

        Returns:
            Dictionary of stat bonuses
        """
        # Default implementation: use get_bonus with empty champion list
        # Subclasses can override for champion-specific bonuses
        return self.get_bonus(None, [champion])


class DemaciaHandler(UniqueTraitHandler):
    """
    Demacia: Rally when team takes damage, gaining Armor and Magic Resist.
    Breakpoints: 3/5/7/11 -> 12/25/40/150 Armor & MR
    """

    # Breakpoint values: {count: (armor, mr)} - Set 16 official values
    BREAKPOINTS = {
        3: (12, 12),
        5: (35, 35),
        7: (35, 35),  # Same as 5, but with 5% max health smite on Rally
        11: (150, 150),
    }

    def __init__(self):
        self.rally_count = 0
        self.last_health_percent = 100.0

    def on_health_lost(self, current_health_percent: float) -> bool:
        """
        Check if rally triggers.

        Args:
            current_health_percent: Current team health as percentage

        Returns:
            True if rally was triggered
        """
        # Rally triggers every time we cross a 25% threshold
        thresholds_crossed_old = int((100 - self.last_health_percent) / 25)
        thresholds_crossed_new = int((100 - current_health_percent) / 25)

        if thresholds_crossed_new > thresholds_crossed_old:
            self.rally_count += thresholds_crossed_new - thresholds_crossed_old
            self.last_health_percent = current_health_percent
            return True

        self.last_health_percent = current_health_percent
        return False

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        """Apply rally effect to Demacia champions."""
        pass  # Combat system will handle this

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        """Return armor/MR based on breakpoint and rally count."""
        count = len(champions)
        armor, mr = 0, 0

        # Get base stats from breakpoint
        if count >= 11:
            armor, mr = self.BREAKPOINTS[11]
        elif count >= 7:
            armor, mr = self.BREAKPOINTS[7]
        elif count >= 5:
            armor, mr = self.BREAKPOINTS[5]
        elif count >= 3:
            armor, mr = self.BREAKPOINTS[3]

        # Rally bonus: each rally adds to the armor/MR
        rally_bonus = self.rally_count * 5  # +5 per rally

        return {
            "armor": armor + rally_bonus,
            "magic_resist": mr + rally_bonus,
        }

    def reset(self) -> None:
        """Reset rally count for new combat."""
        self.rally_count = 0
        self.last_health_percent = 100.0


class IoniaHandler(UniqueTraitHandler):
    """
    Ionia: Random path each game.
    Paths: Spirit, Generosity, Enlightenment, Transcendence, Precision
    """

    PATHS = {
        "spirit": {
            "description": "Gain health, gain stacking AD/AP on attacks",
            "health": 200,
            "stacking_ad_ap": 2,
        },
        "generosity": {
            "description": "Gain gold on takedown, bonus AD/AP per gold",
            "gold_on_takedown": 1,
            "ad_ap_per_10_gold": 5,
        },
        "enlightenment": {
            "description": "Bonus AD/AP based on player level",
            "ad_ap_per_level": 5,
        },
        "transcendence": {
            "description": "Gain health, deal bonus magic damage",
            "health": 150,
            "bonus_magic_damage_percent": 10,
        },
        "precision": {
            "description": "Abilities can crit, gain stacking AD/AP",
            "crit_abilities": True,
            "stacking_ad_ap": 3,
        },
    }

    def __init__(self):
        self.current_path: Optional[str] = None

    def roll_path(self) -> str:
        """Randomly select path at game start."""
        self.current_path = random.choice(list(self.PATHS.keys()))
        return self.current_path

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        """Apply path effects to Ionia champions."""
        pass  # Path effects are applied during combat

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        """Get bonuses based on current path."""
        if self.current_path is None:
            return {}

        path_data = self.PATHS[self.current_path]
        bonuses = {}

        if "health" in path_data:
            bonuses["health"] = path_data["health"]

        if "ad_ap_per_level" in path_data:
            # Assume game_state has player level
            player_level = getattr(game_state, "level", 5)
            bonus = path_data["ad_ap_per_level"] * player_level
            bonuses["attack_damage"] = bonus
            bonuses["ability_power"] = bonus

        return bonuses


class NoxusHandler(UniqueTraitHandler):
    """
    Noxus: Summon Atakhan when enemies lose 15% HP.
    Atakhan power scales with Noxian star levels.
    """

    def __init__(self):
        self.atakhan_summoned = False
        self.enemy_damage_taken = 0.0

    def calculate_atakhan_power(
        self, noxian_champions: list[ChampionInstance]
    ) -> dict[str, float]:
        """
        Calculate Atakhan's stats based on Noxian star levels.

        Args:
            noxian_champions: List of Noxus champions

        Returns:
            Atakhan stats
        """
        total_stars = sum(c.star_level for c in noxian_champions)

        # Base stats + scaling per star
        return {
            "health": 1000 + (total_stars * 200),
            "attack_damage": 50 + (total_stars * 15),
            "armor": 30 + (total_stars * 5),
            "magic_resist": 30 + (total_stars * 5),
        }

    def on_enemy_damage(self, damage_percent: float) -> bool:
        """
        Track enemy damage for Atakhan summon.

        Args:
            damage_percent: Percent of enemy health lost

        Returns:
            True if Atakhan should be summoned
        """
        self.enemy_damage_taken += damage_percent
        if self.enemy_damage_taken >= 15 and not self.atakhan_summoned:
            self.atakhan_summoned = True
            return True
        return False

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        """Apply Noxus effects."""
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        """Noxus champions get bonus damage."""
        return {"bonus_damage_percent": 10}

    def reset(self) -> None:
        """Reset for new combat."""
        self.atakhan_summoned = False
        self.enemy_damage_taken = 0.0


class VoidHandler(UniqueTraitHandler):
    """
    Void: Grants mutations to void champions and attack speed.
    Breakpoints: 2/4/6/9 -> 8/18/28/35% attack speed
    """

    # Attack speed by breakpoint
    ATTACK_SPEED_BREAKPOINTS = {
        2: 8,
        4: 18,
        6: 28,
        9: 35,
    }

    MUTATIONS = {
        "vampiric": {
            "description": "Gain omnivamp",
            "omnivamp": 15,
        },
        "voracious": {
            "description": "Gain attack speed on kill",
            "attack_speed_on_kill": 20,
        },
        "riftborn": {
            "description": "Deal increased damage",
            "damage_amp": 15,
        },
        "adaptive": {
            "description": "Gain armor and MR",
            "armor": 25,
            "magic_resist": 25,
        },
        "unstable": {
            "description": "Abilities deal bonus true damage",
            "bonus_true_damage": 10,
        },
    }

    def __init__(self):
        self.champion_mutations: dict[str, str] = {}  # champion_id -> mutation

    def assign_mutation(
        self, champion: ChampionInstance, mutation: str
    ) -> bool:
        """
        Assign a mutation to a void champion.

        Args:
            champion: The champion to mutate
            mutation: The mutation to assign

        Returns:
            True if mutation was assigned
        """
        if mutation not in self.MUTATIONS:
            return False

        self.champion_mutations[champion.champion.id] = mutation
        return True

    def get_champion_mutation(self, champion: ChampionInstance) -> Optional[str]:
        """Get the mutation assigned to a champion."""
        return self.champion_mutations.get(champion.champion.id)

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        """Apply mutations to void champions."""
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        """Return attack speed and mutation bonuses."""
        count = len(champions)
        bonuses: dict[str, float] = {}

        # Base attack speed from breakpoint
        if count >= 9:
            bonuses["attack_speed"] = self.ATTACK_SPEED_BREAKPOINTS[9]
        elif count >= 6:
            bonuses["attack_speed"] = self.ATTACK_SPEED_BREAKPOINTS[6]
        elif count >= 4:
            bonuses["attack_speed"] = self.ATTACK_SPEED_BREAKPOINTS[4]
        elif count >= 2:
            bonuses["attack_speed"] = self.ATTACK_SPEED_BREAKPOINTS[2]

        # Add mutation bonuses
        for champ in champions:
            mutation = self.get_champion_mutation(champ)
            if mutation and mutation in self.MUTATIONS:
                for stat, value in self.MUTATIONS[mutation].items():
                    if stat != "description" and isinstance(value, (int, float)):
                        bonuses[stat] = bonuses.get(stat, 0) + value

        return bonuses


class YordleHandler(UniqueTraitHandler):
    """
    Yordle: Scaling bonuses per Yordle fielded.
    3-stars grant 50% more!
    """

    BASE_HEALTH = 50
    BASE_ATTACK_SPEED = 0.07

    def __init__(self):
        pass

    def calculate_bonus(
        self, yordle_champions: list[ChampionInstance]
    ) -> dict[str, float]:
        """
        Calculate Yordle bonuses based on fielded Yordles.

        Args:
            yordle_champions: List of Yordle champion instances

        Returns:
            Total bonuses
        """
        total_hp = 0.0
        total_as = 0.0

        for champ in yordle_champions:
            multiplier = 1.5 if champ.star_level == 3 else 1.0
            total_hp += self.BASE_HEALTH * multiplier
            total_as += self.BASE_ATTACK_SPEED * multiplier

        return {"health": total_hp, "attack_speed": total_as}

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        """Apply Yordle bonuses."""
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        """Get Yordle bonuses."""
        return self.calculate_bonus(champions)


class BilgewaterHandler(UniqueTraitHandler):
    """
    Bilgewater: Earn Silver Serpents each round to spend in Black Market.
    Breakpoints: 3/5/7/10 -> 18/30/55/150 Serpents per round
    """

    # Serpents per round by breakpoint (Set 16 official values)
    SERPENTS_PER_ROUND = {
        3: 15,
        5: 35,
        7: 65,
        10: 150,
    }

    def __init__(self):
        self.silver_serpents = 0
        self.serpents_per_round = 0

    def on_round_end(self) -> int:
        """Earn Silver Serpents at round end. Returns serpents earned."""
        earned = self.serpents_per_round
        self.silver_serpents += earned
        return earned

    def on_kill(self, count: int = 1) -> None:
        """Legacy: can still earn bonus serpents on kills."""
        self.silver_serpents += count

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        """Set serpents per round based on breakpoint."""
        count = len(champions)
        if count >= 10:
            self.serpents_per_round = self.SERPENTS_PER_ROUND[10]
        elif count >= 7:
            self.serpents_per_round = self.SERPENTS_PER_ROUND[7]
        elif count >= 5:
            self.serpents_per_round = self.SERPENTS_PER_ROUND[5]
        elif count >= 3:
            self.serpents_per_round = self.SERPENTS_PER_ROUND[3]
        else:
            self.serpents_per_round = 0

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        """Bonus AD/AP based on serpents spent on upgrades."""
        # Serpents are spent in Black Market, not directly as stats
        # This can be used to track spent serpents for stat bonuses
        return {}


# =========================================================================
# ADDITIONAL ORIGIN HANDLERS
# =========================================================================


class FreljordHandler(UniqueTraitHandler):
    """
    Freljord: Summon ice towers that slow and damage enemies.
    Also grants health% and damage amp.
    """

    def __init__(self):
        self.tower_count = 0

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        """Set up ice towers based on trait level."""
        count = len(champions)
        if count >= 7:
            self.tower_count = 3
        elif count >= 5:
            self.tower_count = 2
        elif count >= 3:
            self.tower_count = 1
        else:
            self.tower_count = 0

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        """Return bonuses based on breakpoint - Set 16 official values."""
        count = len(champions)
        if count >= 7:
            return {"health_percent": 16, "damage_amp": 22}
        elif count >= 5:
            return {"health_percent": 12, "damage_amp": 16}
        elif count >= 3:
            return {"health_percent": 8, "damage_amp": 10}
        return {}


class ShurimaHandler(UniqueTraitHandler):
    """
    Shurima: Gain health% and armor/MR. At 4, champions Ascend.
    """

    def __init__(self):
        self.ascension_active = False

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        self.ascension_active = len(champions) >= 4

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        count = len(champions)
        if count >= 4:
            return {"health_percent": 20, "armor": 50, "magic_resist": 50}
        elif count >= 3:
            return {"armor": 50, "magic_resist": 50}
        elif count >= 2:
            return {"health_percent": 20}
        return {}


class ShadowIslesHandler(UniqueTraitHandler):
    """
    Shadow Isles: Collect souls when champions die.
    Breakpoints: 2/3/4/5 -> 1x/1.3x/1.6x/1.9x soul collection
    Also grants AD/AP% bonus: 18/20/22/25%
    """

    # Soul multiplier by breakpoint - Set 16 official values
    SOUL_MULTIPLIERS = {
        2: 1.0,
        3: 1.3,
        4: 1.6,
        5: 1.9,
    }

    # AD/AP% bonus by breakpoint
    AD_AP_BONUS = {
        2: 18,
        3: 20,
        4: 22,
        5: 25,
    }

    def __init__(self):
        self.souls = 0
        self.multiplier = 1.0
        self.ad_ap_percent = 0

    def on_death(self, is_ally: bool, count: int = 1) -> None:
        """Collect souls on any death."""
        self.souls += int(count * self.multiplier)

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        count = len(champions)
        if count >= 5:
            self.multiplier = self.SOUL_MULTIPLIERS[5]
            self.ad_ap_percent = self.AD_AP_BONUS[5]
        elif count >= 4:
            self.multiplier = self.SOUL_MULTIPLIERS[4]
            self.ad_ap_percent = self.AD_AP_BONUS[4]
        elif count >= 3:
            self.multiplier = self.SOUL_MULTIPLIERS[3]
            self.ad_ap_percent = self.AD_AP_BONUS[3]
        elif count >= 2:
            self.multiplier = self.SOUL_MULTIPLIERS[2]
            self.ad_ap_percent = self.AD_AP_BONUS[2]
        else:
            self.multiplier = 0.0
            self.ad_ap_percent = 0

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        # Base AD/AP% plus soul scaling
        return {
            "attack_damage_percent": self.ad_ap_percent,
            "ability_power_percent": self.ad_ap_percent,
            "bonus_ad_per_soul": self.souls,
            "bonus_ap_per_soul": self.souls,
        }


class ZaunHandler(UniqueTraitHandler):
    """
    Zaun: After 4 seconds, become Shimmer-Fused with durability and decaying AS.
    Breakpoints: 3/5/7
    - (3) Refresh after 4 seconds
    - (5) At 60% health, restore 25% health and become Shimmer-Fused instantly
    - (7) Bonuses increased 50%, refresh after 3 seconds
    """

    # Base shimmer bonuses - Set 16 values
    SHIMMER_BONUSES = {
        3: {"durability": 10, "attack_speed": 90},  # 90% decaying AS
        5: {"durability": 10, "attack_speed": 90, "heal_threshold": 60, "heal_amount": 25},
        7: {"durability": 15, "attack_speed": 135},  # 50% increased
    }

    def __init__(self):
        self.shimmer_active = False
        self.shimmer_duration = 4.0  # seconds

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        count = len(champions)
        self.shimmer_active = count >= 3
        if count >= 7:
            self.shimmer_duration = 3.0
        else:
            self.shimmer_duration = 4.0

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        count = len(champions)
        if count >= 7:
            return {"durability": 15, "attack_speed": 135}
        elif count >= 5:
            return {"durability": 10, "attack_speed": 90}
        elif count >= 3:
            return {"durability": 10, "attack_speed": 90}
        return {}


class DarkinHandler(UniqueTraitHandler):
    """
    Darkin: Gain omnivamp and deal damage when healing.
    """

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        count = len(champions)
        if count >= 1:
            return {"omnivamp": 15}
        return {}


class TargonHandler(UniqueTraitHandler):
    """
    Targon: Champions are naturally stronger.
    """

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        if len(champions) >= 1:
            return {
                "health": 150,
                "attack_damage": 15,
                "ability_power": 15,
                "armor": 10,
                "magic_resist": 10,
            }
        return {}


class IxtalHandler(UniqueTraitHandler):
    """
    Ixtal: Complete quests to earn Sunshards.
    """

    def __init__(self):
        self.sunshards = 0

    def on_quest_complete(self, reward: int = 1) -> None:
        """Earn sunshards on quest completion."""
        self.sunshards += reward

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        count = len(champions)
        if count >= 5:
            return {"player_heal": 2}
        return {}


class PiltoverHandler(UniqueTraitHandler):
    """
    Piltover: Build an Invention with modular upgrades.
    """

    def __init__(self):
        self.invention_level = 0
        self.upgrades = []

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        count = len(champions)
        if count >= 6:
            self.invention_level = 3
        elif count >= 4:
            self.invention_level = 2
        elif count >= 2:
            self.invention_level = 1

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        # Invention provides various bonuses based on upgrades
        return {}


# =========================================================================
# CLASS HANDLERS
# =========================================================================


class ArcanistHandler(UniqueTraitHandler):
    """
    Arcanist: All allies gain AP, Arcanists gain bonus AP.
    """

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        count = len(champions)
        if count >= 6:
            return {"ability_power": 40}
        elif count >= 4:
            return {"ability_power": 25}
        elif count >= 2:
            return {"ability_power": 18}
        return {}

    def get_arcanist_bonus(self, count: int) -> dict[str, float]:
        """Extra AP for Arcanist champions - Set 16 official values."""
        if count >= 6:
            return {"ability_power": 60}
        elif count >= 4:
            return {"ability_power": 40}
        elif count >= 2:
            return {"ability_power": 25}
        return {}


class BruiserHandler(UniqueTraitHandler):
    """
    Bruiser: Team gains 150 max HP, plus bonus health% at breakpoints.
    """

    BASE_HEALTH = 150  # All allies get this

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        count = len(champions)
        if count >= 6:
            return {"health": self.BASE_HEALTH, "health_percent": 65}
        elif count >= 4:
            return {"health": self.BASE_HEALTH, "health_percent": 45}
        elif count >= 2:
            return {"health": self.BASE_HEALTH, "health_percent": 25}
        return {}


class DefenderHandler(UniqueTraitHandler):
    """
    Defender: Team gains 12 Armor/MR, plus bonus at breakpoints.
    """

    BASE_ARMOR_MR = 12  # All allies get this

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        count = len(champions)
        base = self.BASE_ARMOR_MR
        if count >= 6:
            return {"armor": base + 80, "magic_resist": base + 80}
        elif count >= 4:
            return {"armor": base + 55, "magic_resist": base + 55}
        elif count >= 2:
            return {"armor": base + 30, "magic_resist": base + 30}
        return {}


class JuggernautHandler(UniqueTraitHandler):
    """
    Juggernaut: Durability scaling (higher when above 50% HP).
    Heal 5% max health when allied Juggernaut dies.
    Breakpoints: 2/4/6 -> 18-25% / 20-30% / 25-33% durability
    """

    # {breakpoint: (low_durability, high_durability)} - low when <50% HP, high when >50% HP
    DURABILITY_BREAKPOINTS = {
        2: (18, 25),
        4: (20, 30),
        6: (25, 33),
    }

    HEAL_PERCENT = 5  # 5% max HP heal on ally death

    def __init__(self):
        self.heal_percent = self.HEAL_PERCENT

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        count = len(champions)
        # Return average durability; actual scaling handled in combat
        if count >= 6:
            low, high = self.DURABILITY_BREAKPOINTS[6]
            return {"durability_low": low, "durability_high": high}
        elif count >= 4:
            low, high = self.DURABILITY_BREAKPOINTS[4]
            return {"durability_low": low, "durability_high": high}
        elif count >= 2:
            low, high = self.DURABILITY_BREAKPOINTS[2]
            return {"durability_low": low, "durability_high": high}
        return {}


class SlayerHandler(UniqueTraitHandler):
    """
    Slayer: Gain AD% and Omnivamp. AD increases up to 50% based on missing health.
    Breakpoints: 2/4/6 -> 22%/33%/44% AD, 10%/16%/20% Omnivamp
    """

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        count = len(champions)
        if count >= 6:
            return {"ad_percent": 44, "omnivamp": 20}
        elif count >= 4:
            return {"ad_percent": 33, "omnivamp": 16}  # Fixed: was 15
        elif count >= 2:
            return {"ad_percent": 22, "omnivamp": 10}
        return {}


class GunslingerHandler(UniqueTraitHandler):
    """
    Gunslinger: Gain AD%. Every 4th attack deals bonus damage.
    Breakpoints: 2/4 -> 20%/35% AD, 100/200 bonus damage every 4th attack
    """

    # {breakpoint: (ad_percent, fourth_attack_damage)}
    BREAKPOINTS = {
        2: (20, 100),
        4: (35, 200),
    }

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        count = len(champions)
        if count >= 4:
            ad, fourth = self.BREAKPOINTS[4]
            return {"ad_percent": ad, "fourth_attack_damage": fourth}
        elif count >= 2:
            ad, fourth = self.BREAKPOINTS[2]
            return {"ad_percent": ad, "fourth_attack_damage": fourth}
        return {}


class InvokerHandler(UniqueTraitHandler):
    """
    Invoker: Team gains mana regen. Invokers gain bonus mana.
    Breakpoints: 2/4 -> 1/2 mana regen, 25%/45% mana for Invokers
    """

    # Breakpoint values: {count: (mana_regen, invoker_mana_percent)}
    BREAKPOINTS = {
        2: (1, 25),
        4: (2, 45),
    }

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        """Return mana regen for all allies."""
        count = len(champions)
        if count >= 4:
            return {"mana_regen": self.BREAKPOINTS[4][0]}
        elif count >= 2:
            return {"mana_regen": self.BREAKPOINTS[2][0]}
        return {}

    def get_invoker_bonus(self, count: int) -> dict[str, float]:
        """Extra mana for Invoker champions (percentage of max mana)."""
        if count >= 4:
            return {"mana_percent": self.BREAKPOINTS[4][1]}
        elif count >= 2:
            return {"mana_percent": self.BREAKPOINTS[2][1]}
        return {}


class LongshotHandler(UniqueTraitHandler):
    """
    Longshot: Gain Damage Amp + bonus per hex distance.
    Breakpoints: 2/3/4/5 -> 18%+2%/hex, 24%+3%/hex, 28%+4%/hex, 32%+5%/hex
    """

    # {breakpoint: (base_damage_amp, per_hex_bonus)}
    BREAKPOINTS = {
        2: (18, 2),
        3: (24, 3),
        4: (28, 4),
        5: (32, 5),
    }

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        count = len(champions)
        if count >= 5:
            base, per_hex = self.BREAKPOINTS[5]
            return {"damage_amp": base, "damage_per_hex": per_hex}
        elif count >= 4:
            base, per_hex = self.BREAKPOINTS[4]
            return {"damage_amp": base, "damage_per_hex": per_hex}
        elif count >= 3:
            base, per_hex = self.BREAKPOINTS[3]
            return {"damage_amp": base, "damage_per_hex": per_hex}
        elif count >= 2:
            base, per_hex = self.BREAKPOINTS[2]
            return {"damage_amp": base, "damage_per_hex": per_hex}
        return {}


class VanquisherHandler(UniqueTraitHandler):
    """
    Vanquisher (Conqueror): Abilities can crit. Gain Crit Chance and Crit Damage.
    Breakpoints: 2/3/4/5 -> 15%/20%/25%/30% Crit Chance & Crit Damage
    """

    # {breakpoint: (crit_chance, crit_damage)}
    BREAKPOINTS = {
        2: (15, 15),
        3: (20, 20),
        4: (25, 25),
        5: (30, 30),
    }

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        count = len(champions)
        if count >= 5:
            cc, cd = self.BREAKPOINTS[5]
            return {"crit_chance": cc, "crit_damage": cd}
        elif count >= 4:
            cc, cd = self.BREAKPOINTS[4]
            return {"crit_chance": cc, "crit_damage": cd}
        elif count >= 3:
            cc, cd = self.BREAKPOINTS[3]
            return {"crit_chance": cc, "crit_damage": cd}
        elif count >= 2:
            cc, cd = self.BREAKPOINTS[2]
            return {"crit_chance": cc, "crit_damage": cd}
        return {}


class WardenHandler(UniqueTraitHandler):
    """
    Warden: Grant shields at 75% and 25% health thresholds.
    Shield is % of max HP: 2/3/4/5 -> 16%/20%/26%/33%
    """

    # Shield as % of max HP
    SHIELD_PERCENT = {
        2: 16,
        3: 20,
        4: 26,
        5: 33,
    }

    def __init__(self):
        self.shield_percent = 0

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        count = len(champions)
        if count >= 5:
            self.shield_percent = self.SHIELD_PERCENT[5]
        elif count >= 4:
            self.shield_percent = self.SHIELD_PERCENT[4]
        elif count >= 3:
            self.shield_percent = self.SHIELD_PERCENT[3]
        elif count >= 2:
            self.shield_percent = self.SHIELD_PERCENT[2]
        else:
            self.shield_percent = 0

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        return {"shield_percent": self.shield_percent}


class QuickstrikerHandler(UniqueTraitHandler):
    """
    Quickstriker: Team gains 15% AS. Quickstrikers gain bonus AS based on target's missing HP.
    Breakpoints: 2/3/4/5 -> 10-30% / 20-45% / 30-60% / 40-80% bonus AS
    """

    TEAM_AS = 15  # Team-wide AS bonus

    # {breakpoint: (min_as, max_as)} - based on target's missing HP
    BREAKPOINTS = {
        2: (10, 30),
        3: (20, 45),
        4: (30, 60),
        5: (40, 80),
    }

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        count = len(champions)
        if count >= 5:
            min_as, max_as = self.BREAKPOINTS[5]
            return {"team_attack_speed": self.TEAM_AS, "attack_speed_min": min_as, "attack_speed_max": max_as}
        elif count >= 4:
            min_as, max_as = self.BREAKPOINTS[4]
            return {"team_attack_speed": self.TEAM_AS, "attack_speed_min": min_as, "attack_speed_max": max_as}
        elif count >= 3:
            min_as, max_as = self.BREAKPOINTS[3]
            return {"team_attack_speed": self.TEAM_AS, "attack_speed_min": min_as, "attack_speed_max": max_as}
        elif count >= 2:
            min_as, max_as = self.BREAKPOINTS[2]
            return {"team_attack_speed": self.TEAM_AS, "attack_speed_min": min_as, "attack_speed_max": max_as}
        return {}


class DisruptorHandler(UniqueTraitHandler):
    """
    Disruptor: Abilities apply Dazzle. Deal bonus damage to Dazzled enemies.
    """

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        count = len(champions)
        if count >= 4:
            return {"dazzle_damage": 45}
        elif count >= 2:
            return {"dazzle_damage": 25}
        return {}


class HuntressHandler(UniqueTraitHandler):
    """
    Huntress: Leap to lowest health enemy. Gain bonus damage.
    Breakpoints: 2/3 -> Base leap / Increased leap damage
    """

    def apply(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> None:
        pass

    def get_bonus(
        self, game_state: Any, champions: list[ChampionInstance]
    ) -> dict[str, float]:
        count = len(champions)
        if count >= 3:
            return {"damage_amp": 25}  # Increased damage at 3
        elif count >= 2:
            return {"damage_amp": 15}  # Base damage at 2
        return {}


# Registry of unique trait handlers
UNIQUE_HANDLERS: dict[str, type[UniqueTraitHandler]] = {
    # Origins
    "demacia": DemaciaHandler,
    "ionia": IoniaHandler,
    "noxus": NoxusHandler,
    "void": VoidHandler,
    "yordle": YordleHandler,
    "bilgewater": BilgewaterHandler,
    "freljord": FreljordHandler,
    "shurima": ShurimaHandler,
    "shadow_isles": ShadowIslesHandler,
    "zaun": ZaunHandler,
    "darkin": DarkinHandler,
    "targon": TargonHandler,
    "ixtal": IxtalHandler,
    "piltover": PiltoverHandler,
    # Classes
    "arcanist": ArcanistHandler,
    "bruiser": BruiserHandler,
    "defender": DefenderHandler,
    "juggernaut": JuggernautHandler,
    "slayer": SlayerHandler,
    "gunslinger": GunslingerHandler,
    "invoker": InvokerHandler,
    "longshot": LongshotHandler,
    "vanquisher": VanquisherHandler,
    "warden": WardenHandler,
    "quickstriker": QuickstrikerHandler,
    "disruptor": DisruptorHandler,
    "huntress": HuntressHandler,
}


def get_handler(trait_id: str) -> Optional[UniqueTraitHandler]:
    """
    Get a handler instance for a trait.

    Args:
        trait_id: The trait ID

    Returns:
        Handler instance, or None if no special handler
    """
    handler_class = UNIQUE_HANDLERS.get(trait_id)
    if handler_class:
        return handler_class()
    return None
