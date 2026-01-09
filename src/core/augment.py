"""Augment System for TFT Set 16.

Manages augment selection rounds and augment effects.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
from pathlib import Path

from src.core.constants import AUGMENT_ROUNDS, AUGMENT_TIMER


class AugmentTier(Enum):
    """Rarity tiers for augments."""
    SILVER = "silver"      # Most common, basic effects
    GOLD = "gold"          # Mid-tier, moderate power
    PRISMATIC = "prismatic"  # Rarest, most powerful


class AugmentCategory(Enum):
    """Categories of augments."""
    ECONOMY = "economy"      # Gold/economy related
    COMBAT = "combat"        # Combat stats/effects
    TRAIT = "trait"          # Trait-specific
    ITEM = "item"            # Item-related
    UTILITY = "utility"      # General utility
    CHAMPION = "champion"    # Champion grants
    LEVELING = "leveling"    # XP/leveling related
    TEAM_UP = "team_up"      # Team-up augments (Set 16)


@dataclass
class Augment:
    """An augment that can be selected."""
    id: str
    name: str
    description: str
    tier: AugmentTier
    category: AugmentCategory
    # Effect parameters (used by game logic)
    effects: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"[{self.tier.value.upper()}] {self.name}"


@dataclass
class AugmentChoice:
    """A set of augment choices offered to a player."""
    stage: str
    options: list[Augment]
    timer_seconds: int
    selected: Optional[Augment] = None

    @property
    def is_decided(self) -> bool:
        return self.selected is not None


# ============================================================================
# AUGMENT DATABASE
# ============================================================================

AUGMENT_DATABASE: dict[str, Augment] = {}

# ============================================================================
# AUGMENT EFFECT PARAMETERS
# Maps augment_id -> effect parameters for game logic
# ============================================================================

AUGMENT_EFFECTS: dict[str, dict] = {
    # ==========================================================================
    # SILVER TIER AUGMENTS
    # ==========================================================================

    # Combat Hexes (Axiom)
    "air_axiom": {
        "hex_type": "air",
        "bonus_attack_speed": 0.20,
        "sunder_percent": 0.30,
        "sunder_duration": 5.0,
    },
    "earth_axiom": {
        "hex_type": "earth",
        "bonus_armor": 25,
        "stun_chance": 0.15,
        "stun_duration": 1.0,
    },
    "fire_axiom": {
        "hex_type": "fire",
        "bonus_ad": 15,
        "bonus_ap": 15,
        "apply_burn": True,
        "apply_wound": True,
    },
    "water_axiom": {
        "hex_type": "water",
        "mana_regen_per_sec": 5,
        "shred_percent": 0.20,
    },
    "wood_axiom": {
        "hex_type": "wood",
        "bonus_health": 200,
        "health_on_takedown": 50,
    },

    # Economy Silver
    "trade_sector": {"free_rerolls_per_round": 1},
    "called_shot": {"set_win_streak": 4, "instant_gold": 8},
    "crafted_crafting": {"rerolls_on_craft": 2},
    "efficient_shopper": {"carousel_duplicates": 4},
    "firesale": {"steal_shop_champion": True},
    "good_for_something_1": {"gold_on_itemless_death": 1},
    "lunch_money": {"gold_per_damage_dealt": 0.5},
    "on_a_roll": {"rerolls_on_star_up": 1},
    "patience_is_a_virtue": {"free_rerolls_if_no_buy": 2},
    "placebo": {"team_attack_speed": 0.10, "instant_gold": 6},
    "risky_moves": {"health_loss_per_combat": 2, "gold_per_combat": 3},
    "rolling_for_days_1": {"instant_rerolls": 10},
    "spoils_of_war_1": {"loot_drop_chance": 0.25},
    "table_scraps": {"carousel_unclaimed": True},

    # Combat Silver
    "featherweights": {"cost_1_2_attack_speed": 0.30, "cost_1_2_move_speed": 0.30},
    "first_aid_kit": {"heal_per_5_sec": 400},
    "partial_ascension": {"damage_bonus_after_15s": 0.30},
    "best_friends_1": {"paired_attack_speed": 0.15, "paired_armor": 20},
    "big_friend_1": {"largest_champ_health": 200, "largest_champ_durability": 0.10},
    "boxing_lessons": {"health_per_front_row": 100},
    "celestial_blessing_1": {"omnivamp": 0.12, "shield_conversion": 0.25},
    "chaotic_evolution": {"random_stats_on_star_up": True},
    "corrosion": {"enemy_front_armor_reduce": 20, "enemy_front_mr_reduce": 20},
    "exiles_1": {"isolated_shield_percent": 0.15},
    "focused_fire": {"stacking_ad": 3},
    "healing_orbs_1": {"heal_on_enemy_death": 200},
    "lineup": {"armor_per_front_row": 8, "mr_per_front_row": 8},
    "makeshift_armor_1": {"no_item_armor": 35, "no_item_mr": 35},
    "preparation_1": {"bench_stacking_ad": 3, "bench_stacking_ap": 3},
    "second_wind": {"heal_missing_hp_after_10s": 0.40},
    "stand_united": {"ad_per_trait": 3, "ap_per_trait": 3},
    "twin_guardians": {"front_pair_team_armor": 15, "front_pair_team_mr": 15},

    # Item Silver
    "pandoras_items": {"randomize_bench_items": True},
    "component_grab_bag": {"random_components": 3},
    "backup_bows": {"recurve_on_attack_threshold": [50, 100, 200]},
    "band_of_thieves_1": {"thiefs_gloves": 1},
    "carve_a_path": {"bf_sword": 1, "bf_on_damage_threshold": [5000, 15000, 30000]},
    "component_buffet": {"component_anvils_instead": True},
    "continuous_conjuration": {"rod_on_magic_damage": [3000, 10000, 25000]},
    "critical_success": {"gloves": 1, "gloves_on_crit_threshold": [30, 75, 150]},
    "eye_for_an_eye": {"component_on_ally_death": True, "max_components": 3},
    "flowing_tears": {"tear": 1, "tear_on_mana_threshold": [500, 1500, 4000]},
    "iron_assets": {"component_anvil": 1, "instant_gold": 5},
    "item_grab_bag": {"random_completed_items": 1},
    "over_encumbered": {"bench_slots_reduced": 3, "component_per_round": 1},
    "small_grab_bag": {"random_components": 2},

    # Champion Silver
    "artillery_barrage": {"grant_champion": "rumble", "enhanced": True},
    "caretakers_ally": {"grant_2cost_on_level": True},
    "leap_of_faith": {"grant_champion": "illaoi", "fighter_convert": True},
    "missed_connections": {"grant_all_1costs": True},
    "one_two_five": {"grant_1cost": 1, "grant_2cost": 1, "grant_5cost": 1, "random_component": 1},
    "ones_twos_three": {"grant_1cost": 2, "grant_2cost": 2, "grant_3cost": 1},
    "recombobulator": {"upgrade_board_cost": True},
    "restart_mission": {"replace_with_2star": True},
    "slice_of_life": {"grant_random_per_stage": True},
    "team_building": {"lesser_duplicator": 1, "duplicator_after_combats": 5},
    "teaming_up": {"random_component": 1, "grant_3cost": 2},

    # Trait Silver
    "branching_out": {"random_emblem": 1},
    "flexible": {"random_emblems": 2, "health_per_emblem": 100},

    # Utility Silver
    "tiny_titans": {"heal_per_round": 3, "can_exceed_100hp": True},
    "augmented_power": {"next_augment_tier_up": True},
    "pandoras_bench": {"randomize_bench_champions": True},
    "silver_destiny": {"random_silver_augment": True, "instant_gold": 3},
    "slightly_magic_roll": {"roll_dice": True},
    "titanic_titan": {"bonus_health": 25, "carousel_priority": True},

    # Leveling Silver
    "silver_spoon": {"instant_xp": 10},

    # ==========================================================================
    # GOLD TIER AUGMENTS
    # ==========================================================================

    # Economy Gold
    "hedge_fund": {"instant_gold": 20},
    "consistent_income": {"gold_per_round": 4, "no_interest": True},
    "advanced_loan": {"instant_gold": 20, "next_augment_tier_down": True},
    "calculated_loss": {"gold_on_loss": 3, "reroll_on_loss": 1},
    "commerce_core": {"instant_rerolls": 3, "free_rerolls_per_round": 1},
    "forward_thinking": {"lose_gold": 15, "regain_gold_after_combats": 5, "gold_bonus": 10},
    "gain_21_gold": {"instant_gold": 21},
    "hard_bargain": {"skip_carousel": True, "bonus_health": 10, "instant_gold": 12},
    "hustler": {"gold_per_combat": 2, "xp_per_combat": 2, "no_interest": True},
    "malicious_monetization": {"instant_gold": 8, "gold_on_enemy_kill": 1},
    "raining_gold": {"instant_gold": 10, "gold_per_round": 2},
    "savings_account": {"gold_on_interest_threshold": [10, 20, 30, 50]},
    "spoils_of_war_2": {"loot_drop_chance": 0.30},
    "trade_sector_gold": {"free_rerolls_per_round": 1, "instant_gold": 8},
    "treasure_hunt": {"chests_on_reroll_spending": True},
    "two_much_value": {"rerolls_per_2cost": 1},

    # Combat Gold
    "cybernetic_uplink": {"item_holder_health": 200, "item_holder_mana_per_sec": 3},
    "makeshift_armor": {"no_item_armor": 60, "no_item_mr": 60},
    "big_friend": {"largest_champ_health": 350, "largest_champ_as": 0.35},
    "thrill_of_hunt": {"heal_on_kill": 500},
    "arcane_viktor_y": {"stun_after_8s": 1.5, "stun_after_20s": 1.5},
    "ascension": {"damage_amp_after_delay": 0.35, "delay_seconds": 12},
    "best_friends_2": {"paired_attack_speed": 0.25, "paired_armor": 35},
    "bodyguard_training": {"armor_per_level": 5, "mr_per_level": 5},
    "bronze_for_life_1": {"damage_amp_per_bronze": 0.04},
    "celestial_blessing_2": {"omnivamp": 0.18, "shield_conversion": 0.35},
    "comeback_story": {"scaling_per_missing_hp": 0.01},
    "early_learnings": {"stacking_ad": 5, "stacking_ap": 5},
    "exiles_2": {"isolated_shield_percent": 0.25},
    "find_your_center": {"center_front_bonus_ad": 40, "center_front_bonus_hp": 400},
    "healing_orbs_2": {"heal_on_enemy_death": 350},
    "hefty_rolls": {"health_per_reroll": 20, "size_per_reroll": 0.02},
    "jeweled_lotus_1": {"team_crit_chance": 0.20},
    "know_your_enemy": {"see_opponent": True, "damage_amp_vs_opponent": 0.12},
    "last_second_save": {"heal_at_threshold": 0.30, "heal_amount": 400},
    "little_buddies": {"high_cost_bonus_per_low_cost": 0.08},
    "makeshift_armor_2": {"no_item_armor": 60, "no_item_mr": 60},
    "mess_hall": {"team_attack_speed": 0.15, "projectile_attacks": True},
    "plot_armor": {"base_armor": 20, "base_mr": 20, "low_hp_bonus": 0.50},
    "preparation_2": {"bench_stacking_ad": 6, "bench_stacking_ap": 6},
    "spirit_link": {"heal_based_on_hp_gap": True},
    "tons_of_stats": {"bonus_ad": 12, "bonus_ap": 12, "bonus_armor": 15, "bonus_mr": 15, "bonus_as": 0.10},
    "warlords_honor": {"stacking_ad": 4, "stacking_ap": 4, "per_combat": True},
    "pair_of_fours": {"dual_4cost_bonus_ad": 30, "dual_4cost_bonus_ap": 30, "dual_4cost_bonus_hp": 400},
    "precision_and_grace": {"dash_on_takedown": True, "as_on_takedown": 0.15},

    # Item Gold
    "item_grab_bag": {"random_completed_items": 1, "random_components": 2},
    "big_grab_bag": {"random_components": 3, "instant_gold": 5, "reforger": 1},
    "blood_offering": {"bloodthirster": 1, "team_shield": 200, "health_loss": 3},
    "care_package": {"care_package_per_stage": True},
    "crowns_will": {"rod": 1, "team_ap": 10},
    "cry_me_a_river": {"tear": 1, "mana_regen": 5},
    "deadlier_blades": {"deathblade": 1, "stacking_ad": 3},
    "deadlier_caps": {"deathcap": 1, "stacking_ap": 5},
    "exclusive_customization": {"lucky_item_chest": 1},
    "feed_the_flames": {"sunfire": 1, "omnivamp_vs_burning": 0.15},
    "heart_of_steel": {"steadfast_heart": 1, "stacking_health": 50},
    "heavy_is_the_crown": {"crown_of_demacia": 1, "lose_on_crown_loss": True},
    "high_voltage": {"ionic_spark": 1, "extended_radius": True},
    "indiscriminate_killer": {"giant_slayer": 1, "damage_amp_all": 0.12},
    "infinity_protection": {"instant_gold": 8, "infinity_force_later": True},
    "maces_will": {"gloves": 1, "team_as": 0.08, "team_crit": 0.10},
    "pandoras_items_2": {"randomize_bench_items": True, "random_components": 2},
    "portable_forge": {"artifact_choice": 4},
    "prizefighter": {"random_components": 2, "component_per_win": 1},
    "promised_protection": {"protectors_vow": 1, "mana_benefit": True},
    "pyromaniac": {"red_buff": 1, "burn_damage_amp": 0.25},
    "replication": {"choose_component": True, "copies_for_rounds": 2},
    "salvage_bin": {"random_completed_items": 1, "component_later": 1, "break_items_on_sell": True},
    "seraphims_staff": {"archangels": 1, "bonus_mana": 20},
    "slammin": {"random_component": 1, "xp_if_bench_no_items": 4},
    "solo_plate": {"gargoyle": 1, "solo_row_bonus": True},
    "spears_will": {"bf_sword": 1, "team_ad": 8, "team_mana": 10},
    "speedy_double_kill": {"guinsoos": 1, "gold_on_elimination": 2},
    "spirit_of_redemption": {"spirit_visage": 1, "area_healing": True},
    "staffsmith": {"rod_completed_items": True},
    "swordsmith": {"sword_completed_items": True},
    "the_golden_dragon": {"moguls_mail": 1, "oversized": True},
    "unsealed_from_steel": {"darkin_choice": 4},
    "urf": {"spatula": 1, "spatula_holder_bonus": True},
    "urfs_gambit": {"spatula_or_pan_on_result": True},
    "woven_magic": {"random_component": 1, "component_per_mana_spent": 500},

    # Champion Gold
    "three_threes": {"grant_2star_3cost": 2, "instant_gold": 5},
    "aura_farming": {"grant_2star_5cost": 1, "available_at": "4-3"},
    "birthday_reunion": {"grant_2cost": 1, "thiefs_gloves_at_6": True, "grant_5cost_at_9": True},
    "bringer_of_ruin": {"grant_atakhan": True, "stacking_on_takedown": True},
    "cluttered_mind": {"grant_1cost": 3, "xp_if_bench_full": 3},
    "delayed_start": {"replace_with_2star": True, "shop_disabled_rounds": 2},
    "double_trouble_gold": {"double_champion_bonus": True, "reward_on_3star": True},
    "duo_queue": {"grant_5cost": 2, "random_components": 2},
    "golemify": {"merge_into_golem": True},
    "heroic_grab_bag": {"lesser_duplicator": 2, "instant_gold": 6},
    "max_build": {"duplicator": 1, "instant_rerolls": 3, "repeat_per_stage": True},
    "pilfer": {"copy_first_kill": True, "thiefs_gloves": 1},
    "poison_pals": {"grant_singed": True, "grant_teemo": True, "synergy": True},
    "reinforcement": {"next_4cost_2star": True},
    "solo_leveling": {"solo_champion_mode": True, "massive_stats": True, "combats": 5},
    "stars_are_born": {"first_1cost_2star": True, "first_2cost_2star": True},
    "starter_kit": {"grant_4cost": 1, "grant_matching_2cost": 1},
    "the_ruined_king": {"grant_2star_viego": True, "shadow_isles_synergy": True},
    "trials_of_twilight": {"grant_xin_zhao": True, "unlock_zaahen_on_3star": True},
    "trifecta_1": {"grant_3cost": 3, "combat_bonus": True},
    "two_trick": {"grant_2star_2cost": 1, "grant_1cost": 2},
    "walk_the_true_path_1": {"duplicators": 2, "grant_champions": True, "ionia_upgrade": True},
    "warpath": {"grant_2star_2cost": 1, "high_cost_on_damage": True},
    "worth_the_wait": {"grant_champion": 1, "copies_per_round": 1},

    # Trait Gold
    "branching_out_plus": {"random_emblem": 1, "enhanced": True},
    "chaos_magic": {"arcanist_mana_regen": 5, "magic_effects": True},
    "darkwills_invasion": {"team_damage_amp": 0.08, "noxus_bonus": 0.04},
    "defense_of_the_placidium": {"team_durability": 0.10, "ionia_bonus": 0.05},
    "demacia_forever": {"demacia_stacking_per_rally": True},
    "evolve_and_overcome": {"xp_per_void_takedown": 1},
    "hexgate_travel": {"piltover_on_invention": True},
    "ixtal_expeditionist": {"grant_ixtal_champions": 2, "bonus_health": 15},
    "legion_of_threes": {"emblem": 1, "3cost_bonus": True, "emblem_holder_bonus": True},
    "lifting_competition": {"bruiser_rewards_per_hp": True},
    "silcos_revenge": {"zaunite_explode_on_death": True, "grant_zaunites": 2},
    "spreading_roots": {"random_emblems": 2, "instant_gold": 6},
    "wild_growth": {"3star_yordle_massive": True},

    # Leveling Gold
    "leveling_up": {"instant_xp": 8},
    "clear_mind": {"xp_if_bench_empty": 4},
    "epic_rolldown": {"rerolls_at_level_8": 20},
    "epoch": {"instant_xp": 6, "instant_rerolls": 2, "xp_per_stage": 4, "rerolls_per_stage": 2},
    "epoch_plus": {"instant_xp": 10, "instant_rerolls": 4, "xp_per_stage": 6, "rerolls_per_stage": 3},
    "explosive_growth": {"instant_xp": 6, "xp_per_round": 1},
    "explosive_growth_plus": {"instant_xp": 10, "xp_per_round": 2},
    "late_game_scaling": {"xp_per_round": 2, "5cost_bonus": True},
    "patient_study": {"xp_on_win": 3, "xp_on_loss": 2},

    # Utility Gold
    "a_magic_roll": {"roll_3_dice": True},
    "corona": {"tacticians_crown": 1, "enhanced": True},
    "gold_destiny": {"random_gold_augment": True, "instant_gold": 5},
    "gold_destiny_plus": {"random_gold_augment": True, "instant_gold": 10},
    "indecision_1": {"new_gold_augment_after_rounds": 3},
    "no_scout_no_pivot": {"lock_fielded_champions": True, "team_stats": True},

    # ==========================================================================
    # PRISMATIC TIER AUGMENTS
    # ==========================================================================

    # Economy Prismatic
    "golden_ticket": {"instant_gold": 25, "gold_per_level_up": 10},
    "going_long": {"no_interest": True, "gold_per_round": 5, "xp_per_round": 3},
    "hedge_fund_prismatic": {"instant_gold": 35, "interest_cap_increase": 5},
    "invested_plus": {"gold_per_gold_above_threshold": True, "rerolls_bonus": True},
    "money_monsoon": {"gold_per_round": 4},
    "prismatic_ticket": {"free_reroll_chance": 0.50},
    "shopping_spree": {"rerolls_per_level": 5},
    "spoils_of_war_3": {"loot_drop_chance": 0.45},

    # Combat Prismatic
    "urf_overtime": {"overtime_attack_speed": 1.0, "overtime_cast_speed": 2.0},
    "double_trouble": {"double_ad": 55, "double_ap": 55, "double_health": 330},
    "bronze_for_life_2": {"damage_amp_per_bronze": 0.06, "durability_per_bronze": 0.04},
    "celestial_blessing_3": {"omnivamp": 0.25, "shield_conversion": 0.50},
    "hold_the_line": {"back_row_bonus_per_front": True},
    "jeweled_lotus_2": {"team_crit_chance": 0.35, "crit_damage_bonus": 0.15},
    "soul_awakening": {"stacking_damage_after_combat_start": True},
    "the_axiomata": {"all_hex_augments": True, "future_rewards": True},
    "tiny_but_deadly": {"team_shrink": True, "speed_bonus": 0.30},

    # Item Prismatic
    "radiant_relics": {"radiant_item_choice": 5, "remover": 1},
    "living_forge": {"artifact_anvil": 1, "artifact_later": True},
    "lucky_gloves": {"perfect_thiefs_gloves": True},
    "band_of_thieves_2": {"thiefs_gloves": 2, "thiefs_gloves_after_combats": 5},
    "belt_overflow": {"giant_belts": 3, "bonus_health": 150},
    "binary_airdrop": {"armed_random_items": True, "random_components": 2},
    "buried_treasures": {"component_per_round": 1},
    "component_heist": {"all_components_after_combat": True},
    "forged_in_strength": {"artifacts": 2, "more_below_hp": True},
    "luxury_subscription": {"package_per_stage": True},
    "min_max": {"golden_remover": 1, "random_components": 3},
    "one_buff_two_buff": {"red_buff": 1, "blue_buff": 1, "duplicator": 1},
    "pandoras_items_3": {"randomize_bench_items": True, "radiant_item": 1},
    "radiant_rascal": {"rascals_gloves": 1, "rotating_radiant": True},
    "retribution": {"hands_of_justice": 1, "ability_crits": True},
    "shimmerscale_essence": {"moguls_mail": 1, "gamblers_blade_later": True},
    "sweet_treats": {"artifact_anvil": 1, "health_per_item": 50},
    "sword_overflow": {"bf_swords": 3, "bonus_as": 0.15},
    "wand_overflow": {"rods": 3, "bonus_as": 0.15},

    # Champion Prismatic
    "birthday_present": {"grant_2star_on_level_up": True},
    "chosen_wolves": {"grant_ambessa": True, "grant_kindred": True, "synergy": True},
    "construct_a_companion": {"next_1cost_3star": True},
    "dragonguards": {"grant_shyvana": True, "grant_jarvan": True, "synergy": True},
    "just_hit": {"duplicator": 1, "instant_rerolls": 10, "instant_gold": 10},
    "trifecta_2": {"grant_3cost": 4, "team_as": 0.20, "enhanced_bonus": True},
    "worth_the_wait_2": {"grant_2cost_per_round": 1},

    # Trait Prismatic
    "ancient_archives": {"tome_of_traits": 2},
    "flexible_prismatic": {"random_emblems": 3, "health_per_emblem": 150},
    "hard_commit": {"emblem": 1, "matching_champions_per_stage": True},
    "tacticals_kitchen": {"emblem": 1, "tacticians_cape_later": True},
    "the_trait_tree": {"emblems": 3, "instant_gold": 8},
    "the_world_runes": {"region_emblems": True, "reward_on_threshold": True},
    "we_stick_together": {"emblem": 1, "anvil": 1, "emblem_holder_as": 0.20},

    # Leveling Prismatic
    "cruel_pact": {"xp_costs_health": True, "xp_health_cost": 4},
    "growth_mindset": {"instant_xp": 12, "xp_cost_reduction": 1},
    "level_up": {"bonus_xp_on_purchase": 2},
    "upward_mobility": {"xp_cost_reduction": 1, "bonus_per_level": True},
    "win_out": {"level_10_at_9": True, "instant_rerolls": 5, "instant_xp": 8},

    # Utility Prismatic
    "call_to_chaos": {"powerful_random_reward": True},
    "coronation": {"tacticians_crown": 1, "enhanced_bonuses": True},
    "cursed_crown": {"team_size": 1, "double_damage_on_loss": True},
    "expected_unexpectedness": {"dice_now_and_per_stage": True},
    "indecision_2": {"new_augment_after_rounds": 3, "instant_gold": 8},
    "nine_lives": {"set_health_9": True, "only_lose_1_hp": True},
    "prismatic_destiny": {"random_prismatic_augment": True, "instant_gold": 10},
    "the_golden_egg": {"egg_hatches_after_turns": 10},
}


def _load_augments_from_json() -> None:
    """Load augments from JSON data file and register them."""
    json_path = Path(__file__).parent.parent.parent / "data" / "augments" / "set16_augments.json"

    if not json_path.exists():
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tier_map = {
        "silver": AugmentTier.SILVER,
        "gold": AugmentTier.GOLD,
        "prismatic": AugmentTier.PRISMATIC,
    }

    category_map = {
        "combat": AugmentCategory.COMBAT,
        "economy": AugmentCategory.ECONOMY,
        "item": AugmentCategory.ITEM,
        "trait": AugmentCategory.TRAIT,
        "utility": AugmentCategory.UTILITY,
        "champion": AugmentCategory.CHAMPION,
        "leveling": AugmentCategory.LEVELING,
    }

    for tier_name, augments in data.get("augments", {}).items():
        tier = tier_map.get(tier_name, AugmentTier.SILVER)

        for aug_data in augments:
            aug_id = aug_data["id"]
            category_str = aug_data.get("category", "utility")
            category = category_map.get(category_str, AugmentCategory.UTILITY)

            # Get effects from our mapping, or empty dict if not mapped
            effects = AUGMENT_EFFECTS.get(aug_id, {})

            AUGMENT_DATABASE[aug_id] = Augment(
                id=aug_id,
                name=aug_data["name"],
                description=aug_data.get("effect", ""),
                tier=tier,
                category=category,
                effects=effects,
            )


# Load augments on module import
_load_augments_from_json()


# Probability distribution for augment tiers across the 3 choices
# Format: (first_choice_tier, second_choice_tier, third_choice_tier): probability
AUGMENT_TIER_DISTRIBUTION = {
    (AugmentTier.SILVER, AugmentTier.SILVER, AugmentTier.SILVER): 0.15,
    (AugmentTier.SILVER, AugmentTier.SILVER, AugmentTier.GOLD): 0.25,
    (AugmentTier.SILVER, AugmentTier.GOLD, AugmentTier.GOLD): 0.20,
    (AugmentTier.GOLD, AugmentTier.GOLD, AugmentTier.GOLD): 0.15,
    (AugmentTier.SILVER, AugmentTier.GOLD, AugmentTier.PRISMATIC): 0.10,
    (AugmentTier.GOLD, AugmentTier.GOLD, AugmentTier.PRISMATIC): 0.08,
    (AugmentTier.SILVER, AugmentTier.PRISMATIC, AugmentTier.PRISMATIC): 0.04,
    (AugmentTier.PRISMATIC, AugmentTier.PRISMATIC, AugmentTier.PRISMATIC): 0.03,
}


class AugmentSystem:
    """Manages augment selection and effects."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize augment system.

        Args:
            seed: Random seed for reproducible results.
        """
        self.rng = random.Random(seed)
        self._player_augments: dict[int, list[Augment]] = {}
        self._tier_sequence: Optional[tuple] = None
        self._current_choice_index: int = 0

    def is_augment_round(self, stage: str) -> bool:
        """Check if a stage is an augment selection round."""
        return stage in AUGMENT_ROUNDS

    def get_augment_timer(self, stage: str) -> int:
        """Get the timer in seconds for an augment choice."""
        return AUGMENT_TIMER.get(stage, 58)

    def get_augment_by_id(self, augment_id: str) -> Optional[Augment]:
        """Get an augment by its ID."""
        return AUGMENT_DATABASE.get(augment_id)

    def get_augments_by_tier(self, tier: AugmentTier) -> list[Augment]:
        """Get all augments of a specific tier."""
        return [a for a in AUGMENT_DATABASE.values() if a.tier == tier]

    def get_augments_by_category(self, category: AugmentCategory) -> list[Augment]:
        """Get all augments of a specific category."""
        return [a for a in AUGMENT_DATABASE.values() if a.category == category]

    def _determine_tier_sequence(self) -> tuple[AugmentTier, AugmentTier, AugmentTier]:
        """Determine the tier sequence for all 3 augment choices in a game."""
        if self._tier_sequence is not None:
            return self._tier_sequence

        sequences = list(AUGMENT_TIER_DISTRIBUTION.keys())
        weights = list(AUGMENT_TIER_DISTRIBUTION.values())
        self._tier_sequence = self.rng.choices(sequences, weights=weights, k=1)[0]
        return self._tier_sequence

    def generate_augment_choices(
        self,
        stage: str,
        player_id: int,
        excluded_augments: Optional[list[str]] = None,
    ) -> AugmentChoice:
        """
        Generate 3 augment options for a player.

        Args:
            stage: Current stage (2-1, 3-2, or 4-2).
            player_id: The player to generate choices for.
            excluded_augments: List of augment IDs to exclude (already owned).

        Returns:
            AugmentChoice with 3 options.
        """
        tier_sequence = self._determine_tier_sequence()

        # Determine which choice this is (0, 1, or 2)
        choice_index = AUGMENT_ROUNDS.index(stage) if stage in AUGMENT_ROUNDS else 0
        target_tier = tier_sequence[choice_index]

        # Get available augments of the target tier
        excluded = set(excluded_augments or [])
        player_existing = self._player_augments.get(player_id, [])
        for aug in player_existing:
            excluded.add(aug.id)

        available = [
            a for a in self.get_augments_by_tier(target_tier)
            if a.id not in excluded
        ]

        # Cannot offer 3 economy augments in same round
        economy_count = 0
        selected_options = []

        self.rng.shuffle(available)
        for augment in available:
            if len(selected_options) >= 3:
                break

            if augment.category == AugmentCategory.ECONOMY:
                if economy_count >= 2:
                    continue
                economy_count += 1

            selected_options.append(augment)

        # If we don't have 3, fill with other tiers
        if len(selected_options) < 3:
            other_tiers = [t for t in AugmentTier if t != target_tier]
            for tier in other_tiers:
                tier_augments = [
                    a for a in self.get_augments_by_tier(tier)
                    if a.id not in excluded and a.id not in [s.id for s in selected_options]
                ]
                self.rng.shuffle(tier_augments)
                for aug in tier_augments:
                    if len(selected_options) >= 3:
                        break
                    if aug.category == AugmentCategory.ECONOMY and economy_count >= 2:
                        continue
                    selected_options.append(aug)
                    if aug.category == AugmentCategory.ECONOMY:
                        economy_count += 1

        return AugmentChoice(
            stage=stage,
            options=selected_options[:3],
            timer_seconds=self.get_augment_timer(stage),
        )

    def select_augment(
        self,
        choice: AugmentChoice,
        player_id: int,
        augment_index: int,
    ) -> Optional[Augment]:
        """
        Record a player's augment selection.

        Args:
            choice: The AugmentChoice to select from.
            player_id: The player making the selection.
            augment_index: Index of the selected augment (0-2).

        Returns:
            The selected Augment, or None if invalid.
        """
        if augment_index < 0 or augment_index >= len(choice.options):
            return None

        selected = choice.options[augment_index]
        choice.selected = selected

        # Track player's augments
        if player_id not in self._player_augments:
            self._player_augments[player_id] = []
        self._player_augments[player_id].append(selected)

        return selected

    def get_player_augments(self, player_id: int) -> list[Augment]:
        """Get all augments a player has selected."""
        return self._player_augments.get(player_id, [])

    def reset_game(self) -> None:
        """Reset augment system for a new game."""
        self._player_augments.clear()
        self._tier_sequence = None
        self._current_choice_index = 0

    def apply_augment_effects(
        self,
        augment: Augment,
        player_state: any,  # Would be PlayerState in actual use
    ) -> None:
        """
        Apply an augment's effects to a player.

        This is a placeholder for the actual effect application logic.
        Each effect would be implemented based on the game state structure.

        Args:
            augment: The augment to apply.
            player_state: The player's state to modify.
        """
        effects = augment.effects

        # Example effect applications (would need actual player_state integration)
        if "instant_gold" in effects:
            # player_state.gold += effects["instant_gold"]
            pass

        if "instant_xp" in effects:
            # player_state.add_xp(effects["instant_xp"])
            pass

        if "random_components" in effects:
            # Generate random component items
            pass

        # Additional effects would be implemented similarly


# Singleton instance
_augment_system: Optional[AugmentSystem] = None


def get_augment_system(seed: Optional[int] = None) -> AugmentSystem:
    """Get or create the augment system singleton."""
    global _augment_system
    if _augment_system is None or seed is not None:
        _augment_system = AugmentSystem(seed)
    return _augment_system
