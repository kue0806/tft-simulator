"""Champion Abilities for TFT Set 16.

Defines all champion abilities with their damage, targeting, and effects.
Based on TFT Set 16: Magic n' Mayhem data.
"""

from typing import Dict, List, Optional, Callable, TYPE_CHECKING
from .ability import (
    AbilityData,
    AbilityTargetType,
    AbilityResult,
    AbilitySystem,
)

if TYPE_CHECKING:
    from .combat_unit import CombatUnit


# =============================================================================
# 1-COST CHAMPION ABILITIES
# =============================================================================

ANIVIA_ABILITY = AbilityData(
    ability_id="anivia_frostbite",
    name="Frostbite",
    description="Anivia fires an ice shard at target dealing magic damage. If target is Chilled, the damage Critically Strikes.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.5,
    base_damage=[325, 455, 650],
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={"crit_if_chilled": True},
)

BLITZCRANK_ABILITY = AbilityData(
    ability_id="blitzcrank_static_field",
    name="Static Field",
    description="Blitzcrank gains a shield for 4 seconds and deals magic damage to all enemies within 1 hex.",
    target_type=AbilityTargetType.AOE_SELF,
    aoe_radius=1,
    cast_time=0.5,
    base_damage=[56, 84, 126],  # Scales with MR
    damage_type="magical",
    damage_scaling=1.0,
    base_shield=[400, 480, 600],
    shield_duration=4.0,
)

BRIAR_ABILITY = AbilityData(
    ability_id="briar_blood_frenzy",
    name="Blood Frenzy",
    description="Briar leaps to the furthest enemy within 2 hexes. For 4 seconds, gains increased Move Speed, 25% AD, and 300% decaying Attack Speed.",
    target_type=AbilityTargetType.FARTHEST_ENEMY,
    cast_time=0.25,
    base_damage=[0, 0, 0],
    custom_data={
        "attack_speed_bonus": [3.0, 3.0, 3.0],  # 300% decaying AS
        "ad_bonus_percent": 0.25,  # 25% AD
        "duration": 4.0,
        "leap_range": 2,
    },
)

CAITLYN_ABILITY = AbilityData(
    ability_id="caitlyn_ace_in_the_hole",
    name="Ace in the Hole",
    description="Fire a bullet at the farthest enemy dealing physical damage. If they die, the bullet ricochets dealing excess damage to the next enemy.",
    target_type=AbilityTargetType.FARTHEST_ENEMY,
    cast_time=1.0,
    base_damage=[515, 775, 1205],
    damage_type="physical",
    damage_scaling=1.0,  # 40/60/100 AP scaling
    can_crit=True,
    custom_data={"ricochets_on_kill": True},
)

ILLAOI_ABILITY = AbilityData(
    ability_id="illaoi_tentacle_smash",
    name="Tentacle Smash",
    description="Illaoi restores health (200/240/280 + 7% HP) and slams a tentacle dealing physical damage in a line.",
    target_type=AbilityTargetType.LINE,
    cast_time=0.5,
    base_damage=[180, 270, 405],
    damage_type="physical",
    damage_scaling=1.0,
    base_healing=[200, 240, 280],
    healing_scaling=1.0,  # +7% HP scaling
    custom_data={"heal_hp_percent": 0.07},
)

JARVAN_IV_ABILITY = AbilityData(
    ability_id="jarvan_demacian_standard",
    name="Demacian Standard",
    description="Jarvan IV gains a shield for 4 seconds. Grant all allies Attack Speed for 4 seconds.",
    target_type=AbilityTargetType.ALL_ALLIES,
    cast_time=0.5,
    base_damage=[0, 0, 0],
    base_shield=[350, 425, 500],
    shield_duration=4.0,
    custom_data={
        "team_attack_speed": [0.15, 0.20, 0.33],  # 15/20/33% AS buff
        "buff_duration": 4.0,
    },
)

JHIN_ABILITY = AbilityData(
    ability_id="jhin_curtain_call",
    name="Curtain Call",
    description="For the next 4 attacks, gain infinite range and replace attacks with cannon shots dealing physical damage. The 4th shot deals 144% more damage.",
    target_type=AbilityTargetType.FARTHEST_ENEMY,
    cast_time=0.25,
    base_damage=[140, 212, 314],  # Base damage per shot
    damage_type="physical",
    damage_scaling=1.0,
    can_crit=True,
    custom_data={
        "num_shots": 4,
        "fourth_shot_bonus": 1.44,  # 4th shot deals 144% more damage
    },
)

KOGMAW_ABILITY = AbilityData(
    ability_id="kogmaw_caustic_spittle",
    name="Caustic Spittle",
    description="Kog'Maw spits at target dealing magic damage and reducing their armor and MR. Adjacent enemies take 50% damage and shred.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    aoe_radius=1,
    cast_time=0.35,
    base_damage=[140, 200, 300],
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "armor_shred": [8, 10, 15],
        "mr_shred": [8, 10, 15],
        "adjacent_damage_percent": 0.50,
        "adjacent_shred_percent": 0.50,
    },
)

LULU_ABILITY = AbilityData(
    ability_id="lulu_whimsy",
    name="Whimsy",
    description="Lulu deals magic damage to a target and forces them to dance for 2 seconds. The projectile bounces to the nearest enemy for reduced damage.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.5,
    base_damage=[285, 425, 635],
    damage_type="magical",
    damage_scaling=1.0,
    applies_cc=True,
    cc_type="polymorph",  # Dance/Polymorph
    cc_duration=[2.0, 2.0, 2.0],
    custom_data={
        "bounce_damage": [120, 180, 270],  # Secondary target damage
    },
)

QIYANA_ABILITY = AbilityData(
    ability_id="qiyana_clear_the_brush",
    name="Clear The Brush",
    description="Qiyana dashes to a nearby hex and swings forward, dealing physical damage to enemies in a 2-hex line.",
    target_type=AbilityTargetType.LINE,
    cast_time=0.25,
    base_damage=[140, 210, 315],
    damage_type="physical",
    damage_scaling=1.0,
    custom_data={"ad_ap_scaling": [20, 30, 45]},
)

RUMBLE_ABILITY = AbilityData(
    ability_id="rumble_junkyard_titan",
    name="Junkyard Titan",
    description="Rumble gains a shield for 4 seconds and fires a burst of flames dealing magic damage in a cone.",
    target_type=AbilityTargetType.CONE,
    cast_time=0.25,
    base_damage=[72, 108, 162],
    damage_type="magical",
    damage_scaling=1.0,
    base_shield=[350, 430, 550],
    shield_duration=4.0,
)

SHEN_ABILITY = AbilityData(
    ability_id="shen_stand_united",
    name="Stand United",
    description="Shen grants a shield to himself and a nearby damaged ally for 4 seconds.",
    target_type=AbilityTargetType.LOWEST_HP_ALLY,
    cast_time=0.5,
    base_damage=[0, 0, 0],
    base_shield=[250, 325, 425],
    shield_duration=4.0,
)

SONA_ABILITY = AbilityData(
    ability_id="sona_power_chord",
    name="Power Chord",
    description="Sona deals magic damage to 2 nearby enemies and restores health to the lowest HP ally.",
    target_type=AbilityTargetType.AOE_ENEMY,
    aoe_radius=2,
    cast_time=0.5,
    base_damage=[130, 195, 300],
    damage_type="magical",
    damage_scaling=1.0,
    base_healing=[40, 50, 80],
    healing_scaling=1.0,
    custom_data={"num_targets": 2},
)

VIEGO_ABILITY = AbilityData(
    ability_id="viego_blade_of_the_ruined_king",
    name="Blade of the Ruined King",
    description="Passive: Viego's attacks deal stacking bonus magic damage. Active: Viego stabs his target dealing physical damage.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.25,
    base_damage=[55, 85, 125],
    damage_type="physical",
    damage_scaling=1.0,
    custom_data={
        "stacking_magic_damage": [24, 36, 54],  # Bonus magic damage per attack (stacks)
    },
)


# =============================================================================
# 2-COST CHAMPION ABILITIES
# =============================================================================

APHELIOS_ABILITY = AbilityData(
    ability_id="aphelios_moonshot",
    name="Moonshot",
    description="Aphelios fires a piercing shot that damages all enemies in a line.",
    target_type=AbilityTargetType.LINE,
    cast_time=0.35,
    base_damage=[225, 340, 525],
    damage_type="physical",
    damage_scaling=0.0,
    can_crit=True,
)

ASHE_ABILITY = AbilityData(
    ability_id="ashe_true_ice_arrow",
    name="True Ice Arrow",
    description="Fire an arrow dealing physical damage to target and adjacent enemies. Low-health enemies take fixed damage. Applies chill for 3 seconds.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    aoe_radius=1,
    cast_time=0.5,
    base_damage=[140, 210, 325],
    damage_type="physical",
    damage_scaling=1.0,
    custom_data={
        "adjacent_damage": [46, 69, 107],
        "chill_percent": 0.30,
        "chill_duration": 3.0,
    },
)

BARD_ABILITY = AbilityData(
    ability_id="bard_travelers_call",
    name="Traveler's Call",
    description="Bard launches 6 spirits at enemies dealing magic damage. Gain 1 bonus spirit per 3-star unit.",
    target_type=AbilityTargetType.MULTI_ENEMY,
    cast_time=0.5,
    base_damage=[115, 155, 220],  # Per spirit
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "base_spirits": 6,
        "bonus_spirit_per_3star": 1,
    },
)

CHOGATH_ABILITY = AbilityData(
    ability_id="chogath_rupture",
    name="Rupture",
    description="Cho'Gath ruptures the ground, dealing magic damage and knocking up enemies.",
    target_type=AbilityTargetType.AOE_ENEMY,
    aoe_radius=2,
    cast_time=0.75,
    base_damage=[250, 375, 600],
    damage_type="magical",
    damage_scaling=1.0,
    applies_cc=True,
    cc_type="knockup",
    cc_duration=[1.25, 1.5, 2.0],
)

EKKO_ABILITY = AbilityData(
    ability_id="ekko_parallel_convergence",
    name="Parallel Convergence",
    description="Create a hex zone for 3 seconds. Next 3 attacks deal bonus magic damage. Zone expires dealing damage plus 10% of damage taken during duration.",
    target_type=AbilityTargetType.AOE_ENEMY,
    aoe_radius=1,
    cast_time=0.5,
    base_damage=[130, 195, 295],  # Zone expire damage
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "bonus_damage_per_attack": [150, 225, 340],
        "zone_duration": 3.0,
        "empowered_attacks": 3,
        "damage_taken_percent": 0.10,
    },
)

GRAVES_ABILITY = AbilityData(
    ability_id="graves_collateral_damage",
    name="Collateral Damage",
    description="Passive: Attacks fire in a cone dealing 45% damage. Active: Fire an explosive shell dealing physical damage to enemies within 1 hex.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    aoe_radius=1,
    cast_time=0.35,
    base_damage=[200, 300, 450],
    damage_type="physical",
    damage_scaling=1.0,
    custom_data={
        "passive_cone_damage_percent": 0.45,
    },
)

NEEKO_ABILITY = AbilityData(
    ability_id="neeko_pop_blossom",
    name="Pop Blossom",
    description="Neeko leaps to target, gains shield for 4 seconds. Slams ground dealing magic damage and applying chill for 2 seconds.",
    target_type=AbilityTargetType.AOE_SELF,
    aoe_radius=2,
    cast_time=0.75,
    base_damage=[80, 120, 180],
    damage_type="magical",
    damage_scaling=1.0,
    base_shield=[400, 475, 600],
    shield_duration=4.0,
    grants_invulnerability=True,
    custom_data={
        "chill_duration": 2.0,
    },
)

ORIANNA_ABILITY = AbilityData(
    ability_id="orianna_command_shockwave",
    name="Command: Shockwave",
    description="Orianna commands her ball to create a shockwave, pulling enemies and dealing damage.",
    target_type=AbilityTargetType.AOE_ENEMY,
    aoe_radius=2,
    cast_time=0.5,
    base_damage=[250, 375, 600],
    damage_type="magical",
    damage_scaling=1.0,
    applies_cc=True,
    cc_type="stun",
    cc_duration=[1.0, 1.25, 1.5],
)

POPPY_ABILITY = AbilityData(
    ability_id="poppy_hammer_shock",
    name="Hammer Shock",
    description="Poppy gains a shield for 4 seconds and slams her hammer dealing physical damage. Knockup enemies hit. Shield fragments return to nearby allies.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    aoe_radius=1,
    cast_time=0.5,
    base_damage=[70, 105, 165],
    damage_type="physical",
    damage_scaling=1.0,
    base_shield=[330, 410, 500],
    shield_duration=4.0,
    applies_cc=True,
    cc_type="knockup",
    cc_duration=[1.0, 1.25, 1.5],
    custom_data={
        "shield_share_percent": 0.30,  # 30% shield to nearby allies
        "shield_share_targets": 2,
    },
)

REKSAI_ABILITY = AbilityData(
    ability_id="reksai_furious_bite",
    name="Furious Bite",
    description="Rek'Sai bites her target, dealing physical damage based on their missing health.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.25,
    base_damage=[250, 375, 600],
    damage_type="physical",
    damage_scaling=0.0,
    custom_data={"missing_hp_bonus": 0.5},  # +50% damage per 1% missing HP (capped)
)

SION_ABILITY = AbilityData(
    ability_id="sion_decimating_smash",
    name="Decimating Smash",
    description="Sion charges up and slams, dealing damage and stunning in a line.",
    target_type=AbilityTargetType.LINE,
    cast_time=1.0,
    base_damage=[300, 450, 700],
    damage_type="physical",
    damage_scaling=0.0,
    applies_cc=True,
    cc_type="stun",
    cc_duration=[1.5, 2.0, 2.5],
)

TEEMO_ABILITY = AbilityData(
    ability_id="teemo_noxious_trap",
    name="Noxious Trap",
    description="Teemo throws poison traps that deal magic damage over time.",
    target_type=AbilityTargetType.AOE_ENEMY,
    aoe_radius=2,
    cast_time=0.25,
    base_damage=[275, 425, 675],  # Total DoT damage
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={"dot_duration": 3.0},
)

TRISTANA_ABILITY = AbilityData(
    ability_id="tristana_explosive_charge",
    name="Explosive Charge",
    description="Tristana places a bomb on her target that explodes after a delay.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.25,
    base_damage=[300, 450, 700],
    damage_type="physical",
    damage_scaling=0.0,
    aoe_radius=1,  # Explosion hits nearby enemies
)

TRYNDAMERE_ABILITY = AbilityData(
    ability_id="tryndamere_undying_rage",
    name="Undying Rage",
    description="Tryndamere becomes enraged for 5 seconds with enhanced strikes dealing physical damage.",
    target_type=AbilityTargetType.SELF_BUFF,
    cast_time=0.25,
    base_damage=[120, 180, 270],  # Enhanced strike damage
    damage_type="physical",
    damage_scaling=1.0,
    can_crit=True,
    custom_data={
        "duration": 5.0,
        "cannot_die": True,
    },
)

TWISTED_FATE_ABILITY = AbilityData(
    ability_id="twisted_fate_stacked_deck",
    name="Stacked Deck",
    description="Passive: Every 4th attack deals bonus magic damage. Active: Throw cards dealing magic damage.",
    target_type=AbilityTargetType.MULTI_ENEMY,
    cast_time=0.25,
    base_damage=[70, 105, 160],  # Card damage
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "passive_bonus_damage": [30, 45, 70],  # Every 4th attack
        "marks_for_graves": True,
    },
)

VI_ABILITY = AbilityData(
    ability_id="vi_relentless_force",
    name="Relentless Force",
    description="Vi slams the ground dealing physical damage and gaining a shield. Every third cast hits enemies within 2 hexes and knocks them up for 1 second.",
    target_type=AbilityTargetType.AOE_SELF,
    aoe_radius=1,
    cast_time=0.25,
    base_damage=[120, 180, 270],
    damage_type="physical",
    damage_scaling=1.0,
    base_shield=[125, 150, 175],
    shield_duration=2.0,
    custom_data={
        "third_cast_damage": [150, 225, 340],
        "third_cast_range": 2,
        "third_cast_knockup": 1.0,
    },
)

XIN_ZHAO_ABILITY = AbilityData(
    ability_id="xin_zhao_three_talon_strike",
    name="Three Talon Strike",
    description="Xin Zhao strikes 3 times dealing physical damage. Heals and stuns the target.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.25,
    base_damage=[65, 100, 150],  # Per strike
    damage_type="physical",
    damage_scaling=1.0,
    base_healing=[95, 135, 170],
    healing_scaling=1.0,
    applies_cc=True,
    cc_type="stun",
    cc_duration=[1.0, 1.25, 1.5],
    custom_data={
        "num_strikes": 3,
    },
)

YASUO_ABILITY = AbilityData(
    ability_id="yasuo_sweeping_blade",
    name="Sweeping Blade",
    description="Yasuo dashes forward and strikes adjacent enemies dealing physical damage. Double damage if only one enemy hit.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.25,
    base_damage=[93, 137, 208],
    damage_type="physical",
    damage_scaling=1.0,
    can_crit=True,
    custom_data={
        "double_damage_if_single_target": True,
    },
)

YORICK_ABILITY = AbilityData(
    ability_id="yorick_dark_procession",
    name="Dark Procession",
    description="Yorick throws dark mist that deals magic damage, heals himself, and applies Chill.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.5,
    base_damage=[200, 300, 500],
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "heal": [100, 150, 250],
        "chill_duration": 3.0,
        "chill_slow": 0.3,  # 30% AS slow
    },
)


# =============================================================================
# 3-COST CHAMPION ABILITIES
# =============================================================================

AHRI_ABILITY = AbilityData(
    ability_id="ahri_fox_fire",
    name="Fox-Fire",
    description="Ahri fires 3 flames at target dealing magic damage. Every 3rd cast, fires 9 flames. Dashes away if enemies nearby.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.35,
    base_damage=[82, 125, 195],  # Per flame (3 flames = 246/375/585 total)
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "base_flames": 3,
        "empowered_flames": 9,
        "empowered_cast_interval": 3,
        "dash_on_nearby_enemy": True,
    },
)

DARIUS_ABILITY = AbilityData(
    ability_id="darius_decimate",
    name="Decimate",
    description="Darius spins dealing physical damage to adjacent enemies and restoring health. Applies Hemorrhage DoT. If target with Hemorrhage drops below 8% HP, instantly kill them. Each stack increases execute threshold by 5%.",
    target_type=AbilityTargetType.AOE_SELF,
    aoe_radius=1,
    cast_time=0.5,
    base_damage=[90, 135, 210],
    damage_type="physical",
    damage_scaling=1.0,
    base_healing=[270, 382, 572],
    healing_scaling=1.0,  # 10% HP + AP scaling
    custom_data={
        "hemorrhage_dps": [15, 25, 40],  # Per second
        "hemorrhage_duration": 4.0,
        "execute_threshold": 0.08,  # 8% base
        "execute_threshold_per_stack": 0.05,  # +5% per stack
    },
)

DR_MUNDO_ABILITY = AbilityData(
    ability_id="dr_mundo_goes_where_he_pleases",
    name="Maximum Dosage",
    description="Dr. Mundo enlarges for 5 seconds, recovering health per second and dealing bonus physical damage on attacks.",
    target_type=AbilityTargetType.SELF_BUFF,
    cast_time=0.25,
    base_damage=[105, 171, 291],  # Bonus damage per attack (5% max HP + 60 AD scaling)
    damage_type="physical",
    base_healing=[114, 177, 275],  # Per second (6% max HP + 60 AP scaling)
    custom_data={
        "duration": 5.0,
        "heal_hp_percent_per_sec": 0.06,
        "damage_hp_percent": 0.05,
    },
)

DRAVEN_ABILITY = AbilityData(
    ability_id="draven_spinning_axes",
    name="Rotating Axe",
    description="Draven throws an enhanced axe dealing physical damage. Gains Fan Cheers stacks on takedowns (doubled on kills). Every 11 stacks grants gold.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.35,
    base_damage=[150, 225, 375],
    damage_type="physical",
    damage_scaling=0.0,
    can_crit=True,
    custom_data={
        "gold_per_11_stacks": [1, 3, 5],
    },
)

GANGPLANK_ABILITY = AbilityData(
    ability_id="gangplank_parrrley",
    name="Powder Kegs",
    description="Gangplank throws kegs at 3 nearest enemies and detonates them. Damage reduces by 25% per connected keg. Applies armor reduction (10 base, 20 on crit).",
    target_type=AbilityTargetType.MULTI_ENEMY,
    aoe_radius=2,
    cast_time=0.35,
    base_damage=[200, 300, 475],  # 190 AD + 10 AP / 285 AD + 15 AP / 450 AD + 25 AP
    damage_type="physical",
    damage_scaling=0.0,
    can_crit=True,
)

GWEN_ABILITY = AbilityData(
    ability_id="gwen_snip_snip",
    name="Snip Snip!",
    description="Gwen dashes around enemies, performing 5+ cuts dealing magic damage. Center target takes bonus damage. Cut count increases per 80 souls.",
    target_type=AbilityTargetType.CONE,
    cast_time=0.25,
    base_damage=[45, 68, 105],  # Per snip to primary target (magic damage)
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "base_snip_count": 5,
        "aoe_damage": [20, 30, 50],  # Damage to other enemies in cone per snip
        "souls_per_extra_cut": 80,
    },
)

JINX_ABILITY = AbilityData(
    ability_id="jinx_rampage",
    name="Whirly-Whirly!",
    description="After 18/18/16 attacks, switches to fishhead launcher firing 3 rockets at random targets.",
    target_type=AbilityTargetType.MULTI_ENEMY,
    cast_time=0.25,
    base_damage=[46, 69, 119],  # Rocket damage (42 AD + 4 AP / 63 AD + 6 AP / 110 AD + 9 AP)
    damage_type="physical",
    damage_scaling=0.0,
    custom_data={
        "attacks_to_switch": [18, 18, 16],
        "rockets_fired": 3,
    },
)

KENNEN_ABILITY = AbilityData(
    ability_id="kennen_slicing_maelstrom",
    name="Slicing Maelstrom",
    description="Kennen gains shield and summons a storm striking 6 times over 3 seconds. First 3 hits on same target stuns for 1.5 seconds.",
    target_type=AbilityTargetType.AOE_SELF,
    aoe_radius=2,
    cast_time=0.5,
    base_damage=[120, 180, 290],  # Per lightning strike
    damage_type="magical",
    damage_scaling=1.0,
    base_shield=[600, 700, 900],
    shield_duration=3.0,
    applies_cc=True,
    cc_type="stun",
    cc_duration=[1.5, 1.5, 1.5],
    custom_data={
        "duration": 3.0,
        "strike_count": 6,
        "stun_on_bolt_count": 3,
        "stun_effectiveness_reduction": 0.50,  # 50% reduced on repeated stuns
    },
)

KOBUKO_ABILITY = AbilityData(
    ability_id="kobuko_best_friends",
    name="Best Friends",
    description="Kobuko and Yuumi empower the lowest health ally with a heal and shield.",
    target_type=AbilityTargetType.LOWEST_HP_ALLY,
    cast_time=0.5,
    base_healing=[300, 450, 700],
    healing_scaling=1.0,
    base_shield=[200, 300, 475],
    shield_duration=4.0,
)

LEBLANC_ABILITY = AbilityData(
    ability_id="leblanc_distortion",
    name="Distortion",
    description="LeBlanc summons 3 mirror phantoms striking target and up to 2 nearby enemies. Primary target takes more damage and is stunned for 1 second.",
    target_type=AbilityTargetType.MULTI_ENEMY,
    cast_time=0.35,
    base_damage=[270, 405, 630],  # Primary target damage
    damage_type="magical",
    damage_scaling=1.0,
    applies_cc=True,
    cc_type="stun",
    cc_duration=[1.0, 1.0, 1.0],
    custom_data={
        "secondary_damage": [150, 225, 350],
        "secondary_targets": 2,
    },
)

LEONA_ABILITY = AbilityData(
    ability_id="leona_solar_flare",
    name="Sunburst",
    description="Passive: Reduces damage per hit (15/30/60). Active: Stuns 3 nearest enemies for 1 second, deals damage, applies Grievous Wounds and 1% Burn for 4 seconds.",
    target_type=AbilityTargetType.MULTI_ENEMY,
    aoe_radius=2,
    cast_time=0.5,
    base_damage=[80, 120, 200],
    damage_type="magical",
    damage_scaling=1.0,
    applies_cc=True,
    cc_type="stun",
    cc_duration=[1.0, 1.0, 1.0],
    custom_data={
        "target_count": 3,
        "passive_damage_reduction": [15, 30, 60],
        "grievous_wounds_duration": 4.0,
        "burn_percent": 0.01,  # 1% max HP per second
        "burn_duration": 4.0,
    },
)

LORIS_ABILITY = AbilityData(
    ability_id="loris_piltover_brawl",
    name="Piltover Brawl",
    description="Loris gains shield for 4 seconds, dashes pushing target backward, stuns hit enemies for 1.25s. After dashing, nearby enemies are taunted.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.25,
    base_damage=[150, 225, 360],
    damage_type="magical",
    damage_scaling=1.0,
    base_shield=[700, 800, 1000],
    shield_duration=4.0,
    applies_cc=True,
    cc_type="stun",
    cc_duration=[1.25, 1.25, 1.25],
    custom_data={
        "taunt_nearby": True,
    },
)

MALZAHAR_ABILITY = AbilityData(
    ability_id="malzahar_void_swarm",
    name="Void Swarm",
    description="Malzahar summons 2 void spawn creatures that attack nearby enemies 8 times each, dealing magic damage per strike.",
    target_type=AbilityTargetType.SUMMON,
    aoe_radius=2,
    cast_time=0.5,
    base_damage=[27, 40, 65],  # Per hit (2 spawns x 8 attacks = 16 total hits)
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "spawn_count": 2,
        "attacks_per_spawn": 8,
    },
)

MILIO_ABILITY = AbilityData(
    ability_id="milio_ultra_mega_fire_kick",
    name="Ultra Mega Fire Kick!!!",
    description="Milio launches bouncing flame hitting targets 3 times. Final bounce deals AoE damage within 1 hex.",
    target_type=AbilityTargetType.MULTI_ENEMY,
    cast_time=0.5,
    base_damage=[190, 285, 445],  # Ball damage per hit
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "bounce_count": 3,
        "ricochet_damage": [160, 240, 375],  # Final AoE damage
        "final_aoe_radius": 1,
    },
)

NAUTILUS_ABILITY = AbilityData(
    ability_id="nautilus_titans_wrath",
    name="Titan's Wrath",
    description="Passive: Every 4 seconds and after skill, next attack deals magic damage in 1-tile radius (scales with MR). Active: Grants shield for 4 seconds.",
    target_type=AbilityTargetType.SELF_BUFF,
    cast_time=0.5,
    base_damage=[45, 68, 108],  # Passive bonus damage (90/135/215% scaling)
    damage_type="magical",
    damage_scaling=1.0,
    base_shield=[465, 562, 717],  # 10% HP + 375/400/425 AP scaling
    shield_duration=4.0,
    custom_data={
        "passive_trigger_interval": 4.0,
        "passive_aoe_radius": 1,
    },
)

SEJUANI_ABILITY = AbilityData(
    ability_id="sejuani_winters_wrath",
    name="Winter's Wrath",
    description="Sejuani gains shield for 4 seconds and strikes in cone and line dealing magic damage. Applies 30% Chill for 4 seconds. Pre-Chilled targets are stunned for 1 second.",
    target_type=AbilityTargetType.CONE,
    aoe_radius=2,
    cast_time=0.5,
    base_damage=[70, 105, 170],
    damage_type="magical",
    damage_scaling=1.0,
    base_shield=[475, 525, 625],
    shield_duration=4.0,
    applies_cc=True,
    cc_type="stun",
    cc_duration=[1.0, 1.0, 1.0],
    custom_data={
        "chill_percent": 0.30,
        "chill_duration": 4.0,
        "stun_on_pre_chilled": True,
    },
)

VAYNE_ABILITY = AbilityData(
    ability_id="vayne_tumble",
    name="Roll",
    description="Vayne rolls to adjacent tile and fires an arrow dealing true damage.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.25,
    base_damage=[106, 160, 245],  # 100 AD + 6 AP / 150 AD + 10 AP / 230 AD + 15 AP
    damage_type="true",
    damage_scaling=1.0,
    custom_data={
        "tumble_range": 1,
    },
)

ZOE_ABILITY = AbilityData(
    ability_id="zoe_sparkpack",
    name="Sparkpack",
    description="Passive: Applies 30% Shred for 6 seconds on damage. Active: Launches 2 bubbles dealing magic damage and applying 30% Chill for 2 seconds.",
    target_type=AbilityTargetType.MULTI_ENEMY,
    cast_time=0.35,
    base_damage=[300, 450, 700],
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "bubble_count": 2,
        "passive_shred_percent": 0.30,
        "passive_shred_duration": 6.0,
        "chill_percent": 0.30,
        "chill_duration": 2.0,
    },
)


# =============================================================================
# 4-COST CHAMPION ABILITIES
# =============================================================================

AMBESSA_ABILITY = AbilityData(
    ability_id="ambessa_rend_and_tear",
    name="Rend and Tear",
    description="Ambessa sweeps around target and dashes dealing physical damage. Then chain slams target for 275% of initial damage. Kill triggers recast at 100% damage.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    aoe_radius=2,
    cast_time=0.25,
    base_damage=[50, 75, 350],  # Sweep damage (45 AD + 5 AP)
    damage_type="physical",
    damage_scaling=0.0,
    custom_data={
        "slam_damage": [138, 206, 963],  # Chain slam (275% of sweep)
        "recast_on_kill": True,
        "recast_damage_percent": 1.0,
    },
)

BELVETH_ABILITY = AbilityData(
    ability_id="belveth_endless_banquet",
    name="Endless Feast",
    description="Bel'Veth transforms gaining 33% max HP, movement speed, and stacking AS on assists (100/100/200%). Attacks deal bonus damage with stacking damage per hit.",
    target_type=AbilityTargetType.SELF_BUFF,
    cast_time=0.25,
    base_damage=[150, 225, 900],  # Initial damage
    damage_type="physical",
    custom_data={
        "max_hp_increase_percent": 0.33,
        "fixed_damage": [15, 25, 100],
        "stacking_damage_per_hit": [3, 5, 30],
        "assist_as_bonus": [1.0, 1.0, 2.0],  # 100/100/200%
    },
)

BRAUM_ABILITY = AbilityData(
    ability_id="braum_unbreakable",
    name="Indestructible Frost",
    description="Braum raises shield for 4 seconds gaining 55/55/90% durability. Redirects projectiles. When struck, deals magic damage to nearby enemies and applies 30% Chill for 2 seconds.",
    target_type=AbilityTargetType.SELF_BUFF,
    cast_time=0.5,
    base_damage=[27, 43, 1245],  # Damage on hit (scales with armor + AP)
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "duration": 4.0,
        "durability_bonus": [0.55, 0.55, 0.90],
        "redirects_projectiles": True,
        "chill_percent": 0.30,
        "chill_duration": 2.0,
    },
)

DIANA_ABILITY = AbilityData(
    ability_id="diana_moonlight_slash",
    name="Moonlight Slash",
    description="Diana gains shield for 2 seconds. Fires moon energy at farthest unmarked enemy within 4 tiles, then dashes to all marked enemies dealing additional damage. With Leona: both apply marks reducing target damage by 10%.",
    target_type=AbilityTargetType.FARTHEST_ENEMY,
    aoe_radius=1,
    cast_time=0.25,
    base_damage=[165, 250, 500],  # Initial damage
    damage_type="magical",
    damage_scaling=1.0,
    base_shield=[200, 250, 300],
    shield_duration=2.0,
    custom_data={
        "secondary_damage": [150, 225, 1350],  # Dash damage to marked
        "mark_range": 4,
        "leona_synergy_damage_reduction": 0.10,
    },
)

FIZZ_ABILITY = AbilityData(
    ability_id="fizz_playful_trickster",
    name="Playful Trickster",
    description="Fizz becomes untargetable, leaps to farthest enemy within 4 hexes dealing magic damage to adjacent enemies. Gains 50% AS and bonus magic on-hit for 4 attacks.",
    target_type=AbilityTargetType.FARTHEST_ENEMY,
    aoe_radius=1,
    cast_time=0.5,
    base_damage=[110, 165, 450],
    damage_type="magical",
    damage_scaling=1.0,
    grants_invulnerability=True,
    custom_data={
        "jump_range": 4,
        "as_buff": 0.50,  # 50% at all levels
        "on_hit_magic": [80, 120, 600],
        "empowered_attacks": 4,
    },
)

GAREN_ABILITY = AbilityData(
    ability_id="garen_judgment",
    name="Judgment",
    description="Garen spins for 3 seconds, gaining 50/50/80% damage reduction and healing 420/566/1589 (10% HP + 300 AP). Each second, deals physical damage to adjacent enemies and reduces their armor/MR by 5.",
    target_type=AbilityTargetType.AOE_SELF,
    aoe_radius=2,
    cast_time=0.25,
    base_damage=[100, 150, 1700],  # Per second
    damage_type="physical",
    damage_scaling=1.0,
    base_healing=[420, 566, 1589],  # 10% max HP + 300 AP
    custom_data={
        "duration": 3.0,
        "durability_bonus": [0.50, 0.50, 0.80],
        "heal_max_hp_percent": 0.10,
        "resistance_shred_per_sec": 5,
    },
)

KAISA_ABILITY = AbilityData(
    ability_id="kaisa_icathian_rain",
    name="Icathian Rainfall",
    description="Kai'Sa dashes away from enemies, then fires 15/15/25 missiles at 4 nearest enemies dealing physical damage.",
    target_type=AbilityTargetType.MULTI_ENEMY,
    aoe_radius=3,
    cast_time=0.25,
    base_damage=[44, 66, 155],  # Per missile (38 AD + 6 AP / 57 AD + 9 AP / 135 AD + 20 AP)
    damage_type="physical",
    damage_scaling=0.0,
    custom_data={
        "num_missiles": [15, 15, 25],
        "max_targets": 4,
        "dash_away": True,
    },
)

KALISTA_ABILITY = AbilityData(
    ability_id="kalista_rend",
    name="Endless Rend",
    description="Kalista summons 20 spears distributed among 3 nearest enemies. After delay, rips them out dealing physical damage and reducing armor by 1 per spear. +1 spear per 25 souls.",
    target_type=AbilityTargetType.MULTI_ENEMY,
    cast_time=0.25,
    base_damage=[35, 53, 465],  # Per spear (32 AD + 3 AP / 48 AD + 5 AP / 450 AD + 15 AP)
    damage_type="physical",
    damage_scaling=0.0,
    custom_data={
        "base_spears": 20,
        "max_targets": 3,
        "armor_shred_per_spear": 1,
        "souls_per_extra_spear": 25,
    },
)

LISSANDRA_ABILITY = AbilityData(
    ability_id="lissandra_frozen_tomb",
    name="Ice Tomb",
    description="Lissandra encases target in ice for 1 second, dealing magic damage. Enemies within 2 tiles take secondary damage. Excess damage distributes to 2 nearest enemies. Chilled take 12% more.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    aoe_radius=2,
    cast_time=0.5,
    base_damage=[275, 415, 2800],  # Primary damage
    damage_type="magical",
    damage_scaling=1.0,
    applies_cc=True,
    cc_type="stun",
    cc_duration=[1.0, 1.0, 1.0],
    custom_data={
        "secondary_damage": [415, 625, 2800],
        "splits_excess_damage": True,
        "excess_targets": 2,
        "chilled_bonus_damage": 0.12,
        "applies_chill": True,
    },
)

LUX_ABILITY = AbilityData(
    ability_id="lux_final_spark",
    name="Final Spark",
    description="Lux fires a light orb at largest enemy cluster, damaging first 2 enemies and rooting for 1 second. Then fires beam dealing primary damage to orb-hit and secondary to others.",
    target_type=AbilityTargetType.LINE,
    cast_time=0.75,
    base_damage=[30, 45, 100],  # Initial orb damage
    damage_type="magical",
    damage_scaling=1.0,
    applies_cc=True,
    cc_type="root",
    cc_duration=[1.0, 1.0, 1.0],
    custom_data={
        "orb_targets": 2,
        "primary_beam_damage": [330, 500, 1600],  # To orb-hit targets
        "secondary_beam_damage": [90, 135, 900],  # To other targets
    },
)

MISS_FORTUNE_ABILITY = AbilityData(
    ability_id="miss_fortune_heartbreaker",
    name="Heartbreaker",
    description="Passive: Attacks bounce to highest HP enemy for 50/50/100% damage. Active: Fires bullets at 2 nearest enemies. Every 3rd cast fires extra bullets.",
    target_type=AbilityTargetType.MULTI_ENEMY,
    cast_time=0.5,
    base_damage=[145, 225, 3070],  # First hit (130 AD + 15 AP / 200 AD + 25 AP / 3000 AD + 70 AP)
    damage_type="physical",
    damage_scaling=0.0,
    can_crit=True,
    custom_data={
        "remaining_hit_damage": [94, 146, 3070],
        "num_targets": 2,
        "passive_bounce_percent": [0.50, 0.50, 1.0],
        "bonus_bullets_on_3rd_cast": True,
    },
)

NASUS_ABILITY = AbilityData(
    ability_id="nasus_fury_of_sands",
    name="Wrath of the Desert",
    description="Nasus steals HP from 4/4/8 nearby enemies, gains 30 armor/MR for 8 seconds. Deals aura damage per second (1.5/1.5/12% HP).",
    target_type=AbilityTargetType.AOE_SELF,
    aoe_radius=2,
    cast_time=0.5,
    base_damage=[23, 41, 583],  # Aura damage per second
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "hp_steal": [485, 620, 3000],
        "num_targets": [4, 4, 8],
        "bonus_armor": 30,
        "bonus_mr": 30,
        "duration": 8.0,
        "aura_hp_percent": [0.015, 0.015, 0.12],
    },
)

NIDALEE_ABILITY = AbilityData(
    ability_id="nidalee_pounce",
    name="Relentless Assault",
    description="Nidalee leaps to lowest-health adjacent enemy dealing magic damage and hitting nearby foes with secondary damage. On kill, recasts at 70%. Tanks take 60%, fighters take 30% more damage.",
    target_type=AbilityTargetType.LOWEST_HP_ENEMY,
    cast_time=0.25,
    base_damage=[320, 500, 2000],  # Primary damage
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "secondary_damage": [100, 150, 600],
        "recast_on_kill": True,
        "recast_damage_percent": 0.70,
        "tank_bonus_damage": 0.60,
        "fighter_bonus_damage": 0.30,
    },
)

RENEKTON_ABILITY = AbilityData(
    ability_id="renekton_slice_and_dice",
    name="Slice and Dice",
    description="Renekton dashes to nearest low-health enemy within 2 tiles dealing damage in path. Performs 4 slashes each dealing damage. Each cast increases slash count by 1.",
    target_type=AbilityTargetType.LOWEST_HP_ENEMY,
    cast_time=0.25,
    base_damage=[80, 120, 360],  # Dash damage (AD scaling)
    damage_type="physical",
    damage_scaling=0.0,
    custom_data={
        "slash_damage": [55, 85, 600],  # Per slash (50 AD + 5 AP / 75 AD + 10 AP / 550 AD + 50 AP)
        "base_slash_count": 4,
        "dash_range": 2,
    },
)

RIFT_HERALD_ABILITY = AbilityData(
    ability_id="rift_herald_charge",
    name="Herald's Charge",
    description="Rift Herald charges forward, damaging and knocking back enemies.",
    target_type=AbilityTargetType.LINE,
    cast_time=0.5,
    base_damage=[400, 600, 1500],
    damage_type="physical",
    damage_scaling=0.0,
    applies_cc=True,
    cc_type="knockup",
    cc_duration=[1.5, 2.0, 3.0],
)

SERAPHINE_ABILITY = AbilityData(
    ability_id="seraphine_gorgeous_encore",
    name="Gorgeous Encore",
    description="Seraphine gains 3 notes per cast. Each note deals magic damage to nearby enemies. At 12 notes, releases wave healing allies and damaging enemies. Wave damage reduces 30% per enemy pierced.",
    target_type=AbilityTargetType.ALL_ALLIES,
    cast_time=0.5,
    base_damage=[25, 40, 200],  # Per note damage
    damage_type="magical",
    damage_scaling=1.0,
    base_healing=[60, 90, 400],  # Wave healing
    custom_data={
        "notes_per_cast": 3,
        "notes_required": 12,
        "wave_damage": [270, 405, 2200],
        "wave_damage_reduction_per_enemy": 0.30,
    },
)

SINGED_ABILITY = AbilityData(
    ability_id="singed_toxic_trail",
    name="Toxic Trail",
    description="Passive: Leaves poison trail dealing magic damage per second. Active: Gains movement speed, heals per second, and +20 armor/MR for 4 seconds. Generates 7 mana per second (scales with AS).",
    target_type=AbilityTargetType.SELF_BUFF,
    cast_time=0.25,
    base_damage=[8, 18, 160],  # Passive poison damage per second
    damage_type="magical",
    damage_scaling=1.0,
    base_healing=[12, 65, 500],  # Heal per second
    custom_data={
        "duration": 4.0,
        "bonus_armor": 20,
        "bonus_mr": 20,
        "mana_per_second": 7,
    },
)

SKARNER_ABILITY = AbilityData(
    ability_id="skarner_impale",
    name="Piercing",
    description="Passive: All allies gain 10/20/100 armor. Active: Gains shield for 4 seconds. Pierces up to 3 enemies in line dealing physical damage (scales with armor) and stunning.",
    target_type=AbilityTargetType.LINE,
    cast_time=0.5,
    base_damage=[84, 126, 1750],  # Scales with armor
    damage_type="physical",
    damage_scaling=0.0,
    base_shield=[750, 950, 3000],
    shield_duration=4.0,
    applies_cc=True,
    cc_type="stun",
    cc_duration=[2.0, 2.25, 8.0],
    custom_data={
        "max_targets": 3,
        "passive_ally_armor": [10, 20, 100],
    },
)

SWAIN_ABILITY = AbilityData(
    ability_id="swain_binding_command",
    name="Binding Command",
    description="Passive: Heals per second and deals magic damage to enemies within 1 tile. Active: Summons demon eye dealing magic damage and stunning enemies.",
    target_type=AbilityTargetType.AOE_SELF,
    aoe_radius=1,
    cast_time=0.5,
    base_damage=[100, 150, 1800],  # Demon eye damage
    damage_type="magical",
    damage_scaling=1.0,
    base_healing=[36, 52, 158],  # Passive heal per second
    applies_cc=True,
    cc_type="stun",
    cc_duration=[1.5, 1.75, 8.0],
    custom_data={
        "passive_damage_per_sec": [20, 30, 90],
    },
)

TARIC_ABILITY = AbilityData(
    ability_id="taric_starlight_touch",
    name="Starlight's Touch",
    description="Passive: All allies gain 10/20/100 MR. At ≤35% HP, gains 90% damage reduction for 2 seconds (once). Active: Shields self and 2 lowest HP allies for 4 seconds. Gains bonus damage for 2/2/10 attacks.",
    target_type=AbilityTargetType.ALL_ALLIES,
    cast_time=0.5,
    base_damage=[200, 300, 3000],  # Bonus attack damage
    damage_type="physical",
    base_shield=[600, 700, 2500],  # Self shield
    shield_duration=4.0,
    custom_data={
        "ally_shield": [125, 150, 1000],
        "passive_ally_mr": [10, 20, 100],
        "low_hp_threshold": 0.35,
        "low_hp_damage_reduction": 0.90,
        "low_hp_dr_duration": 2.0,
        "empowered_attacks": [2, 2, 10],
    },
)

VEIGAR_ABILITY = AbilityData(
    ability_id="veigar_darkstorm",
    name="Darkstorm",
    description="Passive: AP gains increased by 50%. Permanently gains 1 AP per takedown. Active: Rains 12/12/24 meteors dealing magic damage. Meteors can crit.",
    target_type=AbilityTargetType.AOE_ENEMY,
    aoe_radius=3,
    cast_time=0.5,
    base_damage=[66, 99, 199],  # Per meteor
    damage_type="magical",
    damage_scaling=1.0,
    can_crit=True,
    custom_data={
        "meteor_count": [12, 12, 24],
        "passive_ap_multiplier": 1.50,
        "ap_per_takedown": 1,
    },
)

WARWICK_ABILITY = AbilityData(
    ability_id="warwick_eternal_hunger",
    name="Eternal Hunger",
    description="Warwick gains 100/100/400% Attack Speed, 15% Omnivamp, increased Move Speed, and deals bonus physical damage on attack for the rest of combat.",
    target_type=AbilityTargetType.SELF,
    cast_time=0.25,
    base_damage=[0, 0, 0],
    custom_data={
        "attack_speed_bonus": [1.0, 1.0, 4.0],  # 100/100/400%
        "omnivamp": 0.15,  # 15%
        "bonus_ad_on_hit": [45, 70, 550],
        "takedown_as_bonus": [0.20, 0.20, 0.50],  # For WW, Jinx, Vi
        "takedown_duration": 3.0,
        "permanent": True,
    },
)

WUKONG_ABILITY = AbilityData(
    ability_id="wukong_eye_catching_trickster",
    name="Eye-Catching Trickster",
    description="Passive: On death, summons rock clone with items (40% HP, 100/110/400 armor/MR) that damages nearby enemies. Active: Gains armor/MR for 4 seconds and spins dealing damage in 2 range.",
    target_type=AbilityTargetType.AOE_SELF,
    aoe_radius=2,
    cast_time=0.25,
    base_damage=[150, 225, 3000],  # Spin damage (AD scaling)
    damage_type="physical",
    damage_scaling=0.0,
    custom_data={
        "active_armor_mr": [100, 110, 400],
        "buff_duration": 4.0,
        "clone_hp_percent": 0.40,
        "clone_armor_mr": [100, 110, 400],
        "clone_spawn_damage": [150, 200, 2000],
    },
)

YONE_ABILITY = AbilityData(
    ability_id="yone_bloodstained_blade",
    name="Bloodstained Blade of Brothers",
    description="Passive: Alternates physical and magic damage on attacks. Active: Pierces up to 3 tiles, launches enemies for 1.5 seconds, distributes damage among all hit.",
    target_type=AbilityTargetType.LINE,
    cast_time=0.5,
    base_damage=[80, 120, 480],  # Skill physical damage
    damage_type="physical",
    damage_scaling=0.0,
    applies_cc=True,
    cc_type="knockup",
    cc_duration=[1.5, 1.5, 1.5],
    custom_data={
        "skill_magic_damage": [80, 120, 480],
        "distributed_physical": [320, 480, 2160],
        "distributed_magic": [320, 480, 2160],
        "passive_physical_on_hit": [80, 120, 800],
        "passive_magic_on_hit": [140, 210, 1400],
        "max_pierce_tiles": 3,
        "yasuo_synergy_bonus": True,
    },
)

YUNARA_ABILITY = AbilityData(
    ability_id="yunara_natures_wrath",
    name="Nature's Wrath",
    description="Yunara summons nature spirits that damage and slow enemies.",
    target_type=AbilityTargetType.AOE_ENEMY,
    aoe_radius=2,
    cast_time=0.5,
    base_damage=[350, 525, 1300],
    damage_type="magical",
    damage_scaling=1.0,
)


# =============================================================================
# 5-COST CHAMPION ABILITIES
# =============================================================================

AATROX_ABILITY = AbilityData(
    ability_id="aatrox_darkin_blade",
    name="The Darkin Blade",
    description="Aatrox cycles through Slash/Sweep/Slam. Sweep airbornes for 1 second. Slam executes below 15%. All apply 20% Bleed for 4 seconds.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    aoe_radius=2,
    cast_time=0.5,
    base_damage=[100, 195, 2500],  # Slash damage
    damage_type="physical",
    damage_scaling=1.0,
    custom_data={
        "sweep_damage": [120, 234, 3000],  # Cone with knockup
        "slam_damage": [160, 312, 4000],   # Execute slam
        "sweep_airborne_duration": 1.0,
        "execute_threshold": 0.15,
        "bleed_percent": 0.20,
        "bleed_duration": 4.0,
    },
)

ANNIE_ABILITY = AbilityData(
    ability_id="annie_enraged_inferno",
    name="Enraged Inferno",
    description="First cast applies Burn and Wound to all enemies (damage over 45 seconds). Subsequent casts fire single-target fireballs.",
    target_type=AbilityTargetType.ALL_ENEMIES,
    cast_time=0.5,
    base_damage=[1500, 2250, 8000],  # Total DoT damage over 45 seconds (first cast)
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "first_cast_dot_duration": 45.0,
        "applies_burn": True,
        "applies_wound": True,
        "fireball_damage": [240, 360, 3000],  # Subsequent cast fireball
    },
)

AZIR_ABILITY = AbilityData(
    ability_id="azir_arise",
    name="Arise!",
    description="Azir summons a Sand Soldier near the target. If 2 soldiers exist, they all attack together dealing magic damage. Passive: Soldiers deal magic damage when Azir attacks.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.5,
    base_damage=[70, 105, 5000],  # Active damage
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "passive_damage": [100, 150, 3000],
        "max_soldiers": 2,
    },
)

FIDDLESTICKS_ABILITY = AbilityData(
    ability_id="fiddlesticks_crowstorm",
    name="Raven Storm",
    description="Fiddlesticks teleports to largest enemy cluster, stuns for 1.25 seconds. Deals magic damage per second in 2-hex radius. 2 closest enemies take 33% more. Loses 22 mana/sec.",
    target_type=AbilityTargetType.AOE_SELF,
    aoe_radius=2,
    cast_time=0.75,
    base_damage=[100, 150, 6666],  # Damage per second
    damage_type="magical",
    damage_scaling=1.0,
    applies_cc=True,
    cc_type="stun",
    cc_duration=[1.25, 1.25, 1.25],
    custom_data={
        "mana_drain_per_sec": 22,
        "closest_bonus_damage": 0.33,
        "closest_targets": 2,
    },
)

GALIO_ABILITY = AbilityData(
    ability_id="galio_inspirational_might",
    name="Rubber Strength",
    description="Passive: Every 3rd attack deals magic damage in circle. Active: Shields allies within 2 hexes for 4 seconds, then crashes on largest cluster dealing damage.",
    target_type=AbilityTargetType.AOE_ENEMY,
    aoe_radius=2,
    cast_time=1.0,
    base_damage=[260, 439, 3250],  # Ground slam damage
    damage_type="magical",
    damage_scaling=1.0,
    base_shield=[500, 600, 2000],  # Shield to allies
    shield_duration=4.0,
    custom_data={
        "passive_damage": [247, 370, 5850],  # Every 3rd attack
    },
)

KINDRED_ABILITY = AbilityData(
    ability_id="kindred_lambs_respite",
    name="Lamb's Respite",
    description="Kindred creates a zone preventing ally deaths. Doubles Attack Speed and fires secondary arrows.",
    target_type=AbilityTargetType.AOE_SELF,
    aoe_radius=3,
    cast_time=0.5,
    base_damage=[25, 38, 999],  # Secondary arrow damage
    damage_type="physical",
    damage_scaling=1.0,
    custom_data={
        "zone_duration": [2.5, 2.5, 99.0],
        "min_hp_percent": 0.10,
        "attack_speed_bonus": 1.0,  # 100% AS = doubled
        "secondary_arrows": True,
    },
)

LUCIAN_SENNA_ABILITY = AbilityData(
    ability_id="lucian_senna_culling",
    name="The Culling",
    description="Lucian fires multiple shots at targets, each shot creating explosions from Senna.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.5,
    base_damage=[65, 100, 250],  # Per shot
    damage_type="physical",
    damage_scaling=0.0,
    can_crit=True,
    custom_data={
        "num_shots": [8, 10, 20],
        "senna_explosion_damage": [45, 70, 175],  # Per explosion
    },
)

MEL_ABILITY = AbilityData(
    ability_id="mel_radiant_eclipse",
    name="Radiant Eclipse",
    description="Passive: While gaining mana, casts 2 orbs dealing magic damage in 1-hex. Each hit absorbs 1 mana/sec. At 235 mana, gains random radiant item. Active: Deals massive damage to primary and secondary targets in 3-hex.",
    target_type=AbilityTargetType.AOE_ENEMY,
    aoe_radius=3,
    cast_time=0.5,
    base_damage=[900, 1500, 10000],  # Primary target damage
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "secondary_damage": [400, 600, 10000],
        "passive_orb_damage": [40, 60, 90],
        "mana_threshold_for_item": 235,
    },
)

ORNN_ABILITY = AbilityData(
    ability_id="ornn_call_of_the_forge_god",
    name="Call of the Forge God",
    description="Ornn summons an elemental dealing damage and Chilling enemies, then redirects for bonus damage.",
    target_type=AbilityTargetType.LINE,
    cast_time=0.75,
    base_damage=[100, 150, 3000],  # Initial elemental damage
    damage_type="magical",
    damage_scaling=1.0,
    applies_cc=True,
    cc_type="knockup",
    cc_duration=[1.5, 2.0, 4.0],
    custom_data={
        "redirect_damage": [150, 225, 4500],
        "applies_chill": True,
    },
)

SETT_ABILITY = AbilityData(
    ability_id="sett_showstopper",
    name="The Show Stopper",
    description="Sett grabs target and slams forward dealing magic damage plus max health damage. Enemies within 3 hexes take 50% damage.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    aoe_radius=3,
    cast_time=0.5,
    base_damage=[220, 350, 9999],
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "max_hp_percent": [0.03, 0.09, 9.0],  # 3/9/900% max HP scaling
        "aoe_damage_percent": 0.5,
    },
)

SHYVANA_ABILITY = AbilityData(
    ability_id="shyvana_dragons_descent",
    name="Dragon's Descent",
    description="Shyvana transforms gaining max HP and 20/20/90% durability. Dive-bombs dealing damage. Breathes fire for 3.5 seconds dealing magic damage per second.",
    target_type=AbilityTargetType.AOE_ENEMY,
    aoe_radius=2,
    cast_time=0.5,
    base_damage=[100, 150, 3000],  # Dive bomb damage (AD scaling)
    damage_type="physical",
    damage_scaling=0.0,
    custom_data={
        "bonus_max_hp": [600, 1000, 9999],
        "durability_bonus": [0.20, 0.20, 0.90],
        "fire_breath_dps": [185, 280, 13579],  # 165 AD + 20 AP scaling
        "fire_breath_duration": 3.5,
    },
)

THEX_ABILITY = AbilityData(
    ability_id="thex_hextech_arsenal",
    name="Hextech Arsenal",
    description="Passive: Barrage of 4 bullets. Active: Fires lasers and missiles split among enemies.",
    target_type=AbilityTargetType.ALL_ENEMIES,
    cast_time=0.75,
    base_damage=[135, 200, 2000],  # Laser damage
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "bonus_laser_damage": [10, 15, 150],
        "passive_bullets": 4,
    },
)

TAHM_KENCH_ABILITY = AbilityData(
    ability_id="tahm_kench_devour",
    name="Devour",
    description="Tahm Kench devours an enemy for 2.5s, removing from combat. Deals massive magic damage. May steal components/gold.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.5,
    base_damage=[850, 1275, 30000],
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "devour_duration": 2.5,
        "removes_from_combat": True,
        "can_steal_components": True,
    },
)

THRESH_ABILITY = AbilityData(
    ability_id="thresh_soul_prison",
    name="Soul Prison",
    description="Passive: Deals magic damage per second to enemies within 2 hexes. Heals for 50% of damage. Active: Creates 5-second barrier doubling passive damage. Pulls enemies in if fewer than 2. Wall contact deals damage.",
    target_type=AbilityTargetType.AOE_SELF,
    aoe_radius=2,
    cast_time=0.5,
    base_damage=[100, 150, 4000],  # Wall contact damage
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "passive_dps": [30, 50, 400],
        "secondary_damage": [70, 105, 5000],
        "passive_healing_percent": 0.50,
        "wall_duration": 5.0,
    },
)

VOLIBEAR_ABILITY = AbilityData(
    ability_id="volibear_relentless_storm",
    name="The Relentless Storm",
    description="Volibear bites target dealing damage and marking (marked take 60% more). After 5 casts, leaps gaining Storm Form with bonus HP, AS, and lightning every 2 seconds.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.5,
    base_damage=[136, 215, 2117],  # Bite damage
    damage_type="physical",
    damage_scaling=1.0,
    custom_data={
        "slam_damage": [272, 430, 4233],
        "mark_bonus_damage": 0.60,
        "casts_to_storm_form": 5,
        "storm_form_hp": [400, 550, 9999],
        "storm_form_as": [0.50, 0.50, 9.99],
        "lightning_damage": [98, 161, 1155],
        "lightning_interval": 2.0,
    },
)

XERATH_ABILITY = AbilityData(
    ability_id="xerath_eye_of_ascendant",
    name="Eye of the Ascendant",
    description="Xerath fires 10/10/99 arcane bombardments randomly distributed among 4 nearest enemies, each dealing magic damage.",
    target_type=AbilityTargetType.MULTI_ENEMY,
    aoe_radius=1,
    cast_time=0.5,
    base_damage=[390, 650, 2500],  # Per bombardment
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "num_bombardments": [10, 10, 99],
        "max_targets": 4,
    },
)

ZILEAN_ABILITY = AbilityData(
    ability_id="zilean_visions_of_the_end",
    name="Apocalypse Vision",
    description="Zilean throws time bomb at nearest enemy without one. Deals magic damage per second. On death, explodes dealing AoE damage. If timer expires while Zilean lives, target dies. Grants 15 mana on bomb detonation.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    aoe_radius=2,
    cast_time=0.5,
    base_damage=[75, 115, 1000],  # Per second
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "explosion_damage": [150, 225, 4000],
        "execute_timer": [18, 16, 1],
        "instant_kill_on_expire": True,
        "mana_on_detonation": 15,
    },
)


# =============================================================================
# 7-COST (LEGENDARY) CHAMPION ABILITIES
# =============================================================================

AURELION_SOL_ABILITY = AbilityData(
    ability_id="aurelion_sol_celestial_descent",
    name="Celestial Descent",
    description="Aurelion Sol drops stars dealing magic damage in 2-hex radius. Unlocks enhancements via Star Dust: 15(expand), 60(+15% dmg), 100(radius 4), 175(2s airborne), 250(radius 10), 400(+33% fixed), 700(meteor storm), 1988(black hole execute).",
    target_type=AbilityTargetType.AOE_ENEMY,
    aoe_radius=2,
    cast_time=1.0,
    base_damage=[350, 615, 5000],  # Star impact damage
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "shockwave_damage": [50, 75, 1000],
        "meteor_storm_damage": [1000, 1500, 9999],
        "dust_thresholds": {
            "expand_shockwave": 15,
            "star_damage_bonus": 60,
            "radius_4": 100,
            "airborne_effect": 175,
            "radius_10": 250,
            "fixed_damage_amp": 400,
            "meteor_storm": 700,
            "black_hole_execute": 1988,
        },
    },
)

BARON_NASHOR_ABILITY = AbilityData(
    ability_id="baron_nashor_void_fury",
    name="Void Fury",
    description="Passive: CC immune. Attacks damage enemies in line behind target. Charges if no target, knocking up on arrival. Spawns tentacles every 3s (1.5s knockup). Active: Slams 2-hex, spawns spikes, fires 10 acid drops over 3 seconds.",
    target_type=AbilityTargetType.AOE_SELF,
    aoe_radius=2,
    cast_time=0.75,
    base_damage=[280, 420, 20500],  # Slam/spike damage
    damage_type="physical",
    damage_scaling=0.0,
    applies_cc=True,
    cc_type="knockup",
    cc_duration=[1.5, 1.5, 1.5],
    custom_data={
        "acid_damage": [140, 210, 10250],
        "num_acid_globs": 10,
        "acid_duration": 3.0,
        "cc_immune": True,
        "tentacle_interval": 3.0,
        "tentacle_knockup_duration": 1.5,
    },
)

BROCK_ABILITY = AbilityData(
    ability_id="brock_seismic_slam",
    name="Seismic Slam",
    description="Passive: CC immune. Attacks knock enemies airborne and deal 50% damage to nearby. Active: Slams dealing physical damage (reduces 15% per tile, min 40%). First hit knocks up 1.75s. Spawns 12 rocks dealing damage.",
    target_type=AbilityTargetType.ALL_ENEMIES,
    cast_time=0.75,
    base_damage=[186, 265, 1617],  # Slam damage
    damage_type="physical",
    damage_scaling=1.0,
    applies_cc=True,
    cc_type="knockup",
    cc_duration=[1.75, 1.75, 1.75],
    custom_data={
        "rock_damage": [220, 405, 700],
        "num_rocks": 12,
        "min_damage_percent": 0.40,
        "reduction_per_hex": 0.15,
        "cc_immune": True,
        "passive_splash_percent": 0.50,
    },
)

RYZE_ABILITY = AbilityData(
    ability_id="ryze_runic_overload",
    name="Runic Overload",
    description="Fire a runic blast dealing magic damage. Splits into 2 projectiles hitting closest enemies for 33% damage, which split again. Region bonuses: Bilgewater(explosion), Demacia(execute), Freljord(true dmg+chill), Ionia(extra split).",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.5,
    base_damage=[235, 425, 1500],
    damage_type="magical",
    damage_scaling=1.0,
    custom_data={
        "split_ratio": 0.33,
        "split_count": 2,
        # Region bonuses applied dynamically in handler
    },
)

SYLAS_ABILITY = AbilityData(
    ability_id="sylas_stolen_power",
    name="Stolen Power",
    description="Sylas cycles between 3 spells. Cataclysm: AOE stun (1.5/1.75/30s). Demacian Justice: Execute at 15% HP. Final Flash: Shield + damage.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    aoe_radius=2,
    cast_time=0.5,
    base_damage=[100, 180, 1000],  # Cataclysm damage
    damage_type="magical",
    damage_scaling=1.0,
    applies_cc=True,
    cc_type="stun",
    cc_duration=[1.5, 1.75, 30.0],
    custom_data={
        "ability_rotation": ["cataclysm", "demacian_justice", "final_flash"],
        "demacian_justice_damage": [670, 1000, 99999],
        "demacian_justice_execute": 0.15,
        "final_flash_damage": [300, 475, 1000],
        "final_flash_shield": [400, 450, 800],
    },
)

ZAAHEN_ABILITY = AbilityData(
    ability_id="zaahen_divine_challenge",
    name="Divine Challenge",
    description="Dash and slash dealing damage. If target survives, recast. After 25 casts, execute target + 3hex AOE.",
    target_type=AbilityTargetType.SINGLE_ENEMY,
    cast_time=0.25,
    base_damage=[70, 105, 1500],  # Main slash damage
    damage_type="physical",
    damage_scaling=1.0,
    custom_data={
        "ad_scaling": [20, 30, 200],
        "execute_damage": [70, 105, 1500],  # AOE execute damage
        "max_casts": 25,
    },
)


# =============================================================================
# ABILITY REGISTRY
# =============================================================================

# Map champion IDs to their abilities
CHAMPION_ABILITIES: Dict[str, AbilityData] = {
    # 1-cost
    "anivia": ANIVIA_ABILITY,
    "blitzcrank": BLITZCRANK_ABILITY,
    "briar": BRIAR_ABILITY,
    "caitlyn": CAITLYN_ABILITY,
    "illaoi": ILLAOI_ABILITY,
    "jarvan_iv": JARVAN_IV_ABILITY,
    "jhin": JHIN_ABILITY,
    "kogmaw": KOGMAW_ABILITY,
    "lulu": LULU_ABILITY,
    "qiyana": QIYANA_ABILITY,
    "rumble": RUMBLE_ABILITY,
    "shen": SHEN_ABILITY,
    "sona": SONA_ABILITY,
    "viego": VIEGO_ABILITY,

    # 2-cost
    "aphelios": APHELIOS_ABILITY,
    "ashe": ASHE_ABILITY,
    "bard": BARD_ABILITY,
    "chogath": CHOGATH_ABILITY,
    "ekko": EKKO_ABILITY,
    "graves": GRAVES_ABILITY,
    "neeko": NEEKO_ABILITY,
    "orianna": ORIANNA_ABILITY,
    "poppy": POPPY_ABILITY,
    "reksai": REKSAI_ABILITY,
    "sion": SION_ABILITY,
    "teemo": TEEMO_ABILITY,
    "tristana": TRISTANA_ABILITY,
    "tryndamere": TRYNDAMERE_ABILITY,
    "twisted_fate": TWISTED_FATE_ABILITY,
    "vi": VI_ABILITY,
    "xin_zhao": XIN_ZHAO_ABILITY,
    "yasuo": YASUO_ABILITY,
    "yorick": YORICK_ABILITY,

    # 3-cost
    "ahri": AHRI_ABILITY,
    "darius": DARIUS_ABILITY,
    "dr_mundo": DR_MUNDO_ABILITY,
    "draven": DRAVEN_ABILITY,
    "gangplank": GANGPLANK_ABILITY,
    "gwen": GWEN_ABILITY,
    "jinx": JINX_ABILITY,
    "kennen": KENNEN_ABILITY,
    "kobuko_yuumi": KOBUKO_ABILITY,
    "leblanc": LEBLANC_ABILITY,
    "leona": LEONA_ABILITY,
    "loris": LORIS_ABILITY,
    "malzahar": MALZAHAR_ABILITY,
    "milio": MILIO_ABILITY,
    "nautilus": NAUTILUS_ABILITY,
    "sejuani": SEJUANI_ABILITY,
    "vayne": VAYNE_ABILITY,
    "zoe": ZOE_ABILITY,

    # 4-cost
    "ambessa": AMBESSA_ABILITY,
    "belveth": BELVETH_ABILITY,
    "braum": BRAUM_ABILITY,
    "diana": DIANA_ABILITY,
    "fizz": FIZZ_ABILITY,
    "garen": GAREN_ABILITY,
    "kaisa": KAISA_ABILITY,
    "kalista": KALISTA_ABILITY,
    "lissandra": LISSANDRA_ABILITY,
    "lux": LUX_ABILITY,
    "miss_fortune": MISS_FORTUNE_ABILITY,
    "nasus": NASUS_ABILITY,
    "nidalee": NIDALEE_ABILITY,
    "renekton": RENEKTON_ABILITY,
    "rift_herald": RIFT_HERALD_ABILITY,
    "seraphine": SERAPHINE_ABILITY,
    "singed": SINGED_ABILITY,
    "skarner": SKARNER_ABILITY,
    "swain": SWAIN_ABILITY,
    "taric": TARIC_ABILITY,
    "veigar": VEIGAR_ABILITY,
    "warwick": WARWICK_ABILITY,
    "wukong": WUKONG_ABILITY,
    "yone": YONE_ABILITY,
    "yunara": YUNARA_ABILITY,

    # 5-cost
    "aatrox": AATROX_ABILITY,
    "annie": ANNIE_ABILITY,
    "azir": AZIR_ABILITY,
    "fiddlesticks": FIDDLESTICKS_ABILITY,
    "galio": GALIO_ABILITY,
    "kindred": KINDRED_ABILITY,
    "lucian_senna": LUCIAN_SENNA_ABILITY,
    "mel": MEL_ABILITY,
    "ornn": ORNN_ABILITY,
    "sett": SETT_ABILITY,
    "shyvana": SHYVANA_ABILITY,
    "t_hex": THEX_ABILITY,
    "tahm_kench": TAHM_KENCH_ABILITY,
    "thresh": THRESH_ABILITY,
    "volibear": VOLIBEAR_ABILITY,
    "xerath": XERATH_ABILITY,
    "zilean": ZILEAN_ABILITY,

    # 7-cost (Legendary)
    "aurelion_sol": AURELION_SOL_ABILITY,
    "baron_nashor": BARON_NASHOR_ABILITY,
    "brock": BROCK_ABILITY,
    "ryze": RYZE_ABILITY,
    "sylas": SYLAS_ABILITY,
    "zaahen": ZAAHEN_ABILITY,
}


def register_all_abilities(ability_system: AbilitySystem) -> None:
    """Register all champion abilities with the ability system."""
    for champion_id, ability_data in CHAMPION_ABILITIES.items():
        ability_system.register_ability(champion_id, ability_data)


def get_ability_for_champion(champion_id: str) -> Optional[AbilityData]:
    """Get the ability data for a champion by ID."""
    # Normalize ID (lowercase, replace spaces with underscores)
    normalized_id = champion_id.lower().replace(" ", "_").replace("'", "")
    return CHAMPION_ABILITIES.get(normalized_id)
