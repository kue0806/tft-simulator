#!/usr/bin/env python3
"""Combat Test - 전투 시뮬레이션 테스트 스크립트

전투 과정을 시각적으로 확인할 수 있는 테스트 스크립트입니다.
"""

import time
import os
import sys
from typing import Optional

from src.core.game_state import GameState
from src.combat import CombatEngine, HexPosition, Team
from src.data.loaders import load_champions


def clear_screen():
    """화면 지우기"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str):
    """헤더 출력"""
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_unit_status(unit, position=None):
    """유닛 상태 출력"""
    hp_bar_len = 20
    hp_percent = unit.stats.current_hp / unit.stats.max_hp if unit.stats.max_hp > 0 else 0
    hp_filled = int(hp_bar_len * hp_percent)
    hp_bar = "█" * hp_filled + "░" * (hp_bar_len - hp_filled)

    mana_percent = unit.stats.current_mana / unit.stats.max_mana if unit.stats.max_mana > 0 else 0
    mana_filled = int(10 * mana_percent)
    mana_bar = "█" * mana_filled + "░" * (10 - mana_filled)

    team_color = "🔵" if unit.team == Team.BLUE else "🔴"
    star = "★" * unit.star_level

    pos_str = f"({position.row},{position.col})" if position else ""

    print(f"  {team_color} {unit.name}{star} {pos_str}")
    print(f"     HP: [{hp_bar}] {int(unit.stats.current_hp)}/{int(unit.stats.max_hp)}")
    print(f"     MP: [{mana_bar}] {int(unit.stats.current_mana)}/{int(unit.stats.max_mana)}")
    print(f"     AD: {int(unit.stats.attack_damage)} | AS: {unit.stats.attack_speed:.2f} | 상태: {unit.state.name}")


def print_combat_log(events: list, last_n: int = 5):
    """최근 전투 로그 출력"""
    print("\n  [전투 로그]")
    recent = events[-last_n:] if len(events) > last_n else events

    for event in recent:
        tick = event.get('tick', 0)
        event_type = event.get('type', '')

        if event_type == 'attack':
            damage = event.get('damage', 0)
            crit = " (치명타!)" if event.get('crit') else ""
            killed = " 💀 처치!" if event.get('killed') else ""
            print(f"    [{tick:4d}] ⚔️  공격: {damage:.0f} 데미지{crit}{killed}")

        elif event_type == 'ability_start':
            ability = event.get('ability', '')
            print(f"    [{tick:4d}] ✨ 스킬 시전: {ability}")

        elif event_type == 'ability_cast':
            ability = event.get('ability', '')
            damage = event.get('damage', 0)
            targets = len(event.get('targets', []))
            print(f"    [{tick:4d}] 💥 {ability}: {damage:.0f} 데미지 ({targets}명 적중)")

        elif 'item' in event_type or 'buff' in event_type:
            print(f"    [{tick:4d}] 🔮 {event_type}: {event}")


def run_visual_combat(
    blue_champion_name: str,
    red_champion_name: str,
    blue_items: list = None,
    red_items: list = None,
    speed: float = 0.1,
    max_ticks: int = 600,
):
    """
    시각적 전투 시뮬레이션 실행

    Args:
        blue_champion_name: 블루팀 챔피언 이름
        red_champion_name: 레드팀 챔피언 이름
        blue_items: 블루팀 아이템 ID 리스트
        red_items: 레드팀 아이템 ID 리스트
        speed: 업데이트 속도 (초)
        max_ticks: 최대 틱 수
    """
    # 챔피언 데이터 로드
    champions = {c.name.lower(): c for c in load_champions()}

    blue_champ = champions.get(blue_champion_name.lower())
    red_champ = champions.get(red_champion_name.lower())

    if not blue_champ:
        print(f"챔피언을 찾을 수 없습니다: {blue_champion_name}")
        return
    if not red_champ:
        print(f"챔피언을 찾을 수 없습니다: {red_champion_name}")
        return

    # 게임 상태 생성
    game = GameState(num_players=2)
    player1 = game.players[0]
    player2 = game.players[1]

    # 챔피언 인스턴스 생성
    from src.core.player_units import ChampionInstance
    blue_instance = ChampionInstance(champion=blue_champ, star_level=2)
    red_instance = ChampionInstance(champion=red_champ, star_level=2)

    # 아이템 장착
    if blue_items:
        for item_id in blue_items:
            item = player1.items.get_item(item_id)
            if item:
                item_inst = player1.items.add_to_inventory(item)
                player1.items.equip_item(item_inst, blue_instance)

    if red_items:
        for item_id in red_items:
            item = player2.items.get_item(item_id)
            if item:
                item_inst = player2.items.add_to_inventory(item)
                player2.items.equip_item(item_inst, red_instance)

    # 전투 엔진 설정
    engine = CombatEngine(seed=42)

    blue_pos = HexPosition(row=1, col=2)
    red_pos = HexPosition(row=1, col=4)

    blue_board = {blue_pos: blue_instance}
    red_board = {red_pos: red_instance}

    engine.setup_combat_from_boards(blue_board, red_board)

    # 유닛 찾기
    blue_unit = None
    red_unit = None
    for unit in engine.units.values():
        if unit.team == Team.BLUE:
            blue_unit = unit
        else:
            red_unit = unit

    # 스킬 정보
    blue_ability = engine.ability_system.get_ability(blue_unit.champion_id)
    red_ability = engine.ability_system.get_ability(red_unit.champion_id)

    # 아이템 정보
    blue_item_names = [i.item.name for i in blue_instance.items] if blue_instance.items else []
    red_item_names = [i.item.name for i in red_instance.items] if red_instance.items else []

    clear_screen()
    print_header("TFT 전투 시뮬레이션")
    print(f"\n  🔵 {blue_champ.name} (2성) vs 🔴 {red_champ.name} (2성)")
    print(f"  🔵 스킬: {blue_ability.name if blue_ability else '없음'}")
    print(f"  🔴 스킬: {red_ability.name if red_ability else '없음'}")
    if blue_item_names:
        print(f"  🔵 아이템: {', '.join(blue_item_names)}")
    if red_item_names:
        print(f"  🔴 아이템: {', '.join(red_item_names)}")
    print(f"\n  속도: {speed}초/업데이트 (Enter로 시작, 'q'로 종료)")
    input()

    tick_count = 0
    last_events_count = 0

    try:
        while tick_count < max_ticks:
            # 전투 틱
            if not engine.tick():
                break

            tick_count += 1

            # 10틱마다 화면 업데이트 (더 빠른 시뮬레이션)
            if tick_count % 10 == 0:
                clear_screen()
                print_header(f"전투 진행 중 - Tick {tick_count} ({tick_count / 30:.1f}초)")

                print("\n  [유닛 상태]")
                blue_pos = engine.grid.get_unit_position(blue_unit.id)
                red_pos = engine.grid.get_unit_position(red_unit.id)

                print_unit_status(blue_unit, blue_pos)
                print()
                print_unit_status(red_unit, red_pos)

                # 전투 로그
                events = engine.get_events()
                new_events = events[last_events_count:]
                last_events_count = len(events)

                if new_events:
                    print_combat_log(new_events, 5)

                time.sleep(speed)

        # 결과
        result = engine.get_result()

        clear_screen()
        print_header("전투 결과")

        winner_team = "🔵 블루팀" if result.winner == Team.BLUE else "🔴 레드팀"
        print(f"\n  🏆 승자: {winner_team}")
        print(f"  ⏱️  전투 시간: {result.rounds_taken / 30:.1f}초 ({result.rounds_taken} 틱)")
        print(f"  💀 플레이어 데미지: {result.total_damage_to_loser:.0f}")

        print("\n  [최종 상태]")
        print_unit_status(blue_unit)
        print()
        print_unit_status(red_unit)

        print("\n  [유닛 통계]")
        for uid, stats in result.unit_stats.items():
            team = "🔵" if stats['team'] == 'blue' else "🔴"
            alive = "생존" if stats['alive'] else "사망"
            print(f"    {team} {stats['name']}: 데미지 {stats['damage_dealt']:.0f} | 받은 피해 {stats['damage_taken']:.0f} | 킬 {stats['kills']} | {alive}")

        # 전체 전투 로그 요약
        events = engine.get_events()
        attack_count = len([e for e in events if e.get('type') == 'attack'])
        ability_count = len([e for e in events if e.get('type') == 'ability_cast'])

        print(f"\n  [전투 요약]")
        print(f"    총 공격: {attack_count}회")
        print(f"    스킬 시전: {ability_count}회")

    except KeyboardInterrupt:
        print("\n\n전투가 중단되었습니다.")


def run_quick_combat(
    blue_champion: str,
    red_champion: str,
    blue_star: int = 1,
    red_star: int = 1,
    blue_items: list = None,
    red_items: list = None,
):
    """빠른 전투 결과 확인"""
    champions = {c.name.lower(): c for c in load_champions()}

    blue_champ = champions.get(blue_champion.lower())
    red_champ = champions.get(red_champion.lower())

    if not blue_champ or not red_champ:
        print("챔피언을 찾을 수 없습니다.")
        return

    game = GameState(num_players=2)
    player1 = game.players[0]
    player2 = game.players[1]

    from src.core.player_units import ChampionInstance
    blue_instance = ChampionInstance(champion=blue_champ, star_level=blue_star)
    red_instance = ChampionInstance(champion=red_champ, star_level=red_star)

    # 아이템 장착
    if blue_items:
        for item_id in blue_items:
            item = player1.items.get_item(item_id)
            if item:
                item_inst = player1.items.add_to_inventory(item)
                player1.items.equip_item(item_inst, blue_instance)

    if red_items:
        for item_id in red_items:
            item = player2.items.get_item(item_id)
            if item:
                item_inst = player2.items.add_to_inventory(item)
                player2.items.equip_item(item_inst, red_instance)

    engine = CombatEngine(seed=42)

    blue_board = {HexPosition(row=1, col=2): blue_instance}
    red_board = {HexPosition(row=1, col=4): red_instance}

    engine.setup_combat_from_boards(blue_board, red_board)
    result = engine.run_combat()

    # 결과 출력
    print_header("빠른 전투 결과")

    blue_items_str = ", ".join([i.item.name for i in blue_instance.items]) if blue_instance.items else "없음"
    red_items_str = ", ".join([i.item.name for i in red_instance.items]) if red_instance.items else "없음"

    print(f"\n  🔵 {blue_champ.name} ({blue_star}성) - 아이템: {blue_items_str}")
    print(f"  🔴 {red_champ.name} ({red_star}성) - 아이템: {red_items_str}")

    winner = "🔵 블루팀" if result.winner == Team.BLUE else "🔴 레드팀"
    print(f"\n  🏆 승자: {winner}")
    print(f"  ⏱️  전투 시간: {result.rounds_taken / 30:.1f}초")

    # 전투 로그
    events = engine.get_events()
    ability_events = [e for e in events if 'ability' in str(e.get('type', ''))]

    print(f"\n  [스킬 시전 로그]")
    for e in ability_events:
        tick = e.get('tick', 0)
        if e.get('type') == 'ability_start':
            print(f"    [{tick / 30:.1f}초] {e.get('ability')} 시전 시작")
        elif e.get('type') == 'ability_cast':
            print(f"    [{tick / 30:.1f}초] {e.get('ability')} → {e.get('damage', 0):.0f} 데미지")


def list_champions():
    """챔피언 목록 출력"""
    champions = load_champions()

    print_header("챔피언 목록")

    by_cost = {}
    for champ in champions:
        if champ.cost not in by_cost:
            by_cost[champ.cost] = []
        by_cost[champ.cost].append(champ.name)

    for cost in sorted(by_cost.keys()):
        print(f"\n  [{cost}코스트]")
        names = sorted(by_cost[cost])
        for i in range(0, len(names), 4):
            row = names[i:i+4]
            print(f"    {', '.join(row)}")


def list_items():
    """아이템 목록 출력"""
    from src.data.loaders import load_components, load_combined_items

    print_header("아이템 목록")

    print("\n  [컴포넌트]")
    for item in load_components():
        stats = []
        if item.stats.ad: stats.append(f"+{item.stats.ad} AD")
        if item.stats.ap: stats.append(f"+{item.stats.ap} AP")
        if item.stats.armor: stats.append(f"+{item.stats.armor} 방어력")
        if item.stats.mr: stats.append(f"+{item.stats.mr} 마저")
        if item.stats.health: stats.append(f"+{item.stats.health} 체력")
        if item.stats.attack_speed: stats.append(f"+{item.stats.attack_speed}% 공속")
        print(f"    {item.id}: {item.name} - {', '.join(stats)}")

    print("\n  [완성 아이템] (일부)")
    for item in load_combined_items()[:15]:
        if item.components:
            print(f"    {item.id}: {item.name} ({item.components[0]} + {item.components[1]})")


def main():
    """메인 함수"""
    print_header("TFT 전투 테스트")
    print("""
  사용법:
    1. 시각적 전투 (실시간)
    2. 빠른 전투 결과
    3. 챔피언 목록
    4. 아이템 목록
    5. 커스텀 전투 설정
    q. 종료
    """)

    while True:
        choice = input("\n선택: ").strip()

        if choice == '1':
            print("\n기본 전투 예시: Blitzcrank vs Lux")
            run_visual_combat("Blitzcrank", "Lux", speed=0.05)

        elif choice == '2':
            print("\n[빠른 전투 테스트]")
            print("예시 1: Aatrox(2성) vs Garen(2성)")
            run_quick_combat("Aatrox", "Garen", 2, 2)

            print("\n예시 2: Jinx(2성, Giant Slayer) vs Sion(2성)")
            run_quick_combat("Jinx", "Sion", 2, 2,
                           blue_items=['bf_sword', 'recurve_bow'])

        elif choice == '3':
            list_champions()

        elif choice == '4':
            list_items()

        elif choice == '5':
            print("\n[커스텀 전투]")
            blue = input("  블루팀 챔피언: ").strip()
            red = input("  레드팀 챔피언: ").strip()

            star_input = input("  스타 레벨 (기본 2): ").strip()
            star = int(star_input) if star_input else 2

            item_input = input("  블루팀 아이템 (쉼표 구분, 예: bf_sword,recurve_bow): ").strip()
            blue_items = [i.strip() for i in item_input.split(',')] if item_input else None

            mode = input("  모드 (1=시각적, 2=빠른): ").strip()

            if mode == '1':
                run_visual_combat(blue, red, blue_items=blue_items, speed=0.05)
            else:
                run_quick_combat(blue, red, star, star, blue_items=blue_items)

        elif choice.lower() == 'q':
            print("종료합니다.")
            break

        else:
            print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()
