"""Interactive TFT Game for CLI.

Play TFT against 7 AI bots in the terminal.
"""

import os
import random
import sys
from typing import Optional, List, Tuple

from src.core.game_state import GameState, PlayerState
from src.core.stage_manager import RoundType
from src.core.constants import BASE_STAGE_DAMAGE, UNIT_DAMAGE_BY_STAR_AND_COST
from src.game.bot_ai import BotAI, BotDifficulty, BotStrategy
from src.combat import (
    CombatEngine,
    CombatResult,
    HexPosition,
    Team,
)


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    print(f"\n{char * 60}")
    print(f"  {text}")
    print(f"{char * 60}")


def print_divider(char: str = "-"):
    """Print a divider line."""
    print(char * 60)


class InteractiveGame:
    """Interactive TFT game with CLI interface."""

    def __init__(self, difficulty: BotDifficulty = BotDifficulty.MEDIUM):
        """
        Initialize the game.

        Args:
            difficulty: Difficulty of AI bots.
        """
        self.game = GameState(num_players=8, seed=random.randint(0, 999999))
        self.player_id = 0  # Human player is always player 0
        self.bots: list[BotAI] = []
        self.difficulty = difficulty

        # Create bot AIs for players 1-7
        for i in range(1, 8):
            strategy = random.choice(list(BotStrategy))
            bot = BotAI(
                self.game.players[i],
                difficulty=difficulty,
                strategy=strategy,
            )
            self.bots.append(bot)

        self.running = True

    @property
    def player(self) -> PlayerState:
        """Get the human player."""
        return self.game.players[self.player_id]

    def run(self):
        """Run the main game loop."""
        clear_screen()
        self.print_welcome()

        while self.running and not self.game.is_game_over():
            stage = self.game.get_stage_string()
            round_info = self.game.get_current_round_info()

            # Handle different round types
            if round_info.is_carousel:
                self.handle_carousel_round()
            elif round_info.is_augment:
                self.handle_augment_round()
            elif round_info.is_pve:
                self.handle_pve_round()
            else:
                self.handle_pvp_round()

            if not self.running:
                break

            # Check eliminations
            self.check_eliminations()

            # Advance to next round
            self.game.advance_round()

        if self.running:
            self.show_game_over()

    def print_welcome(self):
        """Print welcome message."""
        print_header("TFT Set 16: Lore & Legends", "★")
        print("\n  Welcome to Teamfight Tactics!")
        print(f"  You are Player 1 (ID: 0)")
        print(f"  Playing against 7 AI bots ({self.difficulty.value} difficulty)")
        print("\n  Commands during planning phase:")
        print("    [1-5]  - Buy champion from shop slot (goes to bench)")
        print("    [r]    - Reroll shop (2 gold)")
        print("    [e]    - Buy XP (4 gold)")
        print("    [d]    - Done / End turn")
        print("    [b]    - Show board")
        print("    [p]    - Show all players")
        print("    [s]    - Sell unit from bench")
        print("    [w]    - Place unit from bench to board")
        print("    [x]    - Move unit from board to bench")
        print("    [m]    - Move/swap units on board")
        print("    [i]    - Show items inventory")
        print("    [t]    - Equip item to champion")
        print("    [q]    - Quit game")
        print_divider()
        input("\nPress Enter to start...")

    def handle_carousel_round(self):
        """Handle carousel round."""
        stage = self.game.get_stage_string()
        clear_screen()
        print_header(f"CAROUSEL ROUND - {stage}")

        # Generate carousel
        carousel = self.game.start_carousel()
        pick_order = self.game.get_carousel_pick_order()

        print("\nCarousel Champions:")
        for i, champ in enumerate(carousel):
            status = "✓" if champ.is_available else "✗"
            print(f"  [{i+1}] {status} {champ.champion_name} ({champ.cost}-cost) + {champ.item_name}")

        # Process picks in order
        for pick_info in pick_order:
            pid = pick_info.player_id
            player = self.game.players[pid]

            if pid == self.player_id:
                # Human player picks
                while True:
                    try:
                        choice = input(f"\nYour pick (1-{len(carousel)}): ").strip()
                        if choice.lower() == 'q':
                            self.running = False
                            return
                        idx = int(choice) - 1
                        if 0 <= idx < len(carousel) and carousel[idx].is_available:
                            result = self.game.process_carousel_pick(pid, idx)
                            if result:
                                print(f"You picked: {result.champion_name} + {result.item_name}")
                            break
                        else:
                            print("Invalid choice or champion already taken!")
                    except ValueError:
                        print("Enter a number!")
            else:
                # Bot picks
                available_indices = [i for i, c in enumerate(carousel) if c.is_available]
                if available_indices:
                    bot = self.bots[pid - 1]
                    pick_idx = bot.choose_carousel_pick(carousel)
                    result = self.game.process_carousel_pick(pid, pick_idx)
                    if result:
                        print(f"Player {pid+1} picked: {result.champion_name} + {result.item_name}")

        input("\nPress Enter to continue...")

    def handle_augment_round(self):
        """Handle augment selection round."""
        stage = self.game.get_stage_string()
        clear_screen()
        print_header(f"AUGMENT SELECTION - {stage}")

        # Generate augment choices
        choices = self.game.start_augment_selection()

        # Show player's choices
        if self.player_id in choices:
            player_choice = choices[self.player_id]
            print("\nYour Augment Choices:")
            for i, aug in enumerate(player_choice.options):
                tier_symbol = {"silver": "🥈", "gold": "🥇", "prismatic": "💎"}.get(aug.tier.value, "")
                print(f"  [{i+1}] {tier_symbol} {aug.name} ({aug.tier.value.upper()})")
                print(f"      {aug.description}")
                print()

            while True:
                try:
                    choice = input("Choose augment (1-3): ").strip()
                    if choice.lower() == 'q':
                        self.running = False
                        return
                    idx = int(choice) - 1
                    if 0 <= idx < len(player_choice.options):
                        selected = self.game.select_augment(self.player_id, idx)
                        if selected:
                            print(f"\nYou selected: {selected.name}")
                        break
                    else:
                        print("Invalid choice!")
                except ValueError:
                    print("Enter a number!")

        # Bots choose augments
        for i, bot in enumerate(self.bots):
            pid = i + 1
            if pid in choices:
                aug_idx = bot.choose_augment(choices[pid])
                self.game.select_augment(pid, aug_idx)

        # Now do planning phase
        self.handle_planning_phase()

    def handle_pve_round(self):
        """Handle PvE round."""
        stage = self.game.get_stage_string()
        round_info = self.game.get_current_round_info()

        clear_screen()
        print_header(f"PvE ROUND - {stage}")
        print(f"Monster: {round_info.monster_type or 'Minions'}")

        # Planning phase first
        self.handle_planning_phase()

        if not self.running:
            return

        # Simulate combat (simplified - everyone wins PvE)
        print("\n--- COMBAT ---")
        print("Fighting monsters...")

        for player in self.game.get_alive_players():
            # Simulate PvE combat using actual combat engine
            result = self.game.pve_system.simulate_pve_combat_with_engine(
                stage, player.units, player.player_id
            )
            if result.won:
                loot_str = f"+{result.gold_gained}g"
                if result.items_gained:
                    loot_str += f", {len(result.items_gained)} items"
                if result.special_items_gained:
                    loot_str += f", {', '.join(result.special_items_gained)}"
                print(f"Player {player.player_id + 1}: Victory! ({loot_str})")
                player.gold += result.gold_gained
                player.record_win()
                # Add dropped items to inventory
                for item_id in result.items_gained:
                    item = player.items.get_item(item_id)
                    if item:
                        player.items.add_to_inventory(item)
                for item_id in result.special_items_gained:
                    item = player.items.get_item(item_id)
                    if item:
                        player.items.add_to_inventory(item)
            else:
                print(f"Player {player.player_id + 1}: Defeat! (-{result.damage_taken} HP)")
                player.record_loss(result.damage_taken)

        input("\nPress Enter to continue...")

    def handle_pvp_round(self):
        """Handle PvP round with real combat simulation."""
        stage = self.game.get_stage_string()
        current_stage = self.game.stage_manager.current_stage

        clear_screen()
        print_header(f"PvP ROUND - {stage}")

        # Planning phase
        self.handle_planning_phase()

        if not self.running:
            return

        # Simulate combat
        print("\n--- COMBAT ---")
        alive_players = self.game.get_alive_players()

        # Simple matchmaking: pair up players
        random.shuffle(alive_players)
        pairs = []
        for i in range(0, len(alive_players) - 1, 2):
            pairs.append((alive_players[i], alive_players[i + 1]))

        # Handle odd player (fights ghost army)
        if len(alive_players) % 2 == 1:
            odd_player = alive_players[-1]
            # Ghost army - simulate against a copy of a random opponent
            ghost_opponent = random.choice([p for p in alive_players if p != odd_player])

            if odd_player.units.get_board_count() == 0:
                # No units - auto lose
                damage = self.calculate_damage(current_stage, [(2, 1), (2, 1), (2, 1)])
                odd_player.record_loss(damage)
                print(f"Player {odd_player.player_id + 1} fought ghost army: Defeat! (-{damage} HP)")
            else:
                winner, loser, surviving = self.simulate_combat(odd_player, ghost_opponent)
                if winner == odd_player:
                    odd_player.record_win()
                    print(f"Player {odd_player.player_id + 1} fought ghost army: Victory!")
                else:
                    damage = self.calculate_damage(current_stage, surviving)
                    odd_player.record_loss(damage)
                    print(f"Player {odd_player.player_id + 1} fought ghost army: Defeat! (-{damage} HP)")

        # Simulate each matchup using CombatEngine
        for p1, p2 in pairs:
            # Handle empty boards
            p1_units = p1.units.get_board_count()
            p2_units = p2.units.get_board_count()

            if p1_units == 0 and p2_units == 0:
                # Both empty - draw, no damage
                p1.record_win()
                p2.record_win()
                print(f"  Player {p1.player_id + 1} vs Player {p2.player_id + 1}: Draw (no units)")
                continue
            elif p1_units == 0:
                # P1 has no units - auto lose
                winner, loser = p2, p1
                surviving = [(u.champion.cost, u.star_level) for u in p2.units.get_board_units()]
            elif p2_units == 0:
                # P2 has no units - auto lose
                winner, loser = p1, p2
                surviving = [(u.champion.cost, u.star_level) for u in p1.units.get_board_units()]
            else:
                # Real combat simulation
                winner, loser, surviving = self.simulate_combat(p1, p2)

            # Calculate damage using TFT rules
            damage = self.calculate_damage(current_stage, surviving)

            winner.record_win()
            loser.record_loss(damage)

            # Show result
            p1_mark = "★" if p1 == winner else ""
            p2_mark = "★" if p2 == winner else ""
            you_involved = p1.player_id == self.player_id or p2.player_id == self.player_id

            if you_involved:
                print(f"\n  Player {p1.player_id + 1}{p1_mark} vs Player {p2.player_id + 1}{p2_mark}")
                if winner.player_id == self.player_id:
                    print(f"    You WON! Dealt {damage} damage to Player {loser.player_id + 1}")
                    if surviving:
                        print(f"    ({len(surviving)} units survived)")
                else:
                    print(f"    You LOST! Took {damage} damage")
                    if surviving:
                        print(f"    (Enemy had {len(surviving)} units surviving)")
            else:
                print(f"  Player {p1.player_id + 1}{p1_mark} vs Player {p2.player_id + 1}{p2_mark} (-{damage} HP)")

        input("\nPress Enter to continue...")

    def calculate_board_strength(self, player: PlayerState) -> int:
        """Calculate approximate board strength."""
        strength = 0
        for unit in player.units.get_board_units():
            base = unit.champion.cost * 10
            star_mult = unit.star_level
            strength += base * star_mult
        strength += player.level * 5
        return strength

    def calculate_damage(
        self,
        stage: int,
        surviving_units: List[Tuple[int, int]],
    ) -> int:
        """
        Calculate damage dealt to loser using TFT rules.

        Args:
            stage: Current stage number.
            surviving_units: List of (cost, star_level) for surviving enemy units.

        Returns:
            Total damage to player.
        """
        # Base stage damage
        base_damage = BASE_STAGE_DAMAGE.get(stage, BASE_STAGE_DAMAGE.get(7, 17))

        # Unit damage based on cost and star level
        unit_damage = 0
        for cost, star_level in surviving_units:
            star_table = UNIT_DAMAGE_BY_STAR_AND_COST.get(
                star_level, UNIT_DAMAGE_BY_STAR_AND_COST[1]
            )
            unit_damage += star_table.get(cost, cost)

        return base_damage + unit_damage

    def simulate_combat(
        self,
        player1: PlayerState,
        player2: PlayerState,
    ) -> Tuple[PlayerState, PlayerState, List[Tuple[int, int]]]:
        """
        Simulate combat between two players using CombatEngine.

        Args:
            player1: First player.
            player2: Second player.

        Returns:
            Tuple of (winner, loser, surviving_units as [(cost, star_level), ...])
        """
        stage = self.game.stage_manager.current_stage

        # Create combat engine
        engine = CombatEngine(
            seed=random.randint(0, 999999),
            stage=stage,
        )

        # Build board dictionaries for setup_combat_from_boards
        # The engine expects {HexPosition: ChampionInstance}
        blue_board = {}
        for unit in player1.units.get_board_units():
            if unit.position:
                col, row = unit.position
                pos = HexPosition(row=row, col=col)
                blue_board[pos] = unit

        red_board = {}
        for unit in player2.units.get_board_units():
            if unit.position:
                col, row = unit.position
                pos = HexPosition(row=row, col=col)
                red_board[pos] = unit

        # Setup and run combat
        engine.setup_combat_from_boards(blue_board, red_board)
        result = engine.run_combat()

        # Determine winner based on result
        if result.winner == Team.BLUE:
            winner, loser = player1, player2
            # Get surviving blue units
            surviving = [
                (unit.cost, unit.star_level)
                for unit in engine.units.values()
                if unit.team == Team.BLUE and unit.is_alive
            ]
        elif result.winner == Team.RED:
            winner, loser = player2, player1
            # Get surviving red units
            surviving = [
                (unit.cost, unit.star_level)
                for unit in engine.units.values()
                if unit.team == Team.RED and unit.is_alive
            ]
        else:
            # Draw - random winner, minimal damage
            if random.random() < 0.5:
                winner, loser = player1, player2
            else:
                winner, loser = player2, player1
            surviving = []

        return winner, loser, surviving

    def handle_planning_phase(self):
        """Handle the planning phase where player can buy/sell/position."""
        # Note: Gold income is handled by advance_round() -> end_round_income()
        # Don't add passive_gold here to avoid double-counting
        stage = self.game.get_stage_string()

        # Refresh shops
        self.game.start_planning_phase()

        # Bots take their turns
        for bot in self.bots:
            if bot.player.is_alive:
                bot.take_turn(stage)

        # Human player's turn
        while True:
            self.show_player_status()
            self.show_shop()

            cmd = input("\nCommand: ").strip().lower()

            if cmd == 'q':
                self.running = False
                return
            elif cmd == 'd':
                break
            elif cmd == 'r':
                if self.player.reroll():
                    print("Shop rerolled!")
                else:
                    print("Not enough gold!")
            elif cmd == 'e':
                if self.player.buy_xp():
                    print(f"Bought XP! Level: {self.player.level}, XP: {self.player.xp}")
                else:
                    print("Not enough gold!")
            elif cmd == 'b':
                self.show_board()
            elif cmd == 'p':
                self.show_all_players()
            elif cmd == 's':
                self.handle_sell()
            elif cmd == 'w':
                self.handle_place_unit()
            elif cmd == 'x':
                self.handle_recall_unit()
            elif cmd == 'm':
                self.handle_move_unit()
            elif cmd == 'i':
                self.show_items()
            elif cmd == 't':
                self.handle_equip_item()
            elif cmd in ['1', '2', '3', '4', '5']:
                slot = int(cmd) - 1
                result = self.player.buy_champion(slot)
                if result:
                    print(f"Bought {result.champion.name}! (added to bench)")
                else:
                    champ = self.player.shop.get_slot(slot)
                    if champ is None:
                        print("Empty slot!")
                    elif not self.player.can_afford_champion(champ.cost):
                        print("Not enough gold!")
                    else:
                        print("Bench is full!")
            else:
                print("Unknown command. Try: 1-5, r, e, d, b, p, s, w, x, m, i, t, q")

    def show_player_status(self):
        """Show current player status."""
        stage = self.game.get_stage_string()
        p = self.player

        print_divider()
        print(f"  Stage: {stage}  |  Level: {p.level}  |  XP: {p.xp}/{self.get_xp_needed()}")
        print(f"  HP: {p.health}  |  Gold: {p.gold}  |  Streak: W{p.win_streak}/L{p.loss_streak}")
        print(f"  Board: {p.units.get_board_count()}/{p.level}  |  Bench: {p.units.get_bench_count()}/9")

        # Show traits
        traits = p.get_active_traits()
        if traits:
            trait_str = ", ".join([f"{t}({c})" for t, c in sorted(traits.items(), key=lambda x: -x[1])])
            print(f"  Traits: {trait_str}")
        print_divider()

    def get_xp_needed(self) -> int:
        """Get XP needed for next level."""
        from src.core.constants import LEVEL_XP
        next_level = self.player.level + 1
        if next_level > 10:
            return 0
        return LEVEL_XP.get(next_level, 0)

    def show_shop(self):
        """Show the shop."""
        print("\n  SHOP (2g to reroll):")
        for i in range(5):
            champ = self.player.shop.get_slot(i)
            if champ:
                owned = self.count_owned(champ.id)
                traits = ", ".join(champ.traits[:2])
                print(f"    [{i+1}] {champ.name} ({champ.cost}g) - {traits} [owned: {owned}]")
            else:
                print(f"    [{i+1}] ---")

    def count_owned(self, champion_id: str) -> int:
        """Count copies of a champion owned."""
        count = 0
        for unit in self.player.units.get_bench_units():
            if unit and unit.champion.id == champion_id:
                count += (1 if unit.star_level == 1 else 3 if unit.star_level == 2 else 9)
        for unit in self.player.units.get_board_units():
            if unit.champion.id == champion_id:
                count += (1 if unit.star_level == 1 else 3 if unit.star_level == 2 else 9)
        return count

    def show_board(self):
        """Show the player's board and bench."""
        print("\n  YOUR BOARD:")
        for row in range(4):
            row_str = f"    Row {row}: "
            units = []
            for col in range(7):
                unit = self.player.units.board.get((col, row))
                if unit:
                    star = "★" * unit.star_level
                    units.append(f"{unit.champion.name}{star}")
                else:
                    units.append("·")
            print(row_str + " | ".join(units))

        print("\n  BENCH:")
        bench_str = "    "
        for i, unit in enumerate(self.player.units.bench):
            if unit:
                star = "★" * unit.star_level
                bench_str += f"[{i+1}] {unit.champion.name}{star}  "
            else:
                bench_str += f"[{i+1}] ---  "
        print(bench_str)

        input("\nPress Enter to continue...")

    def show_all_players(self):
        """Show status of all players."""
        print("\n  ALL PLAYERS:")
        print("  " + "-" * 50)
        for i, player in enumerate(self.game.players):
            status = "ALIVE" if player.is_alive else "OUT"
            you = " (YOU)" if i == self.player_id else ""
            board = player.units.get_board_count()
            print(f"    P{i+1}{you}: HP={player.health:3d}  Lv={player.level}  "
                  f"Gold={player.gold:2d}  Units={board}  [{status}]")
        print("  " + "-" * 50)
        input("\nPress Enter to continue...")

    def handle_sell(self):
        """Handle selling a unit from bench."""
        print("\n  BENCH:")
        for i, unit in enumerate(self.player.units.bench):
            if unit:
                star = "★" * unit.star_level
                sell_value = unit.champion.cost * unit.star_level
                print(f"    [{i+1}] {unit.champion.name}{star} (sells for {sell_value}g)")
            else:
                print(f"    [{i+1}] ---")

        try:
            slot = input("\n  Sell which slot (1-9, or 'c' to cancel)? ").strip()
            if slot.lower() == 'c':
                return
            idx = int(slot) - 1
            if 0 <= idx < 9:
                gold = self.player.sell_bench_slot(idx)
                if gold > 0:
                    print(f"  Sold for {gold} gold!")
                else:
                    print("  Empty slot!")
            else:
                print("  Invalid slot!")
        except ValueError:
            print("  Invalid input!")

    def handle_place_unit(self):
        """Handle placing a unit from bench to board."""
        # Show bench
        print("\n  BENCH:")
        has_units = False
        for i, unit in enumerate(self.player.units.bench):
            if unit:
                star = "★" * unit.star_level
                print(f"    [{i+1}] {unit.champion.name}{star} ({unit.champion.cost}g)")
                has_units = True
            else:
                print(f"    [{i+1}] ---")

        if not has_units:
            print("\n  No units on bench!")
            return

        # Check board limit
        board_limit = self.player.get_board_size_limit()
        current_board = self.player.units.get_board_count()
        if current_board >= board_limit:
            print(f"\n  Board is full! ({current_board}/{board_limit})")
            print("  Level up or remove a unit first.")
            return

        try:
            bench_input = input("\n  Select bench slot (1-9, or 'c' to cancel): ").strip()
            if bench_input.lower() == 'c':
                return
            bench_idx = int(bench_input) - 1
            if not (0 <= bench_idx < 9):
                print("  Invalid bench slot!")
                return

            unit = self.player.units.bench[bench_idx]
            if unit is None:
                print("  Empty bench slot!")
                return

            # Show board with available positions
            print("\n  BOARD (select position):")
            print("       0   1   2   3   4   5   6")
            for row in range(4):
                row_str = f"  {row}:  "
                for col in range(7):
                    existing = self.player.units.board.get((col, row))
                    if existing:
                        row_str += f"[{existing.champion.name[:3]}]"
                    else:
                        row_str += " ·  "
                print(row_str)

            pos_input = input("\n  Enter position (col,row like '3,1' or 'c' to cancel): ").strip()
            if pos_input.lower() == 'c':
                return

            parts = pos_input.replace(' ', '').split(',')
            if len(parts) != 2:
                print("  Invalid format! Use 'col,row' like '3,1'")
                return

            col, row = int(parts[0]), int(parts[1])
            if not (0 <= col < 7 and 0 <= row < 4):
                print("  Position out of bounds! (col: 0-6, row: 0-3)")
                return

            # Check if position is occupied
            if (col, row) in self.player.units.board:
                existing = self.player.units.board[(col, row)]
                print(f"  Position occupied by {existing.champion.name}!")
                swap = input("  Swap positions? (y/n): ").strip().lower()
                if swap == 'y':
                    # Swap: move existing to bench, place new unit
                    self.player.units.bench[bench_idx] = existing
                    existing.position = None
                    self.player.units.board[(col, row)] = unit
                    unit.position = (col, row)
                    print(f"  Swapped! {unit.champion.name} placed, {existing.champion.name} to bench.")
                return

            # Place unit
            self.player.units.bench[bench_idx] = None
            self.player.units.board[(col, row)] = unit
            unit.position = (col, row)
            print(f"  Placed {unit.champion.name} at ({col}, {row})!")

        except ValueError:
            print("  Invalid input!")

    def handle_recall_unit(self):
        """Handle moving a unit from board back to bench."""
        # Check if bench has space
        bench_count = sum(1 for u in self.player.units.bench if u is not None)
        if bench_count >= 9:
            print("\n  Bench is full! Sell a unit first.")
            return

        # Show board
        print("\n  BOARD:")
        print("       0   1   2   3   4   5   6")
        has_units = False
        for row in range(4):
            row_str = f"  {row}:  "
            for col in range(7):
                unit = self.player.units.board.get((col, row))
                if unit:
                    star = "★" * unit.star_level
                    row_str += f"[{unit.champion.name[:3]}{star}]"
                    has_units = True
                else:
                    row_str += " ·  "
            print(row_str)

        if not has_units:
            print("\n  No units on board!")
            return

        try:
            pos_input = input("\n  Enter position to recall (col,row like '3,1' or 'c' to cancel): ").strip()
            if pos_input.lower() == 'c':
                return

            parts = pos_input.replace(' ', '').split(',')
            if len(parts) != 2:
                print("  Invalid format! Use 'col,row' like '3,1'")
                return

            col, row = int(parts[0]), int(parts[1])
            if not (0 <= col < 7 and 0 <= row < 4):
                print("  Position out of bounds! (col: 0-6, row: 0-3)")
                return

            unit = self.player.units.board.get((col, row))
            if unit is None:
                print("  No unit at that position!")
                return

            # Find empty bench slot
            bench_idx = None
            for i, slot in enumerate(self.player.units.bench):
                if slot is None:
                    bench_idx = i
                    break

            if bench_idx is None:
                print("  Bench is full!")
                return

            # Move unit to bench
            del self.player.units.board[(col, row)]
            self.player.units.bench[bench_idx] = unit
            unit.position = None
            print(f"  {unit.champion.name} moved to bench slot {bench_idx + 1}!")

        except ValueError:
            print("  Invalid input!")

    def handle_move_unit(self):
        """Handle moving a unit to another board position."""
        # Show board
        print("\n  BOARD:")
        print("       0   1   2   3   4   5   6")
        has_units = False
        for row in range(4):
            row_str = f"  {row}:  "
            for col in range(7):
                unit = self.player.units.board.get((col, row))
                if unit:
                    star = "★" * unit.star_level
                    row_str += f"[{unit.champion.name[:3]}{star}]"
                    has_units = True
                else:
                    row_str += " ·  "
            print(row_str)

        if not has_units:
            print("\n  No units on board!")
            return

        try:
            # Select unit to move
            from_input = input("\n  Select unit to move (col,row or 'c' to cancel): ").strip()
            if from_input.lower() == 'c':
                return

            parts = from_input.replace(' ', '').split(',')
            if len(parts) != 2:
                print("  Invalid format! Use 'col,row' like '3,1'")
                return

            from_col, from_row = int(parts[0]), int(parts[1])
            if not (0 <= from_col < 7 and 0 <= from_row < 4):
                print("  Position out of bounds!")
                return

            unit = self.player.units.board.get((from_col, from_row))
            if unit is None:
                print("  No unit at that position!")
                return

            # Select destination
            to_input = input(f"  Move {unit.champion.name} to (col,row or 'c' to cancel): ").strip()
            if to_input.lower() == 'c':
                return

            parts = to_input.replace(' ', '').split(',')
            if len(parts) != 2:
                print("  Invalid format! Use 'col,row' like '3,1'")
                return

            to_col, to_row = int(parts[0]), int(parts[1])
            if not (0 <= to_col < 7 and 0 <= to_row < 4):
                print("  Position out of bounds!")
                return

            # Check if destination is occupied
            target_unit = self.player.units.board.get((to_col, to_row))
            if target_unit:
                # Swap positions
                self.player.units.board[(from_col, from_row)] = target_unit
                target_unit.position = (from_col, from_row)
                self.player.units.board[(to_col, to_row)] = unit
                unit.position = (to_col, to_row)
                print(f"  Swapped {unit.champion.name} and {target_unit.champion.name}!")
            else:
                # Move to empty position
                del self.player.units.board[(from_col, from_row)]
                self.player.units.board[(to_col, to_row)] = unit
                unit.position = (to_col, to_row)
                print(f"  Moved {unit.champion.name} to ({to_col}, {to_row})!")

        except ValueError:
            print("  Invalid input!")

    def show_items(self):
        """Show player's item inventory and equipped items."""
        print("\n  ITEM INVENTORY:")
        inventory = self.player.items.inventory
        if not inventory:
            print("    No items in inventory")
        else:
            for i, item_inst in enumerate(inventory):
                item = item_inst.item
                stats_str = self._format_item_stats(item)
                # Handle both enum and string types
                item_type = item.type.value if hasattr(item.type, 'value') else item.type
                print(f"    [{i+1}] {item.name} ({item_type}) - {stats_str}")
                if item.effect:
                    print(f"        Effect: {item.effect}")

        # Show possible recipes
        recipes = self.player.items.get_available_recipes()
        if recipes:
            print("\n  AVAILABLE RECIPES:")
            for comp1, comp2, result in recipes:
                print(f"    {comp1.name} + {comp2.name} = {result.name}")

        # Show equipped items on champions
        print("\n  EQUIPPED ITEMS:")
        has_equipped = False
        for unit in self.player.units.get_board_units():
            if unit.items:
                has_equipped = True
                items_str = ", ".join([item.item.name for item in unit.items])
                print(f"    {unit.champion.name}: {items_str}")
        for unit in self.player.units.get_bench_units():
            if unit and unit.items:
                has_equipped = True
                items_str = ", ".join([item.item.name for item in unit.items])
                print(f"    {unit.champion.name} (bench): {items_str}")
        if not has_equipped:
            print("    No items equipped")

        input("\nPress Enter to continue...")

    def _format_item_stats(self, item) -> str:
        """Format item stats as a string."""
        stats = item.stats
        parts = []
        if stats.ad: parts.append(f"+{stats.ad} AD")
        if stats.ap: parts.append(f"+{stats.ap} AP")
        if stats.armor: parts.append(f"+{stats.armor} Armor")
        if stats.mr: parts.append(f"+{stats.mr} MR")
        if stats.health: parts.append(f"+{stats.health} HP")
        if stats.mana: parts.append(f"+{stats.mana} Mana")
        if stats.attack_speed: parts.append(f"+{stats.attack_speed}% AS")
        if stats.crit_chance: parts.append(f"+{stats.crit_chance}% Crit")
        if stats.crit_damage: parts.append(f"+{stats.crit_damage}% Crit DMG")
        if stats.omnivamp: parts.append(f"+{stats.omnivamp}% Omnivamp")
        return ", ".join(parts) if parts else "No stats"

    def handle_equip_item(self):
        """Handle equipping an item to a champion."""
        inventory = self.player.items.inventory
        if not inventory:
            print("\n  No items in inventory!")
            return

        # Show inventory
        print("\n  INVENTORY:")
        for i, item_inst in enumerate(inventory):
            item = item_inst.item
            item_type = item.type.value if hasattr(item.type, 'value') else item.type
            print(f"    [{i+1}] {item.name} ({item_type})")

        try:
            item_input = input("\n  Select item (1-{}, or 'c' to cancel): ".format(len(inventory))).strip()
            if item_input.lower() == 'c':
                return
            item_idx = int(item_input) - 1
            if not (0 <= item_idx < len(inventory)):
                print("  Invalid item!")
                return

            selected_item = inventory[item_idx]

            # Show champions that can equip
            print("\n  SELECT CHAMPION:")
            champions = []

            # Board units
            for unit in self.player.units.get_board_units():
                can_equip = len(unit.items) < 3
                items_str = f" [{len(unit.items)}/3 items]" if unit.items else " [0/3 items]"
                status = "" if can_equip else " (FULL)"
                champions.append((unit, "board"))
                print(f"    [{len(champions)}] {unit.champion.name}{items_str}{status}")

            # Bench units
            for i, unit in enumerate(self.player.units.bench):
                if unit:
                    can_equip = len(unit.items) < 3
                    items_str = f" [{len(unit.items)}/3 items]" if unit.items else " [0/3 items]"
                    status = "" if can_equip else " (FULL)"
                    champions.append((unit, "bench"))
                    print(f"    [{len(champions)}] {unit.champion.name} (bench){items_str}{status}")

            if not champions:
                print("  No champions to equip!")
                return

            champ_input = input("\n  Select champion (1-{}, or 'c' to cancel): ".format(len(champions))).strip()
            if champ_input.lower() == 'c':
                return
            champ_idx = int(champ_input) - 1
            if not (0 <= champ_idx < len(champions)):
                print("  Invalid champion!")
                return

            champion, location = champions[champ_idx]

            # Equip the item
            success = self.player.items.equip_item(selected_item, champion)
            if success:
                # Check if item was auto-combined
                if selected_item.is_component:
                    last_item = champion.items[-1] if champion.items else None
                    if last_item and last_item.is_combined:
                        print(f"  Combined! {champion.champion.name} equipped {last_item.item.name}!")
                    else:
                        print(f"  {champion.champion.name} equipped {selected_item.item.name}!")
                else:
                    print(f"  {champion.champion.name} equipped {selected_item.item.name}!")
            else:
                print("  Failed to equip item! (Champion may be full)")

        except ValueError:
            print("  Invalid input!")

    def check_eliminations(self):
        """Check for and handle player eliminations."""
        for player in self.game.players:
            if player.health <= 0 and player.is_alive:
                self.game.eliminate_player(player.player_id)
                if player.player_id == self.player_id:
                    print(f"\n💀 YOU HAVE BEEN ELIMINATED! 💀")
                    placement = len(self.game.eliminated)
                    print(f"   Final Placement: #{9 - placement}")
                else:
                    print(f"\n  Player {player.player_id + 1} has been eliminated!")

    def show_game_over(self):
        """Show game over screen."""
        clear_screen()
        print_header("GAME OVER", "★")

        placements = self.game.get_placements()
        print("\n  FINAL STANDINGS:")
        print("  " + "-" * 40)

        for i, player in enumerate(placements):
            you = " (YOU)" if player.player_id == self.player_id else ""
            medals = {0: "🥇", 1: "🥈", 2: "🥉"}.get(i, "  ")
            print(f"    {medals} #{i+1}: Player {player.player_id + 1}{you} - "
                  f"Level {player.level}, {player.rounds_played} rounds")

        print("  " + "-" * 40)

        # Check if human player won
        winner = self.game.get_winner()
        if winner and winner.player_id == self.player_id:
            print("\n  🎉 CONGRATULATIONS! YOU WON! 🎉")
        else:
            my_placement = next(
                i + 1 for i, p in enumerate(placements)
                if p.player_id == self.player_id
            )
            if my_placement <= 4:
                print(f"\n  Good game! Top {my_placement} finish!")
            else:
                print(f"\n  Better luck next time! (#{my_placement})")

        print()


def main():
    """Main entry point."""
    print("\n=== TFT Set 16: Lore & Legends ===\n")
    print("Select difficulty:")
    print("  [1] Easy")
    print("  [2] Medium")
    print("  [3] Hard")

    while True:
        choice = input("\nDifficulty (1-3): ").strip()
        if choice == '1':
            difficulty = BotDifficulty.EASY
            break
        elif choice == '2':
            difficulty = BotDifficulty.MEDIUM
            break
        elif choice == '3':
            difficulty = BotDifficulty.HARD
            break
        else:
            print("Enter 1, 2, or 3")

    game = InteractiveGame(difficulty)
    game.run()


if __name__ == "__main__":
    main()
