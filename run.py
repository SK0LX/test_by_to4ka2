import sys
from pprint import pprint
import heapq
from collections import defaultdict
from typing import Any

class Solver:
    def __init__(self, depth = 2):
        self.ROOM_POSITIONS = {'A': 2, 'B': 4, 'C': 6, 'D': 8}
        self.ROOM_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        self.COST = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
        self.TARGET_ROOMS = ['A', 'B', 'C', 'D']
        self.DEPTH = depth

    def check_current_state(self, rooms) -> bool:
        for i, room_type in enumerate(self.TARGET_ROOMS):
            if len(rooms[i]) != self.DEPTH or any(c != room_type for c in rooms[i]):
                return False
        return True


    def can_enter_room(self, room, room_type) -> bool:
        """Можно ли войти в комнату"""
        return all(c == '.' or c == room_type for c in room)


    def get_room_depth(self, room) -> int:
        """Найти самую глубокую свободную позицию в комнате"""
        for depth in range(len(room)):
            if room[depth] == '.':
                return depth
        return -1


    def is_path_clear(self, hallway, start, end) -> bool:
        """Проверка свободного пути в коридоре"""
        if start == end:
            return True
        step = 1 if start < end else -1
        for pos in range(start + step, end + step, step):
            if hallway[pos] != '.':
                return False
        return True


    def heuristic(self, hallway, rooms):
        """Минимальная оценка стоимости до цели"""
        total_cost = 0
        total_cost += self.count_cost_if_patch_is_free(hallway)
        total_cost += self.count_cost_if_typeNode_is_not_current(rooms)
        return total_cost


    def count_cost_if_typeNode_is_not_current(self, rooms) -> int:
        cost = 0
        for room_idx, room in enumerate(rooms):
            for depth, type_node in enumerate(room):
                if type_node != '.' and self.ROOM_INDEX[type_node] != room_idx:
                    current_room_pos = self.ROOM_POSITIONS[self.TARGET_ROOMS[room_idx]]
                    target_room_pos = self.ROOM_POSITIONS[type_node]
                    steps = (depth + 1) + abs(current_room_pos - target_room_pos) + 1
                    cost += steps * self.COST[type_node]
        return cost


    def count_cost_if_patch_is_free(self, hallway) -> int:
        cost = 0
        for pos, type_node in enumerate(hallway):
            if type_node != '.':
                room_pos = self.ROOM_POSITIONS[type_node]
                count_steps = abs(pos - room_pos)
                cost += count_steps * self.COST[type_node]
        return cost


    def generate_hallway_to_room_moves(self, hallway, rooms):
        """
        Данный Ход возможен, когда наша буква(амфипод) в коридоре и
        она напрямую может войти в свою команту(ей ничего не мешает)
        :param hallway:
        :param rooms:
        :return moves:
        """
        moves = []
        for pos, type_node in enumerate(hallway):
            if type_node != '.':
                room_idx = self.ROOM_INDEX[type_node]
                room = rooms[room_idx]
                room_pos = self.ROOM_POSITIONS[type_node]

                if self.can_enter_room(room, type_node) and self.is_path_clear(hallway, pos, room_pos):
                    depth = self.get_room_depth(room)
                    if depth >= 0:
                        self.regenerate_hallway_and_typeNode_go_home(depth, hallway, moves, pos, room_idx, room_pos, rooms,
                                                                type_node)
        return moves


    def regenerate_hallway_and_typeNode_go_home(self, depth: int, hallway, moves: list[Any], pos: int, room_idx: int,
                                                room_pos: int, rooms, type_node):
        """
        просто перегенерируем коридор с нашим ходом(когда буква идет из коридора домой)
        :param depth:
        :param hallway:
        :param moves:
        :param pos:
        :param room_idx:
        :param room_pos:
        :param rooms:
        :param type_node:
        :return:
        """
        steps = abs(pos - room_pos) + (depth + 1)
        cost = steps * self.COST[type_node]
        new_hallway = list(hallway)
        new_hallway[pos] = '.'
        new_rooms = list(rooms)
        new_room = list(new_rooms[room_idx])
        new_room[depth] = type_node
        new_rooms[room_idx] = tuple(new_room)
        moves.append((cost, tuple(new_hallway), tuple(new_rooms)))


    def generate_room_to_hallway_moves(self, hallway, rooms):
        moves = []

        for room_idx, room in enumerate(rooms):
            depth = 0
            while depth < len(room) and room[depth] == '.':
                depth += 1

            if depth < len(room):
                type_node = room[depth]
                room_pos = self.ROOM_POSITIONS[self.TARGET_ROOMS[room_idx]]
                should_move = (self.ROOM_INDEX[type_node] != room_idx or
                               not self.can_enter_room(room, type_node))

                if should_move:
                    for pos in range(len(hallway)):
                        if pos not in self.ROOM_POSITIONS.values() and hallway[pos] == '.':
                            if self.is_path_clear(hallway, room_pos, pos):
                                self.regenerare_hallway_and_tpeNode_go_in_hallway(type_node, depth,
                                                                                  hallway, moves,
                                                                                  pos, room_idx,
                                                                                  room_pos, rooms)

        return moves


    def regenerare_hallway_and_tpeNode_go_in_hallway(self, type_node, depth: int, hallway, moves: list[Any], pos: int, room_idx: int,
                                                     room_pos: int, rooms):
        """
        Просто перегенерируем коридор, когда наша буква идет из команты в коридор
        :param type_node:
        :param depth:
        :param hallway:
        :param moves:
        :param pos:
        :param room_idx:
        :param room_pos:
        :param rooms:
        :return:
        """
        steps = (depth + 1) + abs(room_pos - pos)
        cost = steps * self.COST[type_node]
        new_hallway = list(hallway)
        new_hallway[pos] = type_node
        new_rooms = list(rooms)
        new_room = list(new_rooms[room_idx])
        new_room[depth] = '.'
        new_rooms[room_idx] = tuple(new_room)
        moves.append((cost, tuple(new_hallway), tuple(new_rooms)))


    def generate_moves(self, hallway, rooms) -> int:
        """
        сам путь
        :param hallway:
        :param rooms:
        :return moves:
        """
        moves = []
        moves.extend(self.generate_hallway_to_room_moves(hallway, rooms))
        moves.extend(self.generate_room_to_hallway_moves(hallway, rooms))

        return moves


    def A_star(self, initial_hallway, initial_rooms):
        """
        Алгоритм А*
        Можно еще и Дейкстра использовать, но
        :param initial_hallway:
        :param initial_rooms:
        :return:
        """
        open_set = []
        heapq.heappush(open_set, (0, 0, initial_hallway, initial_rooms))

        g_costs = defaultdict(lambda: float('inf'))
        g_costs[(initial_hallway, initial_rooms)] = 0

        while open_set:
            f_cost, current_cost, hallway, rooms = heapq.heappop(open_set)

            if self.check_current_state(rooms):
                return current_cost

            if current_cost > g_costs[(hallway, rooms)]:
                continue

            for move_cost, new_hallway, new_rooms in self.generate_moves(hallway, rooms):
                new_cost = current_cost + move_cost
                new_state = (new_hallway, new_rooms)

                if new_cost < g_costs[new_state]:
                    g_costs[new_state] = new_cost
                    f_cost = new_cost + self.heuristic(new_hallway, new_rooms)
                    heapq.heappush(open_set, (f_cost, new_cost, new_hallway, new_rooms))

        return -1


def solve(lines: list[str]) -> int:
    """
    Решение задачи о сортировке в лабиринте

    Args:
        lines: список строк, представляющих лабиринт

    Returns:
        минимальная энергия для достижения целевой конфигурации
    """
    hallway = lines[1][1:12]
    room_depth = len(lines) - 3
    rooms = []
    for room_idx in range(4):
        room = []
        for depth in range(room_depth):
            line_idx = 2 + depth
            char_pos = 3 + 2 * room_idx
            room.append(lines[line_idx][char_pos])
        rooms.append(tuple(room))

    solver = Solver(room_depth)
    return solver.A_star(tuple(hallway), tuple(rooms))


def main():
    lines = []
    for line in sys.stdin:
        lines.append(line.rstrip('\n'))
    result = solve(lines)
    print(result)

if __name__ == "__main__":
    main()
