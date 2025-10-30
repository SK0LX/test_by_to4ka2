import sys
from collections import deque


def solve(edges: list[tuple[str, str]]) -> list[str]:
    """
    Решение задачи об изоляции вируса

    Args:
        edges: список коридоров в формате (узел1, узел2)

    Returns:
        список отключаемых коридоров в формате "Шлюз-узел"
    """
    graph = build_graph(edges)
    targets = {n for n in graph if n.isupper()}
    result = []
    virus_pos = 'a'

    while True:
        target = find_target_gateway(graph, virus_pos, targets)
        if not target:
            break
        for neighbor in sorted(graph[virus_pos]):
            if neighbor == target:
                edge = f"{target}-{virus_pos}"
                result.append(edge)
                remove_edge(graph, target, virus_pos)
                if not graph[target]:
                    targets.discard(target)
                break

        else:
            path = find_path(graph, virus_pos, target)
            if not path:
                break
            prev = path[-2]
            edge = f"{target}-{prev}"
            result.append(edge)
            remove_edge(graph, target, prev)
            if not graph[target]:
                targets.discard(target)
        virus_pos = find_next_step(graph, virus_pos, targets)
        if not virus_pos:
            break
    return result

def build_graph(edges):
    graph = {}
    for a, b in edges:
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    for v in graph:
        graph[v].sort()
    return graph

def remove_edge(graph, a, b):
    if b in graph.get(a, []):
        graph[a].remove(b)
    if a in graph.get(b, []):
        graph[b].remove(a)

def find_path(graph, start, goal):
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        for neighbor in sorted(graph.get(node, [])):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None

def find_target_gateway(graph, start, targets):
    queue = deque([(start, 0)])
    visited = {start}
    best = None
    best_dist = float('inf')
    while queue:
        node, d = queue.popleft()
        for neighbor in sorted(graph.get(node, [])):
            if neighbor in visited:
                continue
            if neighbor in targets:
                if d + 1 < best_dist or (d + 1 == best_dist and neighbor < best):
                    best = neighbor
                    best_dist = d + 1
            else:
                visited.add(neighbor)
                queue.append((neighbor, d + 1))
    return best

def find_next_step(graph, start, targets):
    target = find_target_gateway(graph, start, targets)
    if not target:
        return None
    path = find_path(graph, start, target)
    if not path or len(path) < 2:
        return None
    return path[1]

def main():
    edges = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            node1, sep, node2 = line.partition('-')
            if sep:
                edges.append((node1, node2))

    result = solve(edges)
    for edge in result:
        print(edge)


if __name__ == "__main__":
    main()
