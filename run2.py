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
    targets = {node for node in graph if node.isupper()}
    result = []
    virus_pos = 'a'
    while targets and virus_pos:
        edges_list = bfs_from_targets(graph, targets, virus_pos)
        if not edges_list:
            break
        edge_to_remove = edges_list[0]
        result.append(edge_to_remove)
        gateway, node = edge_to_remove.split('-')
        target = gateway
        graph[gateway].remove(node)
        graph[node].remove(gateway)
        next_virus_pos = find_next_step(graph, virus_pos, target)
        if not graph[gateway]:
            targets.discard(gateway)
        virus_pos = next_virus_pos
    return result

def build_graph(edges):
    graph = {}
    for node1, node2 in edges:
        graph.setdefault(node1, []).append(node2)
        graph.setdefault(node2, []).append(node1)
    return graph

def bfs_from_targets(graph, targets, virus_pos):
    queue = deque()
    visited = {}
    queue.append(virus_pos)
    visited[virus_pos] = 0
    answer = set()
    while queue:
        current = queue.popleft()
        current_distance = visited[current]
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited[neighbor] = current_distance + 1
                queue.append(neighbor)
            if neighbor in targets:
                for node in graph[neighbor]:
                    if node not in targets and node in graph[neighbor]:
                        answer.add((visited[node], f"{neighbor}-{node}"))
    sorted_answer = sorted(answer)
    return [edge for dist, edge in sorted_answer]


def find_next_step(graph, virus_pos, target_gateway):
    if target_gateway in graph[virus_pos]:
        return None
    visited = {virus_pos}
    queue = deque([virus_pos])
    prev = {}
    while queue:
        current = queue.popleft()
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                prev[neighbor] = current
                queue.append(neighbor)
                if neighbor == target_gateway:
                    path = []
                    node = neighbor
                    while node != virus_pos:
                        path.append(node)
                        node = prev[node]
                    return path[-1] if path else None

    return None

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
