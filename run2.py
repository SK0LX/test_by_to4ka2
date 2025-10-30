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
        target = find_target_gateway(graph, virus_pos, targets)
        if not target:
            break
        critical_edges = find_critical_edges(graph, virus_pos, target)
        if not critical_edges:
            break
        edge_to_remove = min(critical_edges)
        result.append(edge_to_remove)
        delete_edge, node = edge_to_remove.split('-')
        remove_edge(graph, delete_edge, node)
        if not graph[delete_edge]:
            targets.discard(delete_edge)
        virus_pos = find_next_step(graph, virus_pos, targets)
    return result

def build_graph(edges):
    graph = {}
    for node1, node2 in edges:
        graph.setdefault(node1, []).append(node2)
        graph.setdefault(node2, []).append(node1)
    return graph

def remove_edge(graph, node1, node2):
    if node1 in graph and node2 in graph[node1]:
        graph[node1].remove(node2)
    if node2 in graph and node1 in graph[node2]:
        graph[node2].remove(node1)

def find_target_gateway(graph, virus_pos, targets):
    if not targets:
        return None
    visited = set([virus_pos])
    queue = deque([(virus_pos, 0)])
    best_gateway = None
    best_distance = float('inf')
    while queue:
        current, distance = queue.popleft()

        for neighbor in sorted(graph.get(current, [])):
            if neighbor in visited:
                continue
            if neighbor in targets:
                if distance + 1 < best_distance or (distance + 1 == best_distance and neighbor < best_gateway):
                    best_gateway = neighbor
                    best_distance = distance + 1
            else:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    return best_gateway

def find_critical_edges(graph, virus_pos, target):
    dist = {}
    prev = {}
    queue = deque([virus_pos])
    dist[virus_pos] = 0
    prev[virus_pos] = None
    while queue:
        current = queue.popleft()
        for neighbor in graph.get(current, []):
            if neighbor not in dist:
                dist[neighbor] = dist[current] + 1
                prev[neighbor] = current
                queue.append(neighbor)
    if target not in dist:
        return set()
    critical_edges = []
    for node in sorted(graph[target]):
        if node in dist and dist[node] + 1 == dist[target]:
            critical_edges.append(f"{target}-{node}")
    critical_edges.sort()
    return critical_edges


def find_next_step(graph, virus_pos, targets):
    if not targets:
        return None
    target_gateway = find_target_gateway(graph, virus_pos, targets)
    if not target_gateway:
        return None
    visited = set([virus_pos])
    queue = deque([virus_pos])
    prev = {}
    while queue:
        current = queue.popleft()
        for neighbor in sorted(graph.get(current, [])):
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
                    return path[-1]

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
