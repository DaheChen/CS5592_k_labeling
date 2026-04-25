from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import argparse
import math
import random
import sys
import time


Vertex = int
Edge = Tuple[int, int]


@dataclass
class Result:
    graph_name: str
    algorithm: str
    labels: Dict[Vertex, int]
    edge_weights: Dict[Edge, int]
    k: int
    lower_bound: int
    gap: int
    gap_ratio: float
    valid: bool
    runtime: float
    status: str
    best_strategy: str


class CompleteTripartiteGraph:
    """
    K_{a,b,c} stored by implicit partitions.

    The graph is not stored as an adjacency matrix. Edges are generated from
    partition membership: vertices in different parts are adjacent, vertices in
    the same part are not adjacent.
    """

    def __init__(self, a: int, b: int, c: int):
        if min(a, b, c) <= 0:
            raise ValueError("a, b, c must be positive integers.")

        self.a = a
        self.b = b
        self.c = c
        self.name = f"K_{{{a},{b},{c}}}"
        self.parts = [
            list(range(0, a)),
            list(range(a, a + b)),
            list(range(a + b, a + b + c)),
        ]
        self.part_sizes = [a, b, c]
        self.part_id = {}
        for pid, part in enumerate(self.parts):
            for v in part:
                self.part_id[v] = pid

        self.V = a + b + c
        self.E = a * b + a * c + b * c
        self.delta = max(b + c, a + c, a + b)
        self._validate_topology()

    def vertices(self) -> List[Vertex]:
        return list(range(self.V))

    def edges(self) -> Iterable[Edge]:
        for left in range(3):
            for right in range(left + 1, 3):
                for u in self.parts[left]:
                    for v in self.parts[right]:
                        yield (u, v)

    def neighbors(self, v: Vertex) -> List[Vertex]:
        pid = self.part_id[v]
        out = []
        for other_pid, part in enumerate(self.parts):
            if other_pid != pid:
                out.extend(part)
        return out

    def _validate_topology(self) -> None:
        assert self.V == sum(self.part_sizes)
        assert self.E == self.a * self.b + self.a * self.c + self.b * self.c
        assert self.delta == max(self.b + self.c, self.a + self.c, self.a + self.b)
        for v in self.vertices():
            expected_degree = self.V - self.part_sizes[self.part_id[v]]
            assert len(self.neighbors(v)) == expected_degree


def lower_bound(graph: CompleteTripartiteGraph) -> int:
    return max(math.ceil((graph.E + 1) / 2), graph.delta)


def validate_labeling(graph: CompleteTripartiteGraph, labels: Dict[Vertex, int], store: bool = True):
    if len(labels) != graph.V:
        return False, {}, "Missing labels."
    if min(labels.values()) < 1:
        return False, {}, "Labels must be positive."

    used = set()
    edge_weights = {}
    for u, v in graph.edges():
        w = labels[u] + labels[v]
        if w in used:
            return False, edge_weights, f"Duplicate edge weight found: {w}"
        used.add(w)
        if store:
            edge_weights[(u, v)] = w

    if len(used) != graph.E:
        return False, edge_weights, "Unique edge count does not match E."
    return True, edge_weights, "All edge weights are unique."


def candidate_orders(graph: CompleteTripartiteGraph, restarts: int) -> List[Tuple[str, List[Vertex]]]:
    orders = [
        ("degree_desc", sorted(graph.vertices(), key=lambda v: len(graph.neighbors(v)), reverse=True)),
        ("part_abc", graph.parts[0] + graph.parts[1] + graph.parts[2]),
        ("part_cba", graph.parts[2] + graph.parts[1] + graph.parts[0]),
    ]

    interleaved = []
    for i in range(max(graph.part_sizes)):
        for part in graph.parts:
            if i < len(part):
                interleaved.append(part[i])
    orders.append(("part_interleaved", interleaved))

    for seed in range(restarts):
        rng = random.Random(31000 + seed)
        order = graph.vertices()
        rng.shuffle(order)
        orders.append((f"random_seed_{31000 + seed}", order))
    return orders


def greedy_one_order(graph: CompleteTripartiteGraph, order: List[Vertex], label_cap: int):
    labels: Dict[Vertex, int] = {}
    used_weights = set()

    for v in order:
        labeled_neighbors = [u for u in graph.neighbors(v) if u in labels]
        chosen = None
        for x in range(1, label_cap + 1):
            local = set()
            ok = True
            for u in labeled_neighbors:
                w = x + labels[u]
                if w in used_weights or w in local:
                    ok = False
                    break
                local.add(w)
            if ok:
                chosen = x
                break
        if chosen is None:
            return None
        labels[v] = chosen
        for u in labeled_neighbors:
            used_weights.add(chosen + labels[u])

    valid, _, _ = validate_labeling(graph, labels, store=False)
    return labels if valid else None


def exact_search(graph: CompleteTripartiteGraph, cap: int, seconds: float):
    deadline = time.time() + seconds
    order = sorted(graph.vertices(), key=lambda v: len(graph.neighbors(v)), reverse=True)
    labels: Dict[Vertex, int] = {}
    used_weights = set()

    def dfs(pos: int):
        if time.time() > deadline:
            return None
        if pos == len(order):
            valid, _, _ = validate_labeling(graph, labels, store=False)
            return dict(labels) if valid else None

        v = order[pos]
        labeled_neighbors = [u for u in graph.neighbors(v) if u in labels]
        for x in range(1, cap + 1):
            local = set()
            new_weights = []
            ok = True
            for u in labeled_neighbors:
                w = x + labels[u]
                if w in used_weights or w in local:
                    ok = False
                    break
                local.add(w)
                new_weights.append(w)
            if not ok:
                continue
            labels[v] = x
            for w in new_weights:
                used_weights.add(w)
            found = dfs(pos + 1)
            if found is not None:
                return found
            for w in new_weights:
                used_weights.remove(w)
            del labels[v]
        return None

    return dfs(0)


def cap_schedule(lb: int) -> List[int]:
    caps = []
    for multiplier in [1.0, 1.03, 1.06, 1.1, 1.2, 1.35, 1.5, 2.0]:
        cap = max(lb, math.ceil(lb * multiplier))
        if cap not in caps:
            caps.append(cap)
    return caps


def local_repair_search(
    graph: CompleteTripartiteGraph,
    cap: int,
    attempts: int = 4,
    iterations: int = 700,
    max_candidates: int = 100,
    time_limit: float = 6.0,
):
    deadline = time.time() + time_limit
    vertices = graph.vertices()
    edges = list(graph.edges())
    incident = {v: [] for v in vertices}
    for idx, (u, v) in enumerate(edges):
        incident[u].append(idx)
        incident[v].append(idx)

    for attempt in range(attempts):
        if time.time() > deadline:
            return None
        rng = random.Random(73000 + attempt + cap * 997)
        labels = {v: rng.randint(1, cap) for v in vertices}
        edge_weights = []
        counts = {}
        for u, v in edges:
            w = labels[u] + labels[v]
            edge_weights.append(w)
            counts[w] = counts.get(w, 0) + 1
        penalty = sum(c - 1 for c in counts.values() if c > 1)

        for _ in range(iterations):
            if time.time() > deadline:
                return None
            if penalty == 0:
                valid, _, _ = validate_labeling(graph, labels, store=False)
                return dict(labels) if valid else None

            bad_edges = [idx for idx, w in enumerate(edge_weights) if counts[w] > 1]
            bad_edge = rng.choice(bad_edges)
            a, b = edges[bad_edge]
            v = a if rng.random() < 0.5 else b
            current = labels[v]

            if cap <= max_candidates:
                values = list(range(1, cap + 1))
                rng.shuffle(values)
            else:
                values = {current, 1, cap}
                while len(values) < max_candidates:
                    values.add(rng.randint(1, cap))
                values = list(values)
                rng.shuffle(values)

            best_values = []
            best_penalty = None
            for x in values:
                touched = set()
                deltas = []
                for edge_idx in incident[v]:
                    u, wv = edges[edge_idx]
                    other = wv if u == v else u
                    old_weight = edge_weights[edge_idx]
                    new_weight = x + labels[other]
                    if old_weight == new_weight:
                        continue
                    touched.add(old_weight)
                    touched.add(new_weight)
                    deltas.append((old_weight, new_weight))

                old_local = sum(max(0, counts.get(w, 0) - 1) for w in touched)
                temp_counts = {w: counts.get(w, 0) for w in touched}
                for old_weight, new_weight in deltas:
                    temp_counts[old_weight] -= 1
                    temp_counts[new_weight] = temp_counts.get(new_weight, 0) + 1
                new_local = sum(max(0, c - 1) for c in temp_counts.values())
                candidate_penalty = penalty - old_local + new_local

                if best_penalty is None or candidate_penalty < best_penalty:
                    best_penalty = candidate_penalty
                    best_values = [x]
                    if best_penalty == 0:
                        break
                elif candidate_penalty == best_penalty:
                    best_values.append(x)

            chosen = rng.choice(best_values)
            labels[v] = chosen
            for edge_idx in incident[v]:
                u, wv = edges[edge_idx]
                other = wv if u == v else u
                old_weight = edge_weights[edge_idx]
                new_weight = chosen + labels[other]
                if old_weight == new_weight:
                    continue
                counts[old_weight] -= 1
                if counts[old_weight] == 0:
                    del counts[old_weight]
                counts[new_weight] = counts.get(new_weight, 0) + 1
                edge_weights[edge_idx] = new_weight
            penalty = best_penalty if best_penalty is not None else penalty

    return None


def is_prime(x: int) -> bool:
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    r = int(math.sqrt(x))
    for d in range(3, r + 1, 2):
        if x % d == 0:
            return False
    return True


def next_prime(x: int) -> int:
    y = max(2, x)
    while not is_prime(y):
        y += 1
    return y


def sidon_backup_labels(graph: CompleteTripartiteGraph) -> Dict[Vertex, int]:
    p = next_prime(graph.V + 1)
    return {v: 2 * p * v + ((v * v) % p) + 1 for v in graph.vertices()}


def solve(graph: CompleteTripartiteGraph, restarts: int = 20, exact_seconds: float = 0.25) -> Result:
    start = time.time()
    lb = lower_bound(graph)
    best_labels = None
    best_k = None
    best_strategy = None

    if graph.V <= 10:
        for cap in range(lb, lb + 15):
            labels = exact_search(graph, cap, exact_seconds)
            if labels is not None:
                best_labels = labels
                best_k = max(labels.values())
                best_strategy = f"bounded_dfs_cap_{cap}"
                break

    if graph.V <= 60 and graph.E <= 1200:
        if graph.E <= 100:
            repair_args = {"attempts": 12, "iterations": 1400, "max_candidates": 220, "time_limit": 14.0}
        elif graph.E <= 300:
            repair_args = {"attempts": 6, "iterations": 900, "max_candidates": 160, "time_limit": 8.0}
        else:
            repair_args = {"attempts": 3, "iterations": 500, "max_candidates": 100, "time_limit": 5.0}

        for cap in cap_schedule(lb):
            labels = local_repair_search(graph, cap, **repair_args)
            if labels is None:
                continue
            k = max(labels.values())
            if best_k is None or k < best_k:
                best_labels = labels
                best_k = k
                best_strategy = f"local_repair_cap_{cap}"
                break

    if graph.V <= 80:
        cap = max(lb * 3 + 20, (best_k or lb) + 20)
        for strategy, order in candidate_orders(graph, restarts):
            labels = greedy_one_order(graph, order, cap)
            if labels is None:
                continue
            k = max(labels.values())
            if best_k is None or k < best_k:
                best_labels = labels
                best_k = k
                best_strategy = strategy

    if best_labels is None:
        best_labels = sidon_backup_labels(graph)
        best_k = max(best_labels.values())
        best_strategy = "sidon_backup_after_heuristic_failure"

    valid, edge_weights, _ = validate_labeling(graph, best_labels)
    runtime = time.time() - start
    gap = best_k - lb
    return Result(
        graph_name=graph.name,
        algorithm="Implicit-partition DFS + local repair + multi-start greedy",
        labels=best_labels,
        edge_weights=edge_weights,
        k=best_k,
        lower_bound=lb,
        gap=gap,
        gap_ratio=round(best_k / lb, 6),
        valid=valid,
        runtime=runtime,
        status=(
            "Sidon backup only; not an optimized result"
            if best_strategy == "sidon_backup_after_heuristic_failure"
            else "Valid heuristic upper bound; optimality is not claimed"
        ),
        best_strategy=best_strategy or "unknown",
    )


def print_result(graph: CompleteTripartiteGraph, result: Result, max_items: int = 20) -> None:
    print(f"\nGraph: {result.graph_name}")
    print(f"Algorithm: {result.algorithm}")
    print(f"Strategy: {result.best_strategy}")
    print(f"Status: {result.status}")
    print(f"|V|={graph.V}, |E|={graph.E}, Delta={graph.delta}, lower_bound={result.lower_bound}")
    print(f"k={result.k}, gap={result.gap}, gap_ratio={result.gap_ratio}, valid={result.valid}")
    print(f"runtime={result.runtime:.6f}s")
    print("Sample labels:", dict(list(sorted(result.labels.items()))[:max_items]))
    print("Sample edge weights:", dict(list(sorted(result.edge_weights.items()))[:max_items]))


def prompt_int(name: str, default: int, minimum: int = 1) -> int:
    while True:
        try:
            raw = input(f"Enter {name} [{default}]: ").strip()
        except EOFError:
            return default
        if not raw:
            return default
        try:
            value = int(raw)
            if value < minimum:
                print(f"{name} must be >= {minimum}.")
                continue
            return value
        except ValueError:
            print("Please enter an integer.")


def pause_before_exit() -> None:
    try:
        input("\nPress Enter to exit...")
    except EOFError:
        pass


def main() -> None:
    if len(sys.argv) == 1:
        print("Complete Tripartite Graph K_{a,b,c} Solver")
        print("Press Enter to use the default value shown in brackets.")
        a = prompt_int("a", 3)
        b = prompt_int("b", 4)
        c = prompt_int("c", 5)
        restarts = prompt_int("random restarts", 20, minimum=0)
        graph = CompleteTripartiteGraph(a, b, c)
        result = solve(graph, restarts=restarts)
        print_result(graph, result)
        pause_before_exit()
        return

    parser = argparse.ArgumentParser(description="Complete tripartite graph K_{a,b,c} solver.")
    parser.add_argument("a", type=int, nargs="?", default=3)
    parser.add_argument("b", type=int, nargs="?", default=4)
    parser.add_argument("c", type=int, nargs="?", default=5)
    parser.add_argument("--restarts", type=int, default=20)
    args = parser.parse_args()

    graph = CompleteTripartiteGraph(args.a, args.b, args.c)
    result = solve(graph, restarts=args.restarts)
    print_result(graph, result)


if __name__ == "__main__":
    main()
