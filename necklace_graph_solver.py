# ============================================================
# Cubic Necklace Graph N_{n,3}
# Final Solver
#
# Input:
#   n
#
# Output:
#   V, E, Delta, Lower Bound, Returned k, Gap, Gap Ratio,
#   Validity, Runtime, sample labels, sample edge weights
#
# Method:
#   Structured edge generator + cached adjacency list
#   Multi-start greedy labeling
#
# Note:
#   Returned k is an algorithmic upper bound.
# ============================================================

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
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
    labels: Dict[int, int]
    edge_weights: Dict[Edge, int]
    k: Optional[int]
    lower_bound: int
    gap: Optional[int]
    gap_ratio: Optional[float]
    valid: bool
    runtime: float
    status: str
    best_strategy: str


# ============================================================
# Graph Definition
# ============================================================

class NecklaceGraph:
    """
    Cubic Necklace Graph N_{n,3}

    Vertices:
        v_0, v_1, ..., v_n, v_{n+1}
        u_1, u_2, ..., u_n

    Internal mapping:
        v_i -> i
        u_i -> n + 1 + i

    Edges:
        upper path: v_0-v_1-...-v_n-v_{n+1}
        lower path: u_1-u_2-...-u_n
        rungs: v_i-u_i
        terminal lower edges: v_0-u_1, v_{n+1}-u_n
        long edge: v_0-v_{n+1}
    """

    def __init__(self, n: int):
        if n < 2:
            raise ValueError("For N_{n,3}, n must be at least 2.")

        self.n = n
        self.name = f"N_{{{n},3}}"
        self.V = 2 * n + 2
        self.E = 3 * n + 3
        self.delta = 3

        self.edge_records = list(self._generate_edges())
        self.adj = self._build_adjacency()

        self._validate_topology()

    def v(self, i: int) -> int:
        return i

    def u(self, i: int) -> int:
        return self.n + 1 + i

    def vertices(self) -> List[int]:
        return list(range(self.V))

    def _generate_edges(self):
        n = self.n

        # upper path
        for i in range(0, n + 1):
            yield (self.v(i), self.v(i + 1)), "upper_path"

        # lower path
        for i in range(1, n):
            yield (self.u(i), self.u(i + 1)), "lower_path"

        # rungs
        for i in range(1, n + 1):
            yield (self.v(i), self.u(i)), "rung"

        # terminal lower edges
        yield (self.v(0), self.u(1)), "left_terminal_lower"
        yield (self.v(n + 1), self.u(n)), "right_terminal_lower"

        # long edge
        yield (self.v(0), self.v(n + 1)), "long_edge"

    def _build_adjacency(self):
        adj = {v: [] for v in self.vertices()}
        seen = set()

        for (a, b), edge_type in self.edge_records:
            e = tuple(sorted((a, b)))
            if e in seen:
                continue
            seen.add(e)
            adj[a].append(b)
            adj[b].append(a)

        for v in adj:
            adj[v].sort()

        return adj

    def _validate_topology(self):
        edge_set = set()

        for (a, b), edge_type in self.edge_records:
            if a == b:
                raise ValueError("Self-loop detected.")
            edge_set.add(tuple(sorted((a, b))))

        if len(edge_set) != self.E:
            raise ValueError(f"Edge count error: expected {self.E}, got {len(edge_set)}")

        for v in self.vertices():
            if len(self.adj[v]) != 3:
                raise ValueError(f"Degree error at vertex {v}: expected 3, got {len(self.adj[v])}")

    def neighbors(self, v: int) -> List[int]:
        return self.adj[v]

    def edges(self):
        for (u, v), edge_type in self.edge_records:
            yield tuple(sorted((u, v))), edge_type


# ============================================================
# Utilities
# ============================================================

def lower_bound(graph: NecklaceGraph) -> int:
    return max(math.ceil((graph.E + 1) / 2), graph.delta)


def validate_labeling(graph: NecklaceGraph, labels: Dict[int, int], store: bool = True):
    if len(labels) != graph.V:
        return False, {}, "Missing labels."

    used = set()
    edge_weights = {}

    for edge, edge_type in graph.edges():
        u, v = edge
        w = labels[u] + labels[v]

        if w in used:
            return False, edge_weights, f"Duplicate edge weight found: {w}"

        used.add(w)

        if store:
            edge_weights[edge] = w

    if len(used) != graph.E:
        return False, edge_weights, "Unique edge count does not match E."

    return True, edge_weights, "All edge weights are unique."


def print_result(graph: NecklaceGraph, result: Result, max_labels: int = 20, max_edges: int = 20):
    print("\n" + "=" * 78)
    print("Cubic Necklace Graph Labeling Result")
    print("=" * 78)
    print(f"Graph: {result.graph_name}")
    print(f"Algorithm: {result.algorithm}")
    print(f"Best Strategy: {result.best_strategy}")
    print(f"Status: {result.status}")
    print(f"Valid labeling? {result.valid}")

    print("\n[1] Graph Properties")
    print(f"  |V| = {graph.V}")
    print(f"  |E| = {graph.E}")
    print(f"  Delta = {graph.delta}")
    print(f"  Lower Bound = {result.lower_bound}")

    print("\n[2] K Comparison")
    print(f"  Returned k = {result.k}")
    print(f"  Gap = {result.gap}")
    print(f"  Gap Ratio = {result.gap_ratio}")

    print("\n[3] Runtime")
    print(f"  Runtime = {result.runtime:.6f} seconds")

    print(f"\n[4] First {max_labels} Vertex Labels")
    for i, (v, lab) in enumerate(sorted(result.labels.items())):
        if i >= max_labels:
            break
        print(f"  vertex {v}: label {lab}")

    print(f"\n[5] First {max_edges} Edge Weights")
    for i, (edge, w) in enumerate(sorted(result.edge_weights.items())):
        if i >= max_edges:
            break
        print(f"  edge={edge}, weight={w}")

    print("=" * 78)


def print_assignment_sections(graph: NecklaceGraph, result: Result):
    print("\n" + "=" * 78)
    print("Eight Required Assignment Items - Cubic Necklace Graph")
    print("=" * 78)

    print("\n1. Best data structure")
    print("  Use a structured edge generator, cached edge list, and adjacency-list")
    print("  dictionary. Since N_{n,3} is 3-regular and sparse, O(V+E) storage is better")
    print("  than an O(V^2) adjacency matrix. Labels and edge weights are stored in")
    print("  dictionaries; uniqueness verification uses a hash set.")

    print("\n2. Vertex k-labeling algorithm")
    print("  The solver assigns phi(v) in {1,...,k}; each candidate label is accepted only")
    print("  if all newly formed sums phi(u)+phi(v) are absent from the global used-weight")
    print("  set. The final result is independently verified over all edges.")

    print("\n3. Design strategy and justification")
    print("  Strategy: bounded DFS for small n, min-conflicts local repair, and")
    print("  multi-start greedy traversal.")
    print("  This is suitable because the graph is sparse and nearly path-like, so structured")
    print("  orders such as lower-first or alternating paths exploit its two-track topology,")
    print("  while local repair reduces duplicate weights in medium-size cases.")

    print("\n4. Traversal")
    print("  Traversal orders include natural order, upper-first, lower-first, alternating")
    print("  path order, reverse alternating, terminals-first, terminals-last, rung-pairs,")
    print("  and seeded random restarts.")

    print("\n5. Stored outcome")
    print(f"  labels: Dict[vertex, label], count = {len(result.labels)}")
    print(f"  edge_weights: Dict[(u,v), phi(u)+phi(v)], count = {len(result.edge_weights)}")
    print(f"  sample labels = {dict(list(sorted(result.labels.items()))[:8])}")
    print(f"  sample weights = {dict(list(sorted(result.edge_weights.items()))[:8])}")

    print("\n6. Mathematical comparison")
    print("  | Graph | V | E | Delta | Lower Bound | Returned k | Gap | Gap Ratio | Valid |")
    print("  |---|---:|---:|---:|---:|---:|---:|---:|---|")
    print(
        f"  | {result.graph_name} | {graph.V} | {graph.E} | {graph.delta} | "
        f"{result.lower_bound} | {result.k} | {result.gap} | {result.gap_ratio} | {result.valid} |"
    )
    if result.gap == 0:
        print("  Since returned k equals the lower bound, this tested instance is optimal.")
    else:
        print("  The result is a validated upper bound. More search may reduce the gap.")

    print("\n7. Hardware resources under 8GB RAM")
    print("  Storage is O(V+E). Since V=2n+2 and E=3n+3, a conservative storage-only")
    print("  ceiling in Python is about n <= 1,600,000 (V about 3,200,002 and E about")
    print("  4,800,003). Optimization is CPU-bound; bounded DFS is recommended for")
    print("  small n, while greedy traversal is suitable for much larger sparse cases.")

    print("\n8. Time complexity")
    print("  Generation: O(V+E)=O(n). Validation: O(E). One greedy pass: O(E*k), with")
    print("  constant degree 3. Local repair: O(A*I*k*E). Bounded DFS worst case is")
    print("  O(k^V), so it is used only for small instances and protected by a time limit.")
    print("=" * 78)


# ============================================================
# Multi-Start Greedy
# ============================================================

def build_orders(graph: NecklaceGraph, random_restarts: int = 20):
    n = graph.n
    orders = []

    vertices = graph.vertices()

    upper_all = [graph.v(i) for i in range(0, n + 2)]
    lower_all = [graph.u(i) for i in range(1, n + 1)]

    orders.append(("natural", vertices))
    orders.append(("upper_first", upper_all + lower_all))
    orders.append(("lower_first", lower_all + upper_all))

    alternating = [graph.v(0)]
    for i in range(1, n + 1):
        alternating.append(graph.v(i))
        alternating.append(graph.u(i))
    alternating.append(graph.v(n + 1))
    orders.append(("alternating_paths", alternating))

    reverse = [graph.v(n + 1)]
    for i in range(n, 0, -1):
        reverse.append(graph.v(i))
        reverse.append(graph.u(i))
    reverse.append(graph.v(0))
    orders.append(("reverse_alternating_paths", reverse))

    terminals_first = [graph.v(0), graph.v(n + 1)]
    terminals_first += [graph.v(i) for i in range(1, n + 1)]
    terminals_first += lower_all
    orders.append(("terminals_first", terminals_first))

    terminals_last = [graph.v(i) for i in range(1, n + 1)]
    terminals_last += lower_all
    terminals_last += [graph.v(0), graph.v(n + 1)]
    orders.append(("terminals_last", terminals_last))

    rung_pairs = [graph.v(0)]
    for i in range(1, n + 1):
        rung_pairs.append(graph.v(i))
        rung_pairs.append(graph.u(i))
    rung_pairs.append(graph.v(n + 1))
    orders.append(("rung_pairs", rung_pairs))

    for seed in range(random_restarts):
        rng = random.Random(8000 + seed)
        order = vertices[:]
        rng.shuffle(order)
        orders.append((f"random_seed_{8000 + seed}", order))

    return orders


def greedy_one_order(graph: NecklaceGraph, order: List[int], label_cap: int):
    labels = {}
    used_weights = set()

    for v in order:
        labeled_neighbors = [u for u in graph.neighbors(v) if u in labels]
        assigned = False

        for x in range(1, label_cap + 1):
            local = set()
            new_weights = []
            conflict = False

            for u in labeled_neighbors:
                w = x + labels[u]

                if w in used_weights or w in local:
                    conflict = True
                    break

                local.add(w)
                new_weights.append(w)

            if not conflict:
                labels[v] = x

                for w in new_weights:
                    used_weights.add(w)

                assigned = True
                break

        if not assigned:
            return None

    valid, _, _ = validate_labeling(graph, labels, store=False)

    return labels if valid else None


def exact_search(graph: NecklaceGraph, label_cap: int, seconds: float):
    """Small-instance DFS feasibility search for labels <= label_cap."""
    deadline = time.time() + seconds
    order = sorted(graph.vertices(), key=lambda v: len(graph.neighbors(v)), reverse=True)
    labels = {}
    used_weights = set()

    def dfs(pos: int):
        if time.time() > deadline:
            return None

        if pos == len(order):
            valid, _, _ = validate_labeling(graph, labels, store=False)
            return dict(labels) if valid else None

        v = order[pos]
        labeled_neighbors = [u for u in graph.neighbors(v) if u in labels]

        for x in range(1, label_cap + 1):
            local = set()
            new_weights = []
            conflict = False

            for u in labeled_neighbors:
                w = x + labels[u]
                if w in used_weights or w in local:
                    conflict = True
                    break
                local.add(w)
                new_weights.append(w)

            if conflict:
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


def conflict_count(graph: NecklaceGraph, labels: Dict[int, int]) -> int:
    counts = {}
    for edge, edge_type in graph.edges():
        u, v = edge
        w = labels[u] + labels[v]
        counts[w] = counts.get(w, 0) + 1
    return sum(c - 1 for c in counts.values() if c > 1)


def local_repair_search(
    graph: NecklaceGraph,
    label_cap: int,
    attempts: int = 12,
    iterations: int = 700,
):
    vertices = graph.vertices()
    edges = [edge for edge, edge_type in graph.edges()]

    for attempt in range(attempts):
        rng = random.Random(88000 + attempt + label_cap * 997)
        labels = {v: rng.randint(1, label_cap) for v in vertices}

        for _ in range(iterations):
            weights = {}
            bad_vertices = set()

            for u, v in edges:
                w = labels[u] + labels[v]
                weights.setdefault(w, []).append((u, v))

            duplicated = [items for items in weights.values() if len(items) > 1]
            if not duplicated:
                return labels

            for items in duplicated:
                for u, v in items:
                    bad_vertices.add(u)
                    bad_vertices.add(v)

            v = rng.choice(tuple(bad_vertices))
            current = labels[v]
            best_values = []
            best_score = None

            values = list(range(1, label_cap + 1))
            rng.shuffle(values)

            for x in values:
                labels[v] = x
                score = conflict_count(graph, labels)
                if best_score is None or score < best_score:
                    best_score = score
                    best_values = [x]
                elif score == best_score:
                    best_values.append(x)
                if score == 0:
                    return dict(labels)

            labels[v] = rng.choice(best_values) if best_values else current

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


def sidon_fallback_labels(graph: NecklaceGraph) -> Dict[int, int]:
    p = next_prime(graph.V + 1)
    return {v: 2 * p * v + ((v * v) % p) + 1 for v in graph.vertices()}


def solve(graph: NecklaceGraph, random_restarts: int = 40, exact_seconds: float = 1.5) -> Result:
    start = time.time()
    lb = lower_bound(graph)

    best_labels = None
    best_k = None
    best_strategy = None

    if graph.V <= 14:
        for cap in range(lb, lb + 12):
            labels = exact_search(graph, cap, exact_seconds)
            if labels is not None:
                best_labels = labels
                best_k = max(labels.values())
                best_strategy = f"bounded_dfs_cap_{cap}"
                break

    if graph.V <= 160:
        caps = []
        for multiplier in [1.0, 1.03, 1.06, 1.1, 1.25, 1.5, 2.0]:
            cap = max(lb, math.ceil(lb * multiplier))
            if cap not in caps:
                caps.append(cap)
        for cap in caps:
            labels = local_repair_search(graph, cap)
            if labels is None:
                continue
            k = max(labels.values())
            if best_k is None or k < best_k:
                best_labels = labels
                best_k = k
                best_strategy = f"local_repair_cap_{cap}"
                break

    label_cap = max(lb * 4 + 50, (best_k or lb) + 20)

    trial_count = 0

    for strategy, order in build_orders(graph, random_restarts=random_restarts):
        trial_count += 1

        labels = greedy_one_order(graph, order, label_cap)

        if labels is None:
            continue

        k = max(labels.values())

        if best_k is None or k < best_k:
            best_k = k
            best_labels = labels
            best_strategy = strategy

    if best_labels is None:
        best_labels = sidon_fallback_labels(graph)
        best_k = max(best_labels.values())
        best_strategy = "sidon_backup_after_heuristic_failure"

    valid, edge_weights, reason = validate_labeling(graph, best_labels)

    gap = best_k - lb
    ratio = round(best_k / lb, 6)
    runtime = time.time() - start

    return Result(
        graph_name=graph.name,
        algorithm="Bounded DFS + local repair + multi-start greedy labeling",
        labels=best_labels,
        edge_weights=edge_weights,
        k=best_k,
        lower_bound=lb,
        gap=gap,
        gap_ratio=ratio,
        valid=valid,
        runtime=runtime,
        status="Reached lower bound; optimal for this tested instance" if gap == 0 else (
            "Sidon used only as backup after heuristic failure; not optimized"
            if best_strategy == "sidon_backup_after_heuristic_failure"
            else "Valid upper bound; optimality is not claimed"
        ),
        best_strategy=best_strategy,
    )


# ============================================================
# Main
# ============================================================

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


def main():
    if len(sys.argv) == 1:
        print("Cubic Necklace Graph N_{n,3} Solver")
        print("Press Enter to use the default value shown in brackets.")
        n = prompt_int("n", 8, minimum=2)
        restarts = prompt_int("random restarts", 40, minimum=0)
        graph = NecklaceGraph(n)
        result = solve(graph, random_restarts=restarts)
        print_result(graph, result)
        print_assignment_sections(graph, result)
        pause_before_exit()
        return

    parser = argparse.ArgumentParser(description="Cubic necklace graph N_{n,3} edge irregular labeling solver.")
    parser.add_argument("n", type=int, nargs="?", default=8)
    parser.add_argument("--restarts", type=int, default=40)
    args = parser.parse_args()

    graph = NecklaceGraph(args.n)
    result = solve(graph, random_restarts=args.restarts)

    print_result(graph, result)
    print_assignment_sections(graph, result)


if __name__ == "__main__":
    main()
