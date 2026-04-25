from complete_tripartite_solver import CompleteTripartiteGraph, solve as solve_tripartite
from musical_graph_solver import MusicalGraph, solve as solve_musical
from necklace_graph_solver import NecklaceGraph, solve as solve_necklace


def run_case(case_id, graph_type, params, builder, solver, solver_kwargs):
    graph = builder(*params)
    result = solver(graph, **solver_kwargs)
    return {
        "case": case_id,
        "type": graph_type,
        "graph": result.graph_name,
        "V": graph.V,
        "E": graph.E,
        "Delta": graph.delta,
        "lower_bound": result.lower_bound,
        "k": result.k,
        "gap": result.gap,
        "gap_ratio": result.gap_ratio,
        "valid": result.valid,
        "runtime": result.runtime,
        "strategy": result.best_strategy,
        "status": result.status,
    }


def main():
    cases = [
        # Complete tripartite: dense enough to be meaningful, but avoids Sidon backup.
        ("T1", "Complete Tripartite", (2, 2, 3), CompleteTripartiteGraph, solve_tripartite, {"restarts": 2}),
        ("T2", "Complete Tripartite", (2, 3, 4), CompleteTripartiteGraph, solve_tripartite, {"restarts": 2}),
        ("T3", "Complete Tripartite", (3, 3, 4), CompleteTripartiteGraph, solve_tripartite, {"restarts": 2}),
        ("T4", "Complete Tripartite", (3, 4, 5), CompleteTripartiteGraph, solve_tripartite, {"restarts": 2}),
        ("T5", "Complete Tripartite", (4, 4, 5), CompleteTripartiteGraph, solve_tripartite, {"restarts": 2}),

        # Musical graph: grows to 40 vertices / 100 edges, using local repair not Sidon.
        ("M1", "Musical Graph", (4,), MusicalGraph, solve_musical, {"restarts": 2}),
        ("M2", "Musical Graph", (6,), MusicalGraph, solve_musical, {"restarts": 2}),
        ("M3", "Musical Graph", (8,), MusicalGraph, solve_musical, {"restarts": 2}),
        ("M4", "Musical Graph", (10,), MusicalGraph, solve_musical, {"restarts": 2}),
        ("M5", "Musical Graph", (20,), MusicalGraph, solve_musical, {"restarts": 2}),

        # Necklace graph: sparse structure scales farther while staying optimized.
        ("N1", "Cubic Necklace", (4,), NecklaceGraph, solve_necklace, {"random_restarts": 2}),
        ("N2", "Cubic Necklace", (6,), NecklaceGraph, solve_necklace, {"random_restarts": 2}),
        ("N3", "Cubic Necklace", (12,), NecklaceGraph, solve_necklace, {"random_restarts": 2}),
        ("N4", "Cubic Necklace", (20,), NecklaceGraph, solve_necklace, {"random_restarts": 2}),
        ("N5", "Cubic Necklace", (30,), NecklaceGraph, solve_necklace, {"random_restarts": 2}),
    ]

    rows = [run_case(*case) for case in cases]

    print("| Case | Type | Graph | V | E | Delta | Lower Bound | k | Gap | Gap Ratio | Valid | Runtime(s) | Strategy |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|")
    for row in rows:
        print(
            f"| {row['case']} | {row['type']} | {row['graph']} | {row['V']} | {row['E']} | "
            f"{row['Delta']} | {row['lower_bound']} | {row['k']} | {row['gap']} | "
            f"{row['gap_ratio']} | {row['valid']} | {row['runtime']:.3f} | {row['strategy']} |"
        )

    print("\nNotes:")
    print("- Each graph family has 5 test cases.")
    print("- These tests intentionally avoid Sidon as a reported optimization result.")
    print("- If a row ever reports a strategy containing 'sidon_backup', treat it as a failed optimization case and exclude it from the main comparison table.")
    print("- Positive gaps are verified upper bounds, not exact edge irregularity strengths.")


if __name__ == "__main__":
    main()
