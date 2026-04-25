from complete_tripartite_solver import CompleteTripartiteGraph, solve as solve_tripartite, print_result as print_tripartite
from musical_graph_solver import MusicalGraph, solve as solve_musical, print_result as print_musical
from necklace_graph_solver import NecklaceGraph, solve as solve_necklace, print_result as print_necklace


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
    print("=" * 72)
    print("Graph Edge Irregular Vertex K-Labeling Solver")
    print("=" * 72)
    print("1. Complete Tripartite Graph K_{a,b,c}")
    print("2. 5-Regular Musical Graph C_{n,2}")
    print("3. Cubic Necklace Graph N_{n,3}")
    print("4. Run expanded test table")
    print("=" * 72)

    choice = prompt_int("choice", 1, minimum=1)

    if choice == 1:
        a = prompt_int("a", 3)
        b = prompt_int("b", 4)
        c = prompt_int("c", 5)
        restarts = prompt_int("random restarts", 20, minimum=0)
        graph = CompleteTripartiteGraph(a, b, c)
        result = solve_tripartite(graph, restarts=restarts)
        print_tripartite(graph, result)

    elif choice == 2:
        n = prompt_int("n", 8, minimum=3)
        restarts = prompt_int("random restarts", 20, minimum=0)
        graph = MusicalGraph(n)
        result = solve_musical(graph, restarts=restarts)
        print_musical(graph, result)

    elif choice == 3:
        n = prompt_int("n", 8, minimum=2)
        restarts = prompt_int("random restarts", 40, minimum=0)
        graph = NecklaceGraph(n)
        result = solve_necklace(graph, random_restarts=restarts)
        print_necklace(graph, result)

    elif choice == 4:
        import run_expanded_tests

        run_expanded_tests.main()

    else:
        print("Invalid choice. Please run again and choose 1, 2, 3, or 4.")

    pause_before_exit()


if __name__ == "__main__":
    main()
