from optimart.prune.knapsack import KnapsackItem, solve_knapsack


def test_knapsack_picks_optimal_pair():
    items = [
        KnapsackItem(0, 2, 3),
        KnapsackItem(1, 3, 4),
        KnapsackItem(2, 4, 5),
        KnapsackItem(3, 5, 6),
    ]
    solution = solve_knapsack(items, budget=5)
    assert solution.chosen == frozenset({0, 1})
    assert solution.tokens_used == 5
    assert solution.method == "dp"


def test_protected_items_never_dropped():
    items = [
        KnapsackItem(0, 10, 0.1, protected=True),
        KnapsackItem(1, 4, 9.0, protected=False),
    ]
    solution = solve_knapsack(items, budget=8)
    assert 0 in solution.chosen
    assert 1 not in solution.chosen
    assert solution.protected_overflow is True
