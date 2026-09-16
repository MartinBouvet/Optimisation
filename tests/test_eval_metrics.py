from optimart.eval.baselines import run_arm
from optimart.eval.cases import crafted_cases
from optimart.eval.metrics import answer_covered, percentile, support_recall
from optimart.prune.pipeline import PruneEngine, message_text


def test_percentile_basic():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10], 95) == 10


def test_answer_coverage_normalizes():
    assert answer_covered(["Paris"], "the city of PARIS is large")
    assert not answer_covered(["Paris"], "France is in Europe")


def test_full_context_keeps_every_answer():
    engine = PruneEngine(budget_ratio=0.45)
    case = crafted_cases()[0]
    result = run_arm(engine, case.messages(), "full")
    text = " ".join(message_text(msg) for msg in result.messages)
    assert answer_covered(case.answers, text)
    assert support_recall(case.supports, text) == 1.0
    assert result.reduction == 0.0
