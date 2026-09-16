from optimart.data.segmentation import split_sentences


def test_split_preserves_answer_offsets():
    text = "Paris is the capital of France. Lyon is smaller. Marseille is a port."
    spans = split_sentences(text)
    assert len(spans) == 3
    answer = "France"
    start = text.index(answer)
    end = start + len(answer)
    hits = [s for s in spans if s.overlaps(start, end)]
    assert len(hits) == 1
    assert "France" in hits[0].text
