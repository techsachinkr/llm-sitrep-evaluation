from src.matcher_audit import _exact_binomial_ci, _occurrences, _similarity, resolved_labels


def test_occurrences_attach_numeric_context_without_changing_keys():
    rows = _occurrences("At least 1,200 people were affected; 57 were injured.", "x")
    assert [row.figure.key for row in rows] == [("count", 1200.0), ("count", 57.0)]
    assert all(row.record_id == "x" and row.figure.raw in row.context for row in rows)


def test_context_similarity_prefers_shared_fact_terms():
    official = "The storm affected 1,200 households in Sofala."
    close = "Sofala reports 1,200 households affected by the storm."
    unrelated = "The school reopened after repairs."
    assert _similarity(official, close) > _similarity(official, unrelated)


def test_manual_label_defaults_and_exceptions_are_resolved():
    spec = {"strata": {"x": {"default_label": "ok", "exceptions": {"B": "error"}}}}
    assert resolved_labels(spec, [{"sample_id": "A", "stratum": "x"},
                                  {"sample_id": "B", "stratum": "x"}]) == {
                                      "A": "ok", "B": "error"
                                  }


def test_exact_binomial_interval_contains_observed_rate():
    lo, hi = _exact_binomial_ci(5, 50)
    assert lo < 0.1 < hi
