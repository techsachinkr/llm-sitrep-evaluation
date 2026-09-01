"""Offline tests for src/conventions.py.

Every counter in this module is a paper number, so the expectations below are hand-written
strings with hand-counted matches. No network, no LLM, nothing written outside tmp_path.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.conventions import (
    FIGURE_SOURCE_WINDOW_TOKENS,
    Span,
    analyze_corpus,
    analyze_text,
    figures_with_source,
    find_as_of,
    find_figures,
    find_hedges,
    find_revisions,
    find_sourcing,
    load_corpus,
    main,
    token_gap,
    token_spans,
    write_tables,
)
from src.util import write_json

IDAI = "As of 2 April 2019, at least 598 people have died, according to INGC."


def surfaces(text: str, spans) -> list[str]:
    return [text[s.start:s.end] for s in spans]


# ---------------------------------------------------------------------------
# sourcing
# ---------------------------------------------------------------------------
def test_attribution_phrases_are_counted_once_each():
    text = ("According to OCHA the road is open. Reports indicate flooding. The toll was reported by "
            "the district office. Source: national assessment team.")
    attribution, _ = find_sourcing(text)
    assert [s.label for s in attribution] == ["according_to", "reports_indicate", "reported_by", "source_label"]


def test_as_reported_by_is_one_phrase_not_two():
    attribution, _ = find_sourcing("The figure was as reported by IFRC.")
    assert len(attribution) == 1 and surfaces("The figure was as reported by IFRC.", attribution) == \
        ["as reported by"]


def test_agency_acronyms_are_case_sensitive():
    _, agencies = find_sourcing("WHO reported the outbreak.")
    assert [a.label for a in agencies] == ["agency_acronym"]
    _, agencies = find_sourcing("Nobody knows who reported the outbreak, or who to call, or who cares.")
    assert agencies == []


def test_named_agencies_multiword_and_acronyms():
    text = "OCHA, the IFRC, the Red Cross and the ministry of health met the Government of Malawi."
    _, agencies = find_sourcing(text)
    assert surfaces(text, agencies) == ["OCHA", "IFRC", "Red Cross", "ministry of health",
                                        "Government of Malawi"]


def test_sourcing_cues_total_is_phrases_plus_agencies():
    st = analyze_text(IDAI)
    assert (st.sourcing_phrases, st.agency_mentions, st.sourcing_cues) == (1, 1, 2)


# ---------------------------------------------------------------------------
# hedging
# ---------------------------------------------------------------------------
def test_hedge_lexicon_hits():
    text = ("Approximately 1.5 million people may be affected. The estimate is preliminary and the toll "
            "is unconfirmed; casualties could rise and reports are unverified.")
    assert sorted({s.label for s in find_hedges(text)}) == [
        "approximately", "estimated", "modal_could", "modal_may", "preliminary", "unconfirmed", "unverified"]


def test_month_may_is_not_a_hedge_but_modal_may_is():
    assert [s.label for s in find_hedges("In May 2019 the team deployed.")] == []
    assert [s.label for s in find_hedges("the team may deploy")] == ["modal_may"]


def test_no_hedges_in_a_plain_factual_sentence():
    st = analyze_text("Three health centres reopened on 5 April 2019.")
    assert st.hedges == 0 and st.hedge_types == 0


# ---------------------------------------------------------------------------
# update markers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "as of 2 April 2019",
    "As of 2 April",
    "as of April 2, 2019",
    "as of 2019-04-02",
    "as at 02/04/2019",
    "as of 14:00 GMT",
    "as of today",
    "as of this morning",
    "as of the time of writing",
])
def test_as_of_variants_are_update_markers(text: str):
    assert len(find_as_of(f"The camp held 400 people {text}.")) == 1


@pytest.mark.parametrize("text", ["as of course this is not a date", "as usual the road was closed",
                                  "the report was issued on 2 April 2019"])
def test_non_as_of_strings_do_not_match(text: str):
    assert find_as_of(text) == []


def test_revision_language():
    text = ("Figures were revised upward compared to the previous report; previously reported 200 deaths. "
            "This update supersedes Update No. 4 and covers the reporting period.")
    labels = [s.label for s in find_revisions(text)]
    assert labels == ["figures_updated", "compared_to_previous", "previously_reported", "this_update",
                      "supersedes", "update_number", "reporting_period"]


def test_update_markers_is_as_of_plus_revision():
    st = analyze_text("As of 2 April 2019 the figures were revised.")
    assert (st.as_of_dates, st.revision_markers, st.update_markers) == (1, 1, 2)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def test_figures_are_quantities_not_dates_times_or_report_numbers():
    text = "As of 2 April 2019, Sitrep No. 7 (page 3 of 4) reported 1,641 injured and 45% displaced at 14:00."
    assert surfaces(text, find_figures(text)) == ["1,641", "45"]


def test_grouped_thousands_and_decimals_are_single_figures():
    text = "1,234,567 people and 1.5 million households and 12 boats."
    assert surfaces(text, find_figures(text)) == ["1,234,567", "1.5", "12"]


def test_figure_with_source_positive_case():
    st = analyze_text("at least 598 people, according to INGC")
    assert (st.n_figures, st.n_figures_with_source, st.figure_with_source_rate) == (1, 1, 1.0)


def test_bare_number_does_not_count_as_sourced():
    st = analyze_text("A total of 1,200 households were affected.")
    assert (st.n_figures, st.n_figures_with_source, st.figure_with_source_rate) == (1, 0, 0.0)


def test_figure_with_source_rate_is_none_without_figures():
    assert analyze_text("Access to the district remains constrained, according to OCHA.") \
        .figure_with_source_rate is None


def test_source_cue_must_be_within_the_token_window():
    filler = " ".join(["and"] * 14)
    text = f"1,200 households {filler} according to OCHA"
    assert analyze_text(text, window=FIGURE_SOURCE_WINDOW_TOKENS).n_figures_with_source == 0
    assert analyze_text(text, window=20).n_figures_with_source == 1


def test_token_gap_counts_whole_tokens_between_spans():
    text = "at least 598 people, according to INGC"
    toks = token_spans(text)
    figure = find_figures(text)[0]
    cue = find_sourcing(text)[0][0]
    assert token_gap(figure, cue, toks) == 1          # only "people," sits between them
    assert token_gap(cue, figure, toks) == 1          # symmetric
    assert token_gap(figure, figure, toks) == 0
    assert figures_with_source(text, [figure], [cue], window=0) == [False]
    assert figures_with_source(text, [figure], [cue], window=1) == [True]


def test_token_gap_on_empty_text():
    assert token_gap(Span("a", 0, 1), Span("b", 2, 3), []) == 0


# ---------------------------------------------------------------------------
# aggregate statistics
# ---------------------------------------------------------------------------
def test_analyze_text_on_the_worked_example():
    st = analyze_text(IDAI)
    assert st.n_tokens == 14  # As of 2 April 2019, at least 598 people have died, according to INGC.
    assert (st.sourcing_phrases, st.agency_mentions, st.sourcing_cues) == (1, 1, 2)
    assert (st.as_of_dates, st.revision_markers, st.update_markers) == (1, 0, 1)
    assert (st.n_figures, st.n_figures_with_source) == (1, 1)
    assert st.hedges == 0
    assert st.terms["sourcing:according_to"] == 1 and st.terms["agency:agency_acronym"] == 1
    assert st.per_1k(1) == round(1000 / 14, 3)


def test_analyze_text_handles_empty_and_none():
    for empty in ("", None):
        st = analyze_text(empty)  # type: ignore[arg-type]
        assert st.n_tokens == 0 and st.n_figures == 0 and st.per_1k(3) == 0.0


def test_analyze_corpus_totals_and_rate():
    records = [
        {"event": "e1", "id": "a", "source": "OCHA", "date": "2019-03-20", "text": IDAI},
        {"event": "e1", "id": "b", "source": "OCHA", "date": "2019-03-21",
         "text": "A total of 1,200 households were affected."},
    ]
    rows, terms = analyze_corpus(records)
    assert [r["sitrep_id"] for r in rows] == ["a", "b", "TOTAL"]
    total = rows[-1]
    assert total["event"] == "ALL" and total["source"] == "2 sitreps"
    assert total["n_figures"] == 2 and total["n_figures_with_source"] == 1
    assert total["figure_with_source_rate"] == 0.5      # corpus rate, not the mean of per-sitrep rates
    assert total["sourcing_cues"] == 2 and total["update_markers"] == 1
    assert total["n_tokens"] == rows[0]["n_tokens"] + rows[1]["n_tokens"]
    assert terms["sourcing:according_to"] == 1


def test_corpus_rate_is_not_the_mean_of_sitrep_rates():
    records = [{"id": "a", "text": "598 dead, according to INGC"},
               {"id": "b", "text": "1 2 3 4 5 6 7 8 9 unsourced numbers"}]
    rows, _ = analyze_corpus(records)
    assert rows[0]["figure_with_source_rate"] == 1.0
    assert rows[1]["figure_with_source_rate"] == 0.0
    assert rows[-1]["figure_with_source_rate"] == round(1 / 10, 4)


# ---------------------------------------------------------------------------
# corpus I/O + CLI
# ---------------------------------------------------------------------------
def make_corpus(root: Path) -> Path:
    sit = root / "sitreps_human"
    write_json(sit / "cyclone_idai_2019" / "2019-03-20.json",
               {"id": "rw-1", "title": "T", "source": "OCHA", "date": "2019-03-20", "url": "u",
                "text": IDAI, "collection": "manual"})
    write_json(sit / "cyclone_idai_2019" / "2019-03-21.json",
               {"id": "rw-2", "source": "IFRC", "date": "2019-03-21",
                "text": "A total of 1,200 households were affected."})
    return sit


def test_load_corpus_reads_events_and_skips_unusable(tmp_path):
    sit = make_corpus(tmp_path)
    write_json(sit / "cyclone_idai_2019" / "2019-03-22.json", {"id": "empty", "text": "   "})
    write_json(sit / "cyclone_idai_2019" / "_stats.json", {"id": "stats", "text": "ignored"})
    (sit / "cyclone_idai_2019" / "broken.json").write_text("{not json", encoding="utf-8")
    records = load_corpus(sit)
    assert [r["id"] for r in records] == ["rw-1", "rw-2"]
    assert records[0]["event"] == "cyclone_idai_2019"


def test_load_corpus_of_missing_directory_is_empty(tmp_path):
    assert load_corpus(tmp_path / "nope") == []


def test_write_tables_columns_and_rows(tmp_path):
    records = load_corpus(make_corpus(tmp_path))
    rows, terms = analyze_corpus(records)
    stats_path, terms_path = write_tables(rows, terms, tmp_path / "tables")
    with open(stats_path, encoding="utf-8", newline="") as fh:
        out = list(csv.DictReader(fh))
    assert [r["sitrep_id"] for r in out] == ["rw-1", "rw-2", "TOTAL"]
    assert out[0]["event"] == "cyclone_idai_2019" and out[0]["source"] == "OCHA"
    assert out[-1]["n_figures"] == "2"
    with open(terms_path, encoding="utf-8", newline="") as fh:
        term_rows = list(csv.DictReader(fh))
    assert {"category", "label", "count"} == set(term_rows[0])
    assert any(r["label"] == "according_to" and r["count"] == "1" for r in term_rows)


def test_main_writes_tables_and_returns_zero(tmp_path, capsys):
    sit = make_corpus(tmp_path)
    rc = main(["--corpus", str(sit), "--tables-dir", str(tmp_path / "tables")])
    assert rc == 0
    assert (tmp_path / "tables" / "convention_stats.csv").exists()
    assert (tmp_path / "tables" / "convention_terms.csv").exists()
    out = capsys.readouterr().out
    assert "convention stats over 2 sitrep(s)" in out and "corpus totals" in out


def test_main_smoke_limits_and_suffixes_the_tables(tmp_path, capsys):
    sit = make_corpus(tmp_path)
    rc = main(["--corpus", str(sit), "--tables-dir", str(tmp_path / "tables"), "--smoke", "1"])
    assert rc == 0
    assert (tmp_path / "tables" / "convention_stats_smoke.csv").exists()
    assert not (tmp_path / "tables" / "convention_stats.csv").exists()   # the real table is untouched
    assert "[SMOKE]" in capsys.readouterr().out


def test_main_degrades_gracefully_without_a_corpus(tmp_path, capsys):
    rc = main(["--corpus", str(tmp_path / "missing"), "--tables-dir", str(tmp_path / "tables")])
    assert rc == 2
    assert "No human sitreps found" in capsys.readouterr().out
    assert not (tmp_path / "tables").exists()


def test_main_rejects_a_negative_window(tmp_path):
    assert main(["--corpus", str(tmp_path), "--window", "-1"]) == 2
