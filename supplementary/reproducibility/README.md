# Anonymous reproducibility supplement

This directory is the submission artifact for the paper. It contains no report body, social-media
text, evidence span, API credential, or author identity.

## Re-run order

1. Use `reliefweb_report_manifest.csv` to retrieve the 125 public reports. When a direct landing
   page was not preserved during collection, `retrieval_url` is an exact-title ReliefWeb query;
   title, source, verified date, filename, and SHA-256 disambiguate the document. Preserve each
   `local_filename`, then run `python -m src.ingest_manual_reliefweb`.
2. Obtain HumAID under its licence and run `python -m src.collect_social --events
   cyclone_idai_2019 hurricane_irma_2017 hurricane_maria_2017 puebla_mexico_earthquake_2017
   kerala_floods_2018`. Require every `(tweet_id, text_sha256)` in `humaid_source_manifest.csv` to
   match before making an API call.
3. Copy `frozen_schema.yaml` to `data/processed/schema.yaml`. Re-issue calls using `prompts.json`,
   `model_roles.csv`, and `decoding_parameters.csv`; the repository CLIs are `src.open_code`,
   `src.consolidate`, `src.generate_sitreps`, `src.judge_slots`, and `src.reference_metrics`.
   Exact original generation is checksum-auditable but cannot be promised from mutable aliases.
4. Run `python -m src.paper_analysis`, `python -m src.matcher_audit`, then
   `python -m src.export_reproducibility`.
5. Run `python -m src.figures --tables-dir results/tables/paper/idai/gemini --completeness
   results/tables/idai/gemini/completeness_by_sitrep.csv --reference-metrics
   results/tables/reference_metrics.csv --correlations results/tables/correlations_gemini.csv`.
6. From `paper/`, run `pdflatex main`, `bibtex main`, and two further `pdflatex main` passes.

For a deterministic audit that does not re-call mutable models, copy `tables/*.csv` back to
`results/tables/paper/` and use `per_judgement.csv` with the last-row/observed-case rules implemented
in `src.paper_analysis`. The release tests in `tests/test_export_reproducibility.py` enforce all
manifest counts and model/prompt/schema contracts.

## Files

- `prompts.json`: every system and user instruction, rendered verbatim with sentinel payloads.
- `frozen_schema.yaml`: all 15 definitions and two synthetic examples per slot, author-anonymized.
- `model_roles.csv`: public model identifiers, provider surfaces, call dates, and snapshot limits.
- `decoding_parameters.csv`: temperature, top-p, effort, and every reported output-token cap.
- `reliefweb_report_manifest.csv`: 125 titles, dates, retrieval URLs, hashes, input lengths, and
  uniform inclusion/exclusion reasons. The four exclusions yield the 121-report analysis corpus.
- `machine_report_manifest.csv`: provenance and hashes for all 264 generated outputs; `evaluated`
  identifies the 261 non-empty reports used in analysis. Output text is not public because spot
  checks found that generations can repeat personal contact details from crisis posts.
- `humaid_source_manifest.csv`: non-text tweet identifiers, dates, labels, order, and text hashes.
  Obtain HumAID under its licence and require every hash to match before regeneration.
- `measurement_spec.json`: exact regexes, normalization, pairing, ROUGE, BERTScore, and clustering.
- `matcher_audit.csv` and `matcher_audit_summary.csv`: text-free seed-17 manual labels, context
  hashes, and exact-binomial error bounds for 50 absent and 50 matched primary-event figures.
- `matcher_audit_label_spec.json`: the complete anonymous label decisions used to regenerate the
  public audit tables after inspecting the private worklist.
- `environment.txt`: Python/platform metadata and the complete installed distribution set.
- `per_judgement.csv`: text-free raw document-slot rows, including status and duplicates; the
  publication analysis applies its documented last-row and observed-case rules.
- `tables/human_validation_pairs.csv` and `tables/human_validation_summary.csv`: the 100 blind
  author annotations joined to all four withheld judge labels, plus recomputed all/human/machine
  accuracy, nominal Cohen's kappa, linear-weighted sensitivity, per-label precision/recall/F1, and
  per-judge missingness.
- `tables/`: generated publication tables plus per-report reference metrics and per-day
  figure/source-availability records; these recompute every Table 2 percentage and Table 3 star.

API aliases are not deterministic snapshots. The artifact records the public ID, call date, exact
request parameters, served-model string, and cached response provenance available from each
provider; DeepSeek and OpenRouter did not expose immutable snapshot IDs for these calls.
