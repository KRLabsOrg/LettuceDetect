# Dataset Provenance

How the published `lettucedetect-code-agent` data was constructed, audited, and
repaired — so the decisions behind the final files are reproducible and not
just their end state. Pipeline reference: [Phases](phases.md); design rationale:
[Index](index.md).

## Construction

1. **Preparation (cached, run once).** All SWE-bench splits (21,527 instances)
   were loaded with their official repository-disjoint split assignment; the
   patch-touched source files were fetched at each base commit and the gold fix
   derived (20,906 instances survived prep — the rest had unfetchable repos or
   unparseable patches); raw issue texts were rewritten into natural developer
   requests.

2. **Generation** (`generate_code_agent_hallucinations.py`, Gemma-4-31B via
   local vLLM, `--answer-source gold`). Each instance became exactly one
   sample — clean (the gold fix verbatim, rendered as a function / fragment /
   edit) or hallucinated (the gold fix with injected request-grounded
   mistakes: `wrong_implementation`, `unrequested_change`, `fabricated_api`,
   `--struct-ratio 0.15`). Quality gates (leaky hint words, non-unique edit
   substrings, coverage caps, span length, no-op edits, fabrications present
   in context, answers over `MAX_ANSWER_CHARS`) rejected roughly half of all
   injection attempts; rejected instances were dropped rather than kept with
   noisy labels. Answer references missing from the truncated context were
   grounded at generation time (four repository tiers + Context7 third-party
   signatures — see [Phases § Grounding](phases.md#grounding)).

## Audit

An independent three-way review (pipeline code, generated data, taxonomy)
validated all spans (zero out-of-bounds or overlapping; no hint-word leakage
in any hallucinated answer; repository-disjoint splits confirmed at the
`org__repo` level) and surfaced the issues below, each verified by manually
reading samples before acting.

## Repairs (applied in place — samples were transformed or quarantined, never
regenerated blindly)

1. **Taxonomy hygiene.** Five prose-source rows (readme, wikipedia) carried
   invalid span categories (`""`, `"NUMERAL"`) that `map_label`'s silent
   fallback had let through; each was re-read against its context and fixed by
   hand. `map_label` now raises on unknown labels, menu injection rejects the
   sample (`validation:unknown_native_label`), and `build_hf_dataset.py`
   validates every span category at merge time.

2. **Category relabel.** 27 spans whose own explanations described a
   nonexistent API ("non-existent `--freeze-all` flag") but carried
   `contradiction`/`unsupported_addition` were relabeled to
   `fabricated_reference` after individual reading (18 further regex hits were
   false positives and kept as-is).

3. **Edit-style quarantine.** In edit-style answers, spans inside the
   `replace` (old-code) block were initially suspected to be inverted labels.
   Manual reading showed 303/309 are *correct* — the agent misquoting the
   original file is itself a verifiable fabrication, checkable against the
   context. The genuinely degenerate cases — 146 no-op edits (old == new) and
   5 spans whose text appears verbatim in the context — were quarantined to
   `data/v2/code_agent/quarantine/removed_edit_style.jsonl` (151 samples).

4. **Grounding repair** (`reground_code_agent_contexts.py`). A cache bug had
   silently turned transient GitHub failures (rate-limit windows) into
   permanent grounding misses. After fixing the fetch layer
   (`TransientFetchError`, retry-once, never cached), the repair script
   re-resolved every sample's ungrounded answer references at the base commit
   and prepended the missing `Referenced definitions` blocks to context and
   prompt — labels and answers untouched, labeled spans blanked first so
   injected text is never grounded. ~2,400 samples gained definitions.

5. **Import-evidence policy.** Names the answer itself imports (stdlib,
   third-party) are counted as evidenced by the import statement — they cannot
   be grounded from the repository and a reviewer treats the import as proof
   of existence. Implemented in `remaining_ungrounded`; this is a measurement
   and labeling policy, not a data change.

   The ungrounded-reference rate (share of samples with ≥1 answer reference
   absent from context) moved 32–37% (first release) → 18.5% (deeper
   grounding) → ~12–13% (repair + policy); the residual is dominated by
   local-variable false positives of the audit metric, not fabrication-shaped
   gaps.

## Class-balance raise (≥50% hallucinated)

The initial run targeted 40% of instances for injection; with the ~50% QC pass
rate that yielded ~28% hallucinated samples. To raise the rate without
violating the one-sample-per-instance (no clean/hallucinated twins) design:

- failed injection targets were retried (the runner is resumable and re-rolls
  failures at temperature 0.8),
- quarantined instances were re-targeted as hallucinated,
- a seeded per-split selection of existing clean samples was *converted*:
  the clean line removed (backed up under
  `data/v2/code_agent/quarantine/pre_50pct_cleans/`) and the instance
  regenerated as a hallucinated sample from the same gold answer.

The explicit target set lives in `data/code_hallucination/hall_ids_50pct.txt`
and is passed via the generator's `--hall-ids-file` flag (added because the
default `--ratio` sampling is not stable across ratio changes). After the run,
the dataset is re-audited (`check_context_quality.py`), re-grounded
(`reground_code_agent_contexts.py`), and re-pushed.

## Verification tooling

- `scripts/check_context_quality.py` — grounding coverage, category/format
  distributions, span coverage, answer lengths.
- `demo/code_hallucination_viewer.py` — Streamlit browser with spans
  highlighted by category, for manual review.
- Every repair above was gated on reading the affected samples, not on the
  flagging heuristic alone; heuristics over-flag (the edit-style case is the
  cautionary example — the automated count suggested ~250 bad samples, reading
  showed 6).

## Test-set verification (≥50% release)

After the class-balance raise, the entire test split was individually reviewed
(2,038 samples → 2,015 retained; 1,014 hallucinated / 1,001 clean, 50.3%).
Three tiers:

1. **Full first-pass review** — annotator agents read every sample against the
   rubric in `data/v2/code_agent/annotations/packets/RUBRIC.md`: per-span
   validity, category, boundary tightness, explanation quality (hallucinated);
   fix-plausibility, artifacts, question↔answer match (clean). 92.9% of
   hallucinated samples were accepted as labeled.
2. **Blind second-pass adjudication** — every flagged case re-judged from the
   raw sample with no access to first-pass verdicts, to prevent anchoring (an
   earlier non-blind attempt rubber-stamped 144/144 first-pass verdicts and was
   discarded).
3. **Evidence arbitration** — pass disagreements resolved against the true
   pre-fix sources: span text absent from the original repository means the
   answer introduced it (genuine hallucination; 41/44 cases), present means the
   span marks original code (span dropped; 3/44).

Applied: 235 spans tightened to the minimal hallucinated substring, 23 invalid
spans dropped, 2 categories corrected, 5 samples reclassified clean, 23 removed.
No post-review rebalancing. Full artifacts (verdicts, blind adjudications,
contested cases with resolutions, tightening decisions) live in
`data/v2/code_agent/annotations/`; summary in `annotations/REPORT.md`.
Train/validation remain machine-generated with automated gates only.
