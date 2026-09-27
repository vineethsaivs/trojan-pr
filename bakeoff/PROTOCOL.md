# Bake-off protocol v1 (pre-registered)

Committed before the first scored model call. The commit timestamp of `bakeoff: pre-register protocol
v1` is the proof. Any later change is protocol v2 and reported as such.

## Questions

- Q1: on the 12 introducing PRs, how often does the AI reviewer name the right function and mechanism?
- Q2: how often does it call one of the 12 fix PRs buggy?
- Q3: does Sentinel, planning blind, block the introducing commit, pass the commit before (where it
  exists) and the fix, and stay quiet on the fix PRs?

## Items

`bakeoff/cases.yaml`: 12 H cases (6 dev, 6 held-out) and their 12 fix PRs (B, the benign control),
with full SHAs, the scope path at each commit and sha256 of every input. Inputs per item: the scope
file's hunks of the PR (g2), PR title and body, and the full scope file at the PR head.

## Reviewer runs

| Setting | Value |
|---|---|
| Model | deepseek-v4-flash-0731 (Vultr Serverless Inference; ids in `bakeoff/models.json`) |
| Prompts | `bakeoff/prompts/p0.txt`, `p1.txt` (P1 = P0 + the 10-pattern list), frozen verbatim |
| Runs | 3 per cell; report mean with min and max, never "caught in any run" alone |
| Calls | 24 items x 1 model x 2 prompts x 3 runs = 144 |
| max_tokens | 8000; temperature: provider default (not set) |
| Parsing | first JSON object in `content`; one repair turn; still unparseable = error, kept in the denominator |
| Retries | 429, 5xx, timeouts: 4 attempts, exponential backoff |
| Concurrency | 8 |
| Runner | `bakeoff/run.py`, raw output per job in `bakeoff/raw/<case>-<which>/<model>/<prompt>/r<n>.json` |

## Deviations from the plan, with evidence

1. **glm-5.3 is excluded as a reviewer.** Pilot on the index-only case unsloth-10842 (not one of the
   24 scored items; only the presence of an answer was inspected): no content at max_tokens 6000
   (60 s) or 32000 (218 s, 124,728 characters of reasoning); with thinking disabled it writes 8000
   tokens of analysis and no JSON. Scoring empty answers as misses would measure the harness, not
   the reviewer. minimax-m3 (16000 tokens, no content) and qwen3.8-27b, glm-5.2, qwen3.8-flash-next
   (HTTP 5xx at pilot time) are excluded for the same reason. Details: `bakeoff/models.json`.
2. **Sentinel's planner is deepseek-v4-flash-0731**, not glm-5.3, for the same reason (the plan's own
   third fallback). So fairness rule 1 holds exactly: reviewer and planner are the same model.
3. The minimum viable bake-off (one model) is the headline; the second reviewer model is cut.

## Fairness rules

1. Reviewer and planner get the same scoped diff, PR title and body, and the same 10-pattern bug
   list, from the same model (deepseek-v4-flash-0731). The reviewer also gets the full scope file
   (more context than the planner). Only Sentinel executes code.
2. 3 runs per cell, mean with min and max.
3. Caught = right location + right mechanism + actionable. Right line with wrong consequence = PARTIAL.
4. Benign = the 12 fix PRs. A flag on a fix is a false alarm unless it names a real remaining bug.
5. Blind adjudication (model and prompt hidden).
6. Sentinel is scored by the same standard: CAUGHT = BLOCK at intro with a detected check on the
   scope function and that check passing at the fix; a held-out miss is reported with its reason;
   a BLOCK on a fix PR is a false alarm.

## Scoring (mechanical then human)

- L: a finding names the scope file (basename ok) and the function or a line within 3 of the buggy hunk.
- A: REQUEST_CHANGES, or severity high or medium.
- M (human): states the mechanism class in `cases.yaml`. T (human): names a trigger from the set.
- CAUGHT = L and M and A. PARTIAL = L and (M or T), not CAUGHT. MISSED = the rest.
- FALSE ALARM (B) = an actionable finding claiming a numerical defect that the adjudicator rejects.
- Adjudication: `bakeoff/adjudicate.py`, only L-passing outputs, shuffled, model and prompt hidden,
  hard cap 30 minutes; leftovers judged by the same model with the same rubric, agreement reported.

## Sentinel on the same PRs (triple control)

Per H case, on the frozen harness `harness-frozen-v1` (sandboxd sha256 `71643ef9c19c3336030e9738b77e60b3d06ea5d567585b766a96d3e7c6dcccd2`):
run 1 plans blind from the introducing PR's inputs and runs that plan at pre, intro, fixp and fix;
run 2 plans blind from the fix PR's inputs and runs at fixp (base) and fix (head). The planner never
sees the fix diff (run 1), `corpus/*.yaml`, `CASES.md` or `cases_facts.json`. Mandatory checks
come from the static prepass (only `clip_grad_norm_`-shaped functions have adapter defaults).
Command: `python -m plumbline.cli queue harness-frozen-v1` on VM1. Expected per commit: pre PASS
(or n/a, born with the code), intro FAIL -> BLOCK, fixp FAIL, fix PASS. unsloth-11470 needs
transformers in the sandbox (the hf runner is cut): whatever it returns is reported, and a
dependency failure counts as a miss.

Known in-sample facts, disclosed: the oracle templates were shaped on the 6 dev cases; the planner
and its validator were developed on ds-8313 (dev). The 10-pattern list in P1 and the planner prompt
came from my own upstream fixes, so it is in-sample for every case.

## Caveats printed under every table

"Dev cases shaped the oracle templates (in-sample). Held-out cases were frozen before their first
run (commit {prereg_sha}). The 10-pattern bug list in the reviewer and planner prompts came from my
own fixes, so it is in-sample for every case."

"The fix PRs merged July to September 2026 and the Vultr models were deployed August 25 to
September 15 2026 (`created` in `/v1/models`); training cutoffs are unknown, so a model may have
seen the fixes, which can only inflate reviewer scores."

## sha256 at pre-registration

| file | sha256 |
|---|---|
| `bakeoff/models.json` | `1af145a0b68d978963fa01045f7c1fde7a5806fb6dafbb8487bb2f1299901d41` |
| `bakeoff/prompts/g3.txt` | `01898687da5b7761c76950d9130d227fd60feded0fab972bd67dc157bc916a74` |
| `bakeoff/prompts/p0.txt` | `731c88d06771d93492a7491cc83f7c131e1e589aac035c33ba99b5f404424ee4` |
| `bakeoff/prompts/p1.txt` | `1f9a7703945a29d95dfacf9e73047ad948cadeda454c9fd109dcdba7dd3fe0a8` |
| `bakeoff/run.py` | `170d31a1533824d0f09d53934bb78838f8676fd2d4cb4c7b502afb65fa32043e` |
| `plumbline/plan_schema.json` | `b55ce354828fea45320e83e13a0dfa6fd8d8843ea6c04e638575ccccf41c1778` |
| `plumbline/prompts/planner.txt` | `7909b4f8a45f6256e4a63ec557ca5f015b431c7ed171a4d917d230e8813563ec` |
| `harness/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `harness/adapters.py` | `bf72e8ba9a610de9f32247b1aa083d53fc5c6086a865f05ec8a2c71cdec2f4b6` |
| `harness/extract.py` | `cf7043d72d66f850172dbea3659f865c72e0a51117d701bf87d3d3b5be5419c3` |
| `harness/gold.py` | `0ee5e043204139af8ffcd58f6a53b90d58a56ece92ed6c63e4620f31499c911f` |
| `harness/oracles.py` | `cd8fb2452c379153c9ad7a32e06e8db075378ae549e94a349fe8ca30e3fda368` |
| `harness/references.py` | `74e01e01f7a9eeb33fb5365a4bf0e07712764dcece8868550947fc48f645b7b5` |
| `harness/runjob.py` | `afe3189de78ef887752b5372b226f4ebf9f9cc0a2e2ed41a2b76095d6ddd5953` |
| `harness/shims.py` | `526fa2bdb36e542076856d734ae6c45adc03ca7ccac8ad384074e614994792b4` |
| `harness-frozen-v1/` (sandboxd harness_sha256, VM2) | `71643ef9c19c3336030e9738b77e60b3d06ea5d567585b766a96d3e7c6dcccd2` |

## Amendment 1 (Sat Sep 26 20:00 PDT, before any held-out Sentinel run)

The Sentinel queue's first two runs (ds-8313 intro, ds-8533 intro) fell back to mandatory-only plans
because every deepseek planner request timed out at the 45 s planner cap while the reviewer bake-off
held 8 concurrent calls to the same model (recorded error: `APITimeoutError`). The 45 s cap is a live
demo limit; each reviewer call is allowed 300 s. For batch runs the planner cap is raised to 180 s.
A run whose only failure is a planner timeout is rerun once; the timed-out runs stay in the database
and are reported next to their reruns. Nothing else changes.
