# Plumbline

A PR gate for ML training code that checks the math, not the diff.

Take three numbers: 3, -4 and 2. Their total size (the p=1 norm) is 3 + 4 + 2 = 9. From DeepSpeed PR #4915 on, `clip_grad_norm_` returned 53 for them. A maintainer approved that PR, its required merge checks passed, and the wrong number shipped in 49 releases over 933 days, until my fix, #8313. Nothing crashed. A test against PyTorch added about two weeks later passed too, because it used p=2, where the bug always gives the right answer (and gradients of 1, where it does at every p).

That is the kind of bug I fix. More than 140 of my pull requests have merged upstream since June (DeepSpeed, Unsloth, sentence-transformers, Ray and others), and most fix the same shape: nothing raises, a number is just wrong. This weekend I traced 23 of them back to the change that caused them (21 PRs, 2 direct commits). They shipped, lived a median of 933 days from merge to fix, and 11 of the 23 came in through a PR with an approving review (one of those approvals was from a credited co-author).

Plumbline's Sentinel reads a PR, picks the math the change must preserve, and runs those checks in gVisor sandboxes on Vultr, on four commits: before the PR, the PR, just before the fix, and the fix. It returns BLOCK, PASS or NOT COVERED, with a signed receipt you can verify offline.

**Vultr Agent Arena, Challenge 1 (Blast Radius Zero), plus the NetBird bonus.** VM backend: two Vultr VMs ([Vultr](#vultr)). Agent LLM calls: all through Vultr Serverless Inference ([Vultr](#vultr)). Control plane: VM1 plans, dispatches to the sandbox VM, judges and signs ([How it works](#how-it-works)). Sandboxes: gVisor containers on VM2, never in the app process ([Security and containment](#security-and-containment)). Web app: a judge starts a live Sentinel run and gets a verdict with a signed receipt ([Demo](#demo)). Verifiable output: `plumbline verify` checks any receipt offline. NetBird: no open ports, PIN-gated access by role, P2P between the VMs ([NetBird](#netbird)).

## Three bugs, three kinds of review

The numbers in this list come from hindsight oracles (`replay/cases/<id>/oracle.py`), written after each fix to show the bug. They are not Sentinel results; those are in [Results](#results).

- **DeepSpeed #4915, fixed by #8313.** `clip_grad_norm_` with p=1 on grads [3,-4] and [2] returns 53, must be 9. One maintainer approval, no AI review. 933 days, 49 releases.
- **DeepSpeed #7953, fixed by #8533.** Muon's Gram Newton-Schulz casts to fp16 before normalizing, so a gradient 10,000x larger gives an output of 0.000 where it must be 4.773. One approval, and a Codex review of a commit that already had the cast. 144 days, 8 releases.
- **sentence-transformers #3654, fixed by #3921.** The GOR loss on a batch of one returns NaN, must be 0. Copilot flagged the right line but predicted a crash, and the thread was resolved with no change. 207 days, 11 releases; the latest release (v6.1.0) still has it as of Sep 26.

A reviewer, human or AI, reads the diff. The bug is a number nobody computed.

## What it is

- **Sentinel**, the PR gate. It maps a diff to invariants (split invariance of a norm, grad-accumulation invariance, a float64 reference, finite output on edge inputs, and so on), runs them in gVisor, and blocks only when a check fails on the head and passed on the base (or the code is new in the head).
- **A historical replay** of 12 real introducing PRs and their 12 fix PRs from my traced bugs, run the same way, next to an AI reviewer given the same diff.
- **A Trojan PR game.** Engineer agents on Vultr write PRs against a small training repo (`target/minigpt`), run CI in gVisor and revise on failures. One role is told to hide a numeric bug. The AI reviewer and Sentinel both look at every PR.
- **A web app** where a judge starts a live Sentinel run (planner on Vultr, checks in gVisor on the sandbox VM) and gets the verdict and a signed receipt, plus the case pages with the historical record and the game arena.

## Demo

Judge URL: https://plumbline.eu1.netbird.services (PIN given in person).

- `/cases/ds-8313`: the hero case. Press **Run Sentinel** and watch the 4-column grid fill: before PASS, PR FAIL, just before the fix FAIL, fix PASS, verdict BLOCK with "p=1, grads [3,-4],[2]: got 53, must be 9".
- Download the run's `receipt.json` and check it offline (Python 3.10+, `pip install cryptography`): `python -m plumbline.cli verify receipt.json --pin keys/plumbline.pub`. A committed example: `docs/evidence/receipt-ds-8313.json`.
- `/`: all 23 traced bugs with the historical record, the AI reviewer and Sentinel columns, and my fix.
- `/arena`: three hand-written PRs with green CI and the reviewer's verdicts. One is a Trojan. Press **R** to reveal.

Judges can browse everything but run Sentinel only on three cases: ds-8313 (blocked in every recorded live run), and ds-8533 and st-3921 (NOT COVERED in the pre-registered run, so a live run may say so). Other runs return 403. One live run at a time: a second click joins the running one.

## How it works

```
 judge's browser
      |  HTTPS + PIN
      v
 NetBird reverse proxy (plumbline.eu1.netbird.services)
      |  WireGuard
      v
 VM1 pl-control (Vultr, ATL)            control plane; no Docker; binds the NetBird IP only
   web app: judge :8080, operator :8081, SQLite
   prepass -> planner -> judge -> ed25519 receipt
   agent loops, AI reviewer, bake-off runner
      |                                     |
      | HTTPS                               | NetBird P2P, bearer token
      v                                     v
 Vultr Serverless Inference            VM2 pl-sandbox (Vultr, ATL)
   deepseek-v4-flash-0731                sandboxd :7070 (NetBird IP only)
                                            |  one container per job
                                            v
                                         gVisor runsc: --network none, read-only root,
                                         all caps dropped, uid 10001, harness read-only
```

Pipeline for one PR:

1. **Prepass** (VM1, deterministic). Finds the touched functions and known bug shapes and adds mandatory checks.
2. **Planner** (LLM on Vultr Serverless Inference). Adds checks from 10 oracle families and a fixed float64 reference registry, as JSON against `plumbline/plan_schema.json`. A validator drops bad checks and lists them on the receipt. The planner picks checks and inputs. It never decides a verdict.
3. **Sandbox** (VM2). One gVisor job per commit. Historical mode runs 4 commits (pre, intro, fix parent, fix). Repo mode runs base and head.
4. **Judge** (VM1). Recomputes every status from the raw metric and threshold, then applies the decision rule below. Thresholds come from float epsilon, for example `8 * eps32 * sqrt(n)`.
5. **Receipt.** ed25519-signed canonical JSON, stored in SQLite and served at `/runs/<id>/receipt.json`.

| base | head | decision |
|---|---|---|
| PASS | FAIL | detected: BLOCK |
| missing (new code) | FAIL | detected_new: BLOCK |
| PASS | PASS | clean |
| FAIL | FAIL | preexisting: warning, never blocks |
| FAIL | PASS | improved |
| missing (new code) | PASS | clean_new |
| ERROR on either side (other than missing) | | inconclusive: shown, never blocks |

Verdict: BLOCK if anything is detected; PASS if every touched surface has at least one clean check; otherwise NOT COVERED. A run where nothing ran is never green.

## Results

Every number here comes from `results/numbers.json` (`make numbers`), `bakeoff/scores.json` (`python bakeoff/score.py`), or the protocol and label files cited next to it.

**Setup.** 12 introducing PRs from my traced bugs: 6 dev (in-sample, the oracle templates were shaped on these) and 6 held-out (frozen before their first Sentinel run). Their 12 fix PRs are the benign control: same function, a correct numeric change. Protocol: `bakeoff/PROTOCOL.md`, pre-registered in commit `278d1e7` before the first scored call.

**Fairness.** The reviewer and Sentinel's planner are the same model (deepseek-v4-flash-0731 on Vultr) and get the same scoped diff, PR title and body. The P1 reviewer and the planner both get the ten bug patterns (the reviewer's version has triggers; the planner's is names only); P0 gets none. The reviewer also gets the full file; the planner also gets the static prepass output and its mandatory checks. Only Sentinel executes code. Reviewer numbers are 3 runs per cell, mean [min, max].

| | Introducing PRs, n=12 | Fix PRs flagged (false alarms), n=12 |
|---|---|---|
| Sentinel, pre-registered | BLOCK 2 (dev 2 of 6, held-out 0 of 6), PASS 1 (a miss), NOT COVERED 9 | 0 (PASS 2, NOT COVERED 10) |
| AI reviewer with the bug list (P1) | caught 3.67 [3, 4] | 1.67 [1, 2] |
| AI reviewer without the bug list (P0) | caught 0.33 [0, 1] | 1.33 [0, 2] |

Dev cases shaped the oracle templates (in-sample). Held-out cases were frozen before their first Sentinel run (commit `278d1e7`). The 10-pattern bug list in the reviewer and planner prompts came from my own fixes, so it is in-sample for every case. The fix PRs merged July to September 2026 and deepseek-v4-flash-0731 was listed on Vultr on August 17 2026 (`created` in `bakeoff/models.json`); its training cutoff is unknown, so it may have seen the fixes, which can only inflate reviewer scores.

- **Sentinel blocked** ds-8313 and ds-8334 (both dev), each with the check failing at the PR and passing at the fix. On ds-8313 the planner returned no usable plan in the pre-registered run, so the block came from the prepass's built-in `clip_grad_norm_` checks (split invariance and a torch reference at p in {2, 1, 3}), which were written knowing this bug. In some live runs the planner's own checks fail too. The ds-8334 block came from a check the planner chose. **The one PASS** (st-3868, held-out) is a miss: the only clean check could not see the bug. **The 9 NOT COVERED** runs: in 4 the planner's checks could not call the real code (wrong arguments), 2 planner outputs were unusable, and 3 modules could not load in the sandbox (missing dependencies or cross-repo imports). Every reason is in `results/miss_reasons.json`.
- **Caught** for the reviewer means right place, right mechanism, and actionable. The 43 outputs that needed judgment were labelled blind (model, prompt and run hidden) by a panel of three independent Claude agents, majority vote, because I delegated the adjudication step; 39 of 43 were unanimous, and the panel agreed with a deepseek LLM judge on 34 of 43 (amendment 6 in `bakeoff/PROTOCOL.md`, labels in `bakeoff/labels.jsonl`).
- **Reading the table.** A plain AI review caught about 0 of the 12. Given my own bug list, it caught more than Sentinel blocked (3.67 vs 2 of 12), but it also called 1 to 2 of my 12 correct fixes buggy and changed its answer between runs. Sentinel flagged none of the fixes (though it covered only 2 of them), and each BLOCK names the check that broke and its numbers. The reviewer guesses; Sentinel computes. They are complementary, and Sentinel's coverage is the open problem.
- The reviewer run had 144 outputs; 1 hit the token cap (kept in the denominator as a miss) and 8 more had malformed JSON that the scorer salvaged in the reviewer's favour.

**Post-hoc, reported separately (amendment 5).** Written after every pre-registered result was seen. A harness fix for constructor arguments the plan leaves out, then the stored st-3921 plans re-executed once with no replanning: intro BLOCK 3 of 12 (dev 3 of 6, held-out 0 of 6), fix PRs 0 of 12. The headline stays 2 of 12.

**Trojan game.**

- **Hand-written library** (built as the fallback, labeled hand-written): 4 of 4 sabotage patches blocked with CI green, 4 of 4 honest patches passed.
- **Stage round** (a demo, not a measurement). Hand-written, because the pre-registered rule needed an agent Trojan that Sentinel blocked, and there was none. CI green on 3 of 3, reviewer approved 3 of 3. Sentinel blocked the Trojan (S1, which re-plants the ds-8313 bug shape in minigpt) with "p=1, grads [3,-4],[2]: got 53, must be 9" and passed both honest cards.
- **Agent round** (pre-registered, `eval/GAME_PROTOCOL.md`): 8 saboteur PRs, CI green 8, reviewer approved 6, Sentinel blocked 0. 8 honest PRs, Sentinel blocked 0. By the protocol, all 8 saboteur PRs count as evasions. 2 malformed episodes were rerun once (not in the protocol; disclosed), and 1 saboteur declined to hide a bug.
- **Post-hoc read by hand** (not a measurement): 6 of the 8 saboteur diffs change no number by more than about 1e-6. The other 2 hide a change Sentinel's checks do not reach: a logged loss value, and a non-default config flag.

## Vultr

- **Two VMs in Atlanta**, Ubuntu 24.04. `pl-control` is the control plane: web app, SQLite, prepass, planner, judge, receipt signing, agent loops, reviewer and bake-off runner. It has no Docker and never runs PR code. `pl-sandbox` runs sandboxd, Docker and gVisor (`runsc` release-20260921.0).
- **Serverless Inference for every LLM call the product makes.** One client, `base_url=https://api.vultrinference.com/v1` (`plumbline/llm.py`), serves the planner, the reviewer, the game agents and the bake-off LLM judge; deepseek-v4-flash-0731 is the model. Outside the product, Claude was my coding assistant and labelled 43 reviewer outputs for scoring (amendment 6). glm-5.3 was the plan, but on Vultr it returned no answer within budget, so it is excluded, with evidence in `bakeoff/models.json`.
- **Receipts carry the infra.** Each receipt records the model calls (model, tokens, latency, response hash) and the Vultr instance id and region from the metadata service.
- Evidence: `docs/evidence/vultr_instances.txt`, `docs/evidence/port_probe.txt`, `docs/evidence/containment_probe.txt`, `docs/evidence/receipt-ds-8313.json`.

## NetBird

1. **No open ports.** The judge URL goes through the NetBird reverse proxy to `pl-control:8080` on its NetBird IP. The VMs sit behind a Vultr firewall group with no inbound rules, and all 12 public port probes on both VMs are closed (`docs/evidence/port_probe.txt`). The apps and sandboxd bind the NetBird IP only.
2. **Gated access by role.** Judges get a PIN at the proxy (no PIN or a wrong PIN: 401). The judge role can browse everything but run only the three hero cases (other runs return 403). The operator role is my laptop as a NetBird peer on `:8081`. The default allow-all policy is off.
3. **Peer-to-peer.** Sandbox jobs go VM1 to VM2 over a direct WireGuard P2P link (`docs/evidence/netbird_p2p.txt`: `netbird status -d` on pl-control shows pl-sandbox with Connection type: P2P). My laptop reaches `pl-sandbox` P2P, but its link to `pl-control` is relayed, so the operator UI makes no P2P claim.
4. **Lifecycle-bound URLs.** Not built (cut for time).

![NetBird peers](docs/img/netbird-peers.jpg)
The three peers (`pl-control`, `pl-sandbox`, my laptop), all connected, each in its own group.

![NetBird reverse proxy service](docs/img/netbird-proxy-service.jpg)
The public service `plumbline.eu1.netbird.services` routes to `pl-control`'s NetBird IP on :8080, with PIN auth on.

![NetBird access logs](docs/img/netbird-access-logs.jpg)
Requests to the judge URL, each authenticated with the PIN (client IPs masked).

![NetBird access policies](docs/img/netbird-policies.jpg)
Default allow-all is off. Two one-way rules remain: laptop to the VMs on 3 TCP ports, and `pl-control` to `pl-sandbox` on TCP 7070 only. VM2 cannot reach VM1.

## Security and containment

- **Every check, CI run and agent-written file runs on VM2 in gVisor**, one container per job, removed afterwards: `--runtime=runsc-trace --network none --read-only`, tmpfs work dirs, `--cap-drop ALL`, `no-new-privileges`, uid 10001, 1 CPU, 3 GB, 256 pids, harness mounted read-only, no secrets in the environment (`sandboxd/server.py`).
- **Containment probe** (`docs/evidence/containment_probe.py` and `.txt`, run with sandboxd's exact flags): no egress (1.1.1.1), no DNS, no cloud metadata service, no route to VM1's NetBird IP, no writes to the root, job or harness directories; uid 10001, gVisor kernel.
- **The sandbox is not trusted.** The judge on VM1 recomputes every status from the raw metric, so a sandbox that reports PASS on a bad metric is still detected (`tests/test_judge.py`). A timeout or error never counts as a detection or a pass.
- **Receipts.** Canonical JSON with the plan, every check result, image and harness hashes, model calls, Vultr instance and region, and git sha, signed with ed25519 on VM1. `verify` exits 0 on a valid receipt and 1 if any signed field changes. The public key is `keys/plumbline.pub`.
- **sandboxd** needs a bearer token, listens only on the NetBird IP, and runs at most 3 jobs at once. Docker container logs are off (`--log-driver none`); gVisor keeps per-job strace logs, and a cron job (`infra/cloud-init/sandbox.yaml.tmpl`) deletes job directories and logs older than 12 hours.
- **The judge URL** allows one live run at a time (a second click joins the running one), refuses cross-site POSTs, and cannot be framed.
- **Secrets** live in a gitignored `.env` and 0600 files on the VMs. The NetBird setup key was revoked after both VMs joined.

## Limits

- **Coverage is low.** Pre-registered, Sentinel blocked 2 of 12 introducing PRs, both dev cases. Held-out: 0 of 6. 9 of 12 are NOT COVERED, which is a result, not a pass. Blind plans often fail to bind to real code.
- **What Sentinel will not catch:** bugs that satisfy every checked relation; new math with no stated invariant; corners the planner does not pick; errors below the derived float tolerance; intentional behaviour changes the PR does not declare (a false positive source); data poisoning and performance bugs; an adaptive adversary aiming at an uncovered invariant; a wrong template; surfaces the fixtures cannot bind (reported NOT COVERED, never PASS).
- **CPU only.** No GPU, Triton, NCCL, multi-node or GPU `torch.compile` paths. On CPU the DeepSpeed #7953 bug shows only in eager mode (torch.compile hides it); GPU compile was not tested.
- **In-sample.** The oracle templates were shaped on the dev cases, the bug list came from my own fixes, and the prepass's `clip_grad_norm_` checks were written knowing the ds-8313 bug.
- **One model.** Reviewer, planner and agents are all deepseek-v4-flash-0731. The models may have seen the fixes.
- **Agent saboteurs beat Sentinel in the agent round** (0 of 8 blocked), though by a post-hoc read most did not change any number. The stage round uses hand-written patches and says so.
- **Threat model gap.** PR code and the harness share a process in the sandbox, so a deliberately malicious PR could read the job file, write fake result lines to the harness's output and exit before the harness does. The judge recomputes statuses and the sandbox has no network or secrets, but this is not solved; the fix is one child process per check, read by the parent.
- **DeepSpeed #4915 was latent on DeepSpeed's own path.** No in-tree caller passes p other than 2, so the public API returned a wrong number; DeepSpeed's own training was not affected.
- **Ops.** The NetBird proxy link to VM1 is relayed, so the first request after idle is slow; a keep-warm loop covers it. The web apps and sandboxd run as root on their VMs (the job containers do not).

## Reproduce

You need a Vultr account with Serverless Inference, a NetBird account, and two Ubuntu 24.04 VMs.

No cloud needed:

```bash
pip install -r requirements.txt                                      # or just cryptography, for verify
python3 tests/test_judge.py                                          # decision rule, lying sandbox, tamper
python -m plumbline.cli verify docs/evidence/receipt-ds-8313.json --pin keys/plumbline.pub
```

Infra:

```bash
cp .env.example .env    # fill in VULTR_INFERENCE_API_KEY, NETBIRD_SETUP_KEY, ...
set -a; . ./.env; set +a
sed "s|\${NB_SETUP_KEY}|$NETBIRD_SETUP_KEY|" infra/cloud-init/control.yaml.tmpl > infra/cloud-init/control.yaml   # gitignored
sed "s|\${NB_SETUP_KEY}|$NETBIRD_SETUP_KEY|" infra/cloud-init/sandbox.yaml.tmpl > infra/cloud-init/sandbox.yaml   # gitignored
# create both VMs with that user_data and a firewall group with zero inbound rules
# after both VMs join NetBird: add your laptop as a peer, ssh aliases pl-control / pl-sandbox (root, NetBird IPs);
# VM1: python3 -m venv /opt/plumbline/.venv && /opt/plumbline/.venv/bin/pip install -r requirements.txt
# VM2: python3 -m venv /opt/sbx && /opt/sbx/bin/pip install fastapi uvicorn
# both: /etc/plumbline/env (0600) with SANDBOXD_TOKEN; VM1 also VULTR_INFERENCE_API_KEY and SANDBOXD_URL
make deploy-sandbox     # harness, sandboxd, minigpt to VM2; then on VM2:
                        # docker build --network host -t pl/runner:cpu -f sandboxd/runner.Dockerfile .
python bakeoff/fetch_inputs.py   # laptop, needs gh: the 24 PR inputs (deploy-control copies them)
make deploy-control     # app, planner, judge, agents to VM1
```

Runs (VM1 unless noted):

```bash
python -m plumbline.cli keygen /etc/plumbline/signing.pem   # prints the public key
python -m plumbline.cli hist ds-8313                        # one historical Sentinel run
python bakeoff/run.py --runs 3 --workers 8                  # AI reviewer, 144 reviews
python -m plumbline.cli queue harness-frozen-v2             # Sentinel on the 24 PRs
python -m plumbline.agents --sabotage-check                 # hand library through CI + Sentinel
python -m plumbline.agents --overnight                      # agent round
make numbers                                                # laptop: results/numbers.json
```

Frozen harnesses are git snapshots of `harness/` copied to VM2 (v1 = `278d1e7`, v2 = `e5c88ee` for the pre-registered headline, v3 = `01b1fd6` for the game, v4 = `82538e5` for amendment 5), e.g. `git archive e5c88ee harness | ssh pl-sandbox 'mkdir -p /opt/plumbline/harness-frozen-v2 && tar -x --strip-components=1 -C /opt/plumbline/harness-frozen-v2'`. Their sha256 values are in `bakeoff/PROTOCOL.md` (all four, see Errata) and in every receipt.

## Built this weekend vs prior work

**Built this weekend (Sat Sep 26 09:00 to Sun Sep 27 12:00 PDT).** Everything in this repository's git history: Sentinel (planner, oracle templates, harness, judge), sandboxd and the gVisor setup, the agents and the game target, the historical replay harness and hindsight replays, the reviewer bake-off and its results, receipts and `verify`, the web app, the Vultr and NetBird setup, and every number in this README. Saturday afternoon's prototype and replay scripts were written during the event with AI assistance and are imported in the first commit `219e9ef` at Sat Sep 26 18:47 PDT. Development used Claude Code (every commit carries a `Co-Authored-By: Claude` trailer), and Claude agents labelled 43 reviewer outputs for scoring (amendment 6). The tracing of the 23 bugs to their introducing changes was done this weekend.

**Prior work, used but not built this weekend.** (1) My upstream fixes: more than 140 merged pull requests since June 2026 in DeepSpeed, Unsloth, unsloth-zoo, sentence-transformers, Ray and others. They are why I know these bug shapes and they are the historical test cases. The only upstream code in the repo is `proto/intro/ds-8313.diff`, the public diff of DeepSpeed PR #4915 (Apache-2.0, not my code), used as a prototype input. All other upstream files are fetched at pinned public commits at run time. (2) My pre-event checklist of ten numerical bug patterns (written Sep 17, not in this repo). It supplies the ten pattern names in the prompts and the grep heuristics behind the prepass regexes (`plumbline/prepass.py`, which calls it patterns.md). The code that uses them was written this weekend. (3) Third-party software: gVisor, Docker, the NetBird client, FastAPI, PyTorch, htmx, the openai Python client, IBM Plex (OFL); open models served by Vultr Serverless Inference. The game target's shape is inspired by karpathy/nanoGPT (MIT); its code was written this weekend.

## Repo layout

```
plumbline/        VM1 control plane: app, prepass, planner, judge, receipts, agents, llm client, cli, numbers
  prompts/        planner, engineer, saboteur prompts
  templates/ static/   web app (htmx, IBM Plex)
sandboxd/         VM2 job server and the runner image (runner.Dockerfile)
harness/          runs only inside gVisor: job runner, oracle families, adapters, references
  surface/        check surface for the game target
target/minigpt/   small GPT that learns 3-digit addition, the repo the agents send PRs to
sabotage/         hand library: S1 to S4 (hidden bugs), H1 to H4 (honest)
bakeoff/          reviewer bake-off: protocol, cases, prompts, runner, scoring, raw outputs
eval/             game protocol, agent round and hand round records
replay/cases/     one hindsight oracle per traced bug (written after the fix; upstream code not committed)
corpus/           case descriptions for the gold-plan corpus
proto/            Saturday prototype, imported in the first commit
results/          numbers.json (single source for UI and README), cases.json, miss_reasons.json
infra/            cloud-init templates and systemd units
keys/             receipt public key
tests/            judge, planner, prepass, harness binding and adapter tests
docs/             screenshots and evidence
```

## License

MIT (`LICENSE`). IBM Plex fonts are under the SIL Open Font License (`plumbline/static/fonts/OFL.txt`).

## Acknowledgements

- The maintainers of DeepSpeed, Unsloth, sentence-transformers and Ray, who reviewed and merged the fixes these cases come from.
- Related work this builds on: CTRL-ALT-DECEIT, SHADE-Arena, ResearchArena, Auditing Sabotage Bench, AI control, metamorphic testing, SWR-Bench.
- karpathy/nanoGPT (MIT) for the shape of the game target.
- Vultr and NetBird for the infrastructure, and Cerebral Valley for the event.
