Plumbline (Trojan PR), Vultr Agent Arena hackathon. Solo. Submit by Sun 11:30 PDT. Claude usage is limited: work lean.

Never read PLAN.md whole (it is long). Start of each block: `tail -25 NOTES.md`, then PLAN.md section 9 (schedule)
and section 13 (rules) via sed ranges, then only the sections the current block lists.
Case facts come from CASES.md and research/cases_facts.json, never from memory.
SESSION_KICKOFF.md has the kickoff and per-block resume prompts.

Hard rules:
- All LLM calls via Vultr Serverless Inference. All untrusted code runs in gVisor on VM2; VM1 never executes PR code.
- Never print, commit or paste secrets. Source .env, never cat it. Private files are in .gitignore; check `git status` before commits.
- Every number on stage, in the UI or README comes from results/numbers.json or a NOTES.md line with its command.
- GitHub read-only for upstream repos. Never touch ~/Desktop/oss-contributions.
- No em dashes anywhere.
