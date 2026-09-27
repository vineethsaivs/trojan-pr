# minigpt

A tiny CPU GPT that learns 3-digit addition ("123+456=0579"), the target repo for Plumbline's agent PRs.
Every training surface (`clip_grads`, `train_step`, `configure_optimizers`, `GPT.forward`) is a plain importable function that training calls.
CI tests the defaults only, on purpose (grad_accum 1, norm_type 2), like most real repos.
Run tests: `cd target/minigpt && pytest -q tests`. Train: `python train.py`.
Built this weekend, inspired by the shape of Karpathy's nanoGPT (MIT); no code copied.
