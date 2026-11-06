# Tight-crop A/B pipeline

State: failed
Stage: train_A_tiny
PID: 28152
Updated UTC: 2026-11-01T19:49:44.913012+00:00
Elapsed: 1686.6 seconds

Four fixed E2 runs, followed by paired valid and historical 044 evaluation.
`last.pt` E2 EMA is the treatment; `best.pt` is diagnostic only.

- A_tiny: running; epochs 1/2; retries 0/1
- B_tiny: pending; epochs 0/2; retries 0/1
- A_base: pending; epochs 0/2; retries 0/1
- B_base: pending; epochs 0/2; retries 0/1

Error: RuntimeError: A_tiny has STOP marker; refusing automatic retry

Read `status.json` for machine-readable progress and `logs/` for command output.
