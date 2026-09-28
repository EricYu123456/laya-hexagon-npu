# Service acceptance clarification

After the supplementary 15-decision result, and before HTTP deployment testing,
the service verifier was changed to distinguish integration from fidelity.
Originally its single `passed` flag combined both. This is a post-result
clarification, not a pass for the failed development accuracy gate.

The unchanged [supplementary report](supplementary-development-npu.json) records
**1/15 decision differences (6.6667%)**, mean TV **0.9462%**, and `passed: false`.
The long-English department choice changes from `technical` to `billing`; its
TV is 2.6835%. This failure is retained. No graph or calibration changed after it.

The primary fidelity gates remain unchanged: the declared 1,840-decision selection
has 3.5326% mismatch/2.5021% mean TV, and the predeclared synthetic 80-decision long
validation has 0% mismatch/2.0614% mean TV. Both pass their original 5% thresholds.

HTTP integration requires exact replay of the standalone NPU answers, correct
input usage, actual per-request NPU execution, bucket switching, invalid-input
rejection, and zero fallbacks/cache misses. `integration_passed` (and its `passed`
alias) reports only that result. `development_fidelity_passed` separately retains
the 15-decision accuracy failure. Cgroup memory and process continuity are checked
separately. Service integration cannot turn an NPU accuracy failure into a pass.
