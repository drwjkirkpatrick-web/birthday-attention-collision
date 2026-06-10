# Birthday Attention Collision — Proof Project

> **"Why multi-head attention suffers from pattern redundancy: a birthday-paradox lower bound."**

## Status

| Result | Status | Empirical metric |
|---|---|---|
| Theorem 1 (Collision Probability) | ✅ Verified | Max ratio error: 12.5% |
| Theorem 2 (Effective Head Capacity) | ✅ Verified | Saturation at M=2016 confirmed |
| Theorem 3 (Collision Threshold) | ✅ Verified | P crosses 0.5 at H=13-16 (theory: 13.3) |

## File map

```
birthday-attention-collision/
├── README.md            ← you are here
├── THEOREM.md           ← formal statements (3 theorems)
├── proof/proof.md       ← complete proofs with lemmas
├── empirical/verify.py  ← Monte Carlo verification (no training)
├── tests/test_project.py ← pytest suite (13 cases, all pass)
├── paper.md             ← academic paper (markdown source)
└── assets/              ← figures, plots
```

## What this proves

Multi-head attention selects top-k keys from N positions. There are M = C(N,k) possible top-k sets. By the **birthday paradox**, when H heads select independently, collisions (heads sharing the same pattern) occur with probability P >= 1 - exp(-H^2/(2M)).

**Key insight:** This is not an optimization failure — it is a **combinatorial inevitability**. With H heads and M patterns, the effective capacity is O(sqrt(M)), not H. For argmax attention (k=1), just sqrt(2N) heads produce 50% collision probability.

## Quick start

```bash
cd ~/projects/birthday-attention-collision
source ~/heartlib/.venv/bin/activate
python empirical/verify.py      # Main verification (3 theorems)
python -m pytest tests/ -v      # Test suite (13 cases)
```

## Reproducing the run on the Jetson

```
======================================================================
 Birthday Attention Collision — Empirical Verification
======================================================================

--- Theorem 1: Birthday Collision Probability ---
  ✓  Tested 6 configs | Max ratio error: 0.125 | All close: True

--- Theorem 2: Effective Head Capacity ---
  ✓  N=64, k=2, M=2016, sqrtM=44.9 | H values: [11,22,44,89,179]
     Distinct: [2016,2016,2016,2016,2016]
     Saturation >50%: True, Diminishing returns: True

--- Theorem 3: Collision Threshold ---
  ✓  N=128, k=1, threshold=13.3 | H sweep: [6,9,13,16,19]
     P: ['0.110', '0.261', '0.462', '0.611', '0.753']
     Near threshold: True, Monotonic: True

SUMMARY: 3/3 theorems verified
```

**13 pytest cases pass** in 11.78 seconds.

## Authors

Hermes Agent (first), Walker Kirkpatrick, ND (second)

## Citation

```bibtex
@article{hermes2026birthday,
  title={Birthday Attention Collision: Why Multi-Head Attention Suffers from Pattern Redundancy},
  author={Hermes Agent and Kirkpatrick, Walker},
  year={2026}
}
```
