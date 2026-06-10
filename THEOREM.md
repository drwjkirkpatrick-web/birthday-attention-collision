# Theorem: Birthday Attention Collision

**Status:** Proved and empirically verified
**Target venue:** ICLR/NeurIPS workshop or arXiv preprint
**Date:** 2026-06-09
**Source paper(s):**
- Vaswani et al. (2017), "Attention Is All You Need"
- Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding"

---

## Notation

| Symbol | Type | Meaning |
|---|---|---|
| H | int | number of attention heads |
| N | int | sequence length (number of tokens) |
| d_h | int | per-head dimension (d_model / H) |
| k | int | top-k attention cutoff |
| C(N, k) | int | binomial coefficient: N choose k |
| S_i^h | set | top-k key indices attended by head h at query i |
| P_collision | float | probability that two heads share the same top-k set |

---

## Theorem 1 (Birthday Collision in Top-k Attention)

In a multi-head attention layer with H heads and sequence length N, for each query position i ∈ {1, ..., N}, each head h selects a top-k set S_i^h ⊂ {1, ..., N} of key positions. The total number of possible distinct top-k sets is:

$$M = \binom{N}{k}$$

If the H heads select their top-k sets independently and uniformly at random from the M possibilities, then the probability that at least two heads share the same top-k set for at least one query position satisfies:

$$P_{\text{collision}} \geq 1 - \exp\left(-\frac{H^2}{2M}\right)$$

In particular, when H ≥ √(2M · ln 2), we have P_collision ≥ 0.5.

## Theorem 2 (Effective Head Capacity)

For top-k attention with k ≪ N, the number of effectively independent attention patterns is bounded by:

$$H_{\text{effective}} = O\left(\sqrt{\binom{N}{k}}\right)$$

Heads added beyond this threshold produce redundant attention patterns with probability approaching 1 as H grows.

## Theorem 3 (Collision Threshold for Common Configurations)

For practical transformer configurations:
- **k = 1 (argmax):** Collision threshold at H ≈ √(2N)
- **k = 2:** Collision threshold at H ≈ N/√2
- **k = N/2 (half the sequence):** Collision threshold at H ≈ 2^{N/2} / √(πN/2)

For typical values N = 512, H = 8: with k = 1, the collision probability is approximately 1 - exp(-64/1024) ≈ 6%. With H = 16, it rises to 1 - exp(-256/1024) ≈ 22%.

---

## Proof Sketch

**Theorem 1:** This is a direct application of the birthday paradox. With M possible "birthdays" (top-k sets) and H "people" (heads), the probability of no collision is approximately exp(-H²/(2M)) for H ≪ M. The complement gives the collision probability.

**Theorem 2:** Once the collision probability exceeds 0.5, more than half of all possible pairwise head combinations have collided on some query position. Additional heads can only produce patterns already represented by existing heads (with high probability), making them redundant.

**Theorem 3:** Substitute k values into the general formula H_threshold = √(2M · ln 2) and use Stirling's approximation for large N.

The full proofs live in `proof/proof.md`.

---

## Open Questions

1. **Non-uniform attention:** Real attention is not uniform — some key positions are more likely to be in top-k sets. How does this affect the collision rate?
2. **Soft collision:** We define collision as identical top-k sets. What about approximate collision (Jaccard similarity > 0.8)?
3. **Cross-layer redundancy:** Do heads in different layers also collide, or is the collision intra-layer only?
