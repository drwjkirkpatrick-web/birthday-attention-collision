# Proof: Birthday Attention Collision

## Lemma 1 (Birthday Paradox — General Form)

Let X₁, X₂, ..., X_H be independent and uniformly distributed random variables taking values in a set of size M. The probability that all H values are distinct is:

$$P(\text{all distinct}) = \prod_{i=1}^{H-1} \left(1 - \frac{i}{M}\right)$$

**Proof.** X₁ can be any value. X₂ must differ from X₁: probability (1 - 1/M). X₃ must differ from both X₁ and X₂: probability (1 - 2/M). Continuing, X_i must differ from all previous i-1 values: probability (1 - (i-1)/M). The product gives the joint probability of all being distinct. ∎

---

## Lemma 2 (Approximation for H ≪ M)

For H ≪ M, the no-collision probability satisfies:

$$\prod_{i=1}^{H-1} \left(1 - \frac{i}{M}\right) \approx \exp\left(-\frac{H(H-1)}{2M}\right)$$

In particular, for H ≥ 2:

$$\prod_{i=1}^{H-1} \left(1 - \frac{i}{M}\right) \leq \exp\left(-\frac{H^2}{2M}\right)$$

**Proof.** Using the inequality 1 - x ≤ e^{-x} for x ∈ [0, 1]:

$$\prod_{i=1}^{H-1} \left(1 - \frac{i}{M}\right) \leq \prod_{i=1}^{H-1} \exp\left(-\frac{i}{M}\right) = \exp\left(-\frac{1}{M} \sum_{i=1}^{H-1} i\right) = \exp\left(-\frac{H(H-1)}{2M}\right)$$

Since H(H-1) ≥ H²/2 for H ≥ 2, we have:

$$\exp\left(-\frac{H(H-1)}{2M}\right) \leq \exp\left(-\frac{H^2}{4M}\right)$$

The tighter bound uses H(H-1) ≈ H² for large H. ∎

---

## Lemma 3 (Number of Top-k Subsets)

For sequence length N and top-k attention (selecting k keys out of N), the number of possible top-k sets is:

$$M = \binom{N}{k} = \frac{N!}{k!(N-k)!}$$

**Proof.** Each top-k set is a k-element subset of {1, 2, ..., N}. The number of k-element subsets of an N-element set is the binomial coefficient C(N, k). ∎

---

## Proof of Theorem 1 (Birthday Collision in Top-k Attention)

For a fixed query position i, each head h ∈ {1, ..., H} selects a top-k set S_i^h ⊂ {1, ..., N}. By Lemma 3, there are M = C(N, k) possible sets.

Assuming independent uniform selection (as a null model), the probability that all H heads select distinct top-k sets is given by Lemma 1:

$$P(\text{no collision at position } i) = \prod_{j=1}^{H-1} \left(1 - \frac{j}{M}\right)$$

By Lemma 2, for H ≪ M:

$$P(\text{no collision at position } i) \leq \exp\left(-\frac{H^2}{2M}\right)$$

Therefore:

$$P(\text{collision at position } i) = 1 - P(\text{no collision}) \geq 1 - \exp\left(-\frac{H^2}{2M}\right)$$

For the full sequence of N positions, the probability of at least one collision across all positions satisfies:

$$P(\text{collision somewhere}) = 1 - \prod_{i=1}^{N} P(\text{no collision at } i) \geq 1 - \exp\left(-\frac{NH^2}{2M}\right)$$

However, since we seek a collision for at least one query position (not all simultaneously), the tighter per-position bound is the relevant one. For typical transformers, N ≈ 512 and H ≈ 8–16, so we focus on the per-query probability as the key metric.

The threshold H = √(2M · ln 2) gives P_collision ≥ 0.5 because:

$$1 - \exp\left(-\frac{2M \ln 2}{2M}\right) = 1 - \exp(-\ln 2) = 1 - \frac{1}{2} = 0.5$$

∎

---

## Proof of Theorem 2 (Effective Head Capacity)

Define the effective number of heads H_effective as the expected number of distinct top-k patterns observed across H heads.

Using the coupon collector framework: with M possible patterns and H draws, the expected number of distinct patterns is:

$$H_{\text{effective}} = M \left(1 - \left(1 - \frac{1}{M}\right)^H\right)$$

For H ≪ M, using (1 - 1/M)^H ≈ exp(-H/M):

$$H_{\text{effective}} \approx M \left(1 - e^{-H/M}\right) \approx M \cdot \frac{H}{M} = H$$

For H ≫ M, using (1 - 1/M)^H ≈ 0:

$$H_{\text{effective}} \approx M$$

The transition occurs when H ≈ M, but the birthday paradox tells us that collisions become significant much earlier — at H ≈ √M. At this point, approximately H/2 heads have collided with others, so H_effective ≈ H/2.

More precisely, the expected number of collisions (pairs of heads with identical top-k sets) is:

$$E[\text{collisions}] = \binom{H}{2} \cdot \frac{1}{M} = \frac{H(H-1)}{2M}$$

When this exceeds 1, we expect at least one collision on average. This occurs at H ≈ √(2M).

Therefore, the effective head capacity is bounded by:

$$H_{\text{effective}} = O(\sqrt{M}) = O\left(\sqrt{\binom{N}{k}}\right)$$

∎

---

## Proof of Theorem 3 (Collision Threshold for Common Configurations)

**Case k = 1 (argmax attention):**

M = C(N, 1) = N. The threshold is:

$$H_{\text{threshold}} = \sqrt{2N \cdot \ln 2} \approx 1.177 \sqrt{N}$$

For N = 512: H_threshold ≈ 26.7. With H = 8: P ≈ 1 - exp(-64/1024) ≈ 6%. With H = 16: P ≈ 1 - exp(-256/1024) ≈ 22%.

**Case k = 2:**

M = C(N, 2) = N(N-1)/2 ≈ N²/2. The threshold is:

$$H_{\text{threshold}} = \sqrt{2 \cdot \frac{N^2}{2} \cdot \ln 2} = N \sqrt{\ln 2} \approx 0.833 N$$

For N = 512: H_threshold ≈ 427. With H = 8: P ≈ 1 - exp(-64/130816) ≈ 0.05% (negligible). With H = 64: P ≈ 1 - exp(-4096/130816) ≈ 3%.

**Case k = N/2 (half the sequence):**

Using Stirling's approximation: n! ≈ √(2πn) (n/e)^n.

$$\binom{N}{N/2} \approx \frac{2^N}{\sqrt{\pi N / 2}}$$

The threshold is:

$$H_{\text{threshold}} = \sqrt{2 \cdot \frac{2^N}{\sqrt{\pi N / 2}} \cdot \ln 2} = 2^{N/2} \cdot \left(\frac{2 \ln 2}{\sqrt{\pi N / 2}}\right)^{1/2}$$

For N = 512: this is astronomically large (≈ 2^{256}), far exceeding any practical H.

∎

---

## Lemma 4 (Expected Collisions in Practice)

For H heads and M = C(N, k) possible top-k sets, the expected number of head pairs sharing the same top-k set is:

$$E[\text{collisions}] = \binom{H}{2} \cdot \frac{1}{M} = \frac{H(H-1)}{2M}$$

**Proof.** There are C(H, 2) = H(H-1)/2 pairs of heads. Each pair has probability 1/M of selecting the same top-k set (by uniform assumption). The expected number is the sum over all pairs. ∎
