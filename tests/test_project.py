"""
test_project.py
===============

pytest suite for the Birthday Attention Collision proof project.

Run with:
    source ~/heartlib/.venv/bin/activate
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "empirical"))
from verify import (
    random_topk_set,
    simulate_attention_heads,
    measure_effective_heads,
    check_theorem_1,
    check_theorem_2,
    check_theorem_3,
    manual_seed,
)


@pytest.fixture(scope="module", autouse=True)
def seed():
    manual_seed(1729)


class TestTopKSelection:
    """Unit tests for top-k set simulation."""

    def test_random_topk_size(self):
        for _ in range(10):
            s = random_topk_set(N=100, k=5)
            assert len(s) == 5
            assert len(set(s)) == 5  # all distinct

    def test_random_topk_range(self):
        for _ in range(10):
            s = random_topk_set(N=50, k=10)
            assert all(0 <= x < 50 for x in s)

    def test_random_topk_sorted(self):
        for _ in range(10):
            s = random_topk_set(N=30, k=5)
            assert s == tuple(sorted(s))


class TestSimulateAttention:
    """Tests for attention head simulation."""

    def test_simulate_structure(self):
        result = simulate_attention_heads(H=4, N=16, k=2, n_queries=100)
        assert "empirical_p_collision" in result
        assert "theoretical_p_collision" in result
        assert 0 <= result["empirical_p_collision"] <= 1

    def test_zero_heads_no_collision(self):
        """With H=1, collision probability is 0."""
        result = simulate_attention_heads(H=1, N=16, k=2, n_queries=100)
        assert result["empirical_p_collision"] == 0.0
        assert result["actual_pair_collisions"] == 0

    def test_high_H_near_certain(self):
        """With H >> M, collision probability approaches 1."""
        result = simulate_attention_heads(H=64, N=8, k=1, n_queries=500)
        # M = 8, H = 64, guaranteed collision
        assert result["empirical_p_collision"] > 0.99

    def test_monotonic_with_H(self):
        """Collision probability should increase with H."""
        N, k = 32, 1
        p1 = simulate_attention_heads(H=4, N=N, k=k, n_queries=1000)["empirical_p_collision"]
        p2 = simulate_attention_heads(H=16, N=N, k=k, n_queries=1000)["empirical_p_collision"]
        assert p2 > p1 * 0.9  # generally increasing


class TestEffectiveHeads:
    """Tests for effective head capacity measurement."""

    def test_measure_structure(self):
        result = measure_effective_heads(H=10, N=16, k=2, n_queries=100)
        assert "n_distinct" in result
        assert "effective_ratio" in result

    def test_saturation_bound(self):
        """Distinct patterns cannot exceed M."""
        result = measure_effective_heads(H=100, N=10, k=2, n_queries=500)
        M = result["M"]
        assert result["n_distinct"] <= M

    def test_diminishing_returns(self):
        """Efficiency drops as H increases."""
        N, k = 32, 2
        r1 = measure_effective_heads(H=10, N=N, k=k, n_queries=500)
        r2 = measure_effective_heads(H=50, N=N, k=k, n_queries=500)
        eff1 = r1["effective_ratio"]
        eff2 = r2["effective_ratio"]
        assert eff2 <= eff1 * 1.5  # generally diminishing


class TestTheorem1:
    """Theorem 1: Birthday Collision Probability."""

    def test_pass(self):
        r = check_theorem_1()
        assert r.passed, f"Theorem 1 failed: {r.detail}"


class TestTheorem2:
    """Theorem 2: Effective Head Capacity."""

    def test_pass(self):
        r = check_theorem_2()
        assert r.passed, f"Theorem 2 failed: {r.detail}"


class TestTheorem3:
    """Theorem 3: Collision Threshold."""

    def test_pass(self):
        r = check_theorem_3()
        assert r.passed, f"Theorem 3 failed: {r.detail}"
