"""Tests for src/factors.py — factor weight matrix construction."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_simulation import SimulatedMarketData
from src.factors import FACTOR_COUNTS, build_factor_weight_matrix


@pytest.fixture(scope="module")
def characteristics():
    sim = SimulatedMarketData(N=20, K=7, T=30, seed=0)
    return sim.generate_market_caps(), sim.generate_bm_ratios(), sim.generate_accounting_data()


@pytest.mark.parametrize("K", FACTOR_COUNTS)
def test_supported_factor_counts_give_k_rows(characteristics, K):
    caps, bm, acct = characteristics
    W = build_factor_weight_matrix(K=K, market_caps=caps, bm_ratios=bm, accounting=acct)
    assert W.shape == (K, 20)


@pytest.mark.parametrize("K", [0, 2, 4, 6, 8])
def test_unsupported_factor_counts_raise(characteristics, K):
    caps, bm, acct = characteristics
    with pytest.raises(ValueError, match="K must be one of"):
        build_factor_weight_matrix(K=K, market_caps=caps, bm_ratios=bm, accounting=acct)
