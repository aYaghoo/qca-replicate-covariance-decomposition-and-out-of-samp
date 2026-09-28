import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def compute_market_weights(market_caps: pd.Series) -> np.ndarray:
    """Compute value-weighted market portfolio weights.

    Parameters
    ----------
    market_caps: Series of market caps for N stocks

    Returns
    -------
    w: (N,) array of value weights (sum to 1)
    """
    caps = market_caps.values.astype(float)
    return caps / caps.sum()


def double_sort_smb_hml(
    market_caps: pd.Series,
    bm_ratios: pd.Series,
    returns: pd.DataFrame,
    size_breakpoint: float = 0.5,
    bm_breakpoints: Tuple[float, float] = (0.30, 0.70),
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct SMB and HML factor weight vectors from double sort.

    Creates 6 value-weighted portfolios from size x book-to-market double sort.
    SMB = (SL + SM + SH)/3 - (BL + BM + BH)/3
    HML = (SH + BH)/2 - (SL + BL)/2

    Parameters
    ----------
    market_caps: Series(N) of market capitalizations
    bm_ratios: Series(N) of book-to-market ratios
    returns: DataFrame(T, N) of stock returns
    size_breakpoint: median split breakpoint for size
    bm_breakpoints: low/high breakpoints for BM sort

    Returns
    -------
    w_smb: (N,) weight vector for SMB factor
    w_hml: (N,) weight vector for HML factor
    """
    N = len(market_caps)
    caps = market_caps.values.astype(float)
    bm = bm_ratios.values.astype(float)

    # Size split at median
    size_med = np.percentile(caps, size_breakpoint * 100)
    small = caps <= size_med
    big = caps > size_med

    # BM sort into low (bottom 30%), mid (middle 40%), high (top 30%)
    bm_low = np.percentile(bm, bm_breakpoints[0] * 100)
    bm_high = np.percentile(bm, bm_breakpoints[1] * 100)
    low_bm = bm <= bm_low
    high_bm = bm >= bm_high
    mid_bm = ~low_bm & ~high_bm

    def vw_weights(mask: np.ndarray) -> np.ndarray:
        w = np.zeros(N)
        if mask.sum() == 0:
            return w
        w[mask] = caps[mask] / caps[mask].sum()
        return w

    # 6 portfolios
    w_SL = vw_weights(small & low_bm)
    w_SM = vw_weights(small & mid_bm)
    w_SH = vw_weights(small & high_bm)
    w_BL = vw_weights(big & low_bm)
    w_BM = vw_weights(big & mid_bm)
    w_BH = vw_weights(big & high_bm)

    w_smb = (w_SL + w_SM + w_SH) / 3 - (w_BL + w_BM + w_BH) / 3
    w_hml = (w_SH + w_BH) / 2 - (w_SL + w_BL) / 2

    return w_smb, w_hml


def decile_factor_weights(
    characteristic: np.ndarray,
    market_caps: np.ndarray,
    long_low: bool = True,
) -> np.ndarray:
    """Construct long-short factor weights from decile sort on a characteristic.

    Factor = R_low_decile - R_high_decile (or reverse if long_low=False).
    Uses value-weighted portfolios within each decile.

    Parameters
    ----------
    characteristic: (N,) array of firm characteristic values
    market_caps: (N,) array of market capitalizations
    long_low: if True, long bottom decile and short top decile

    Returns
    -------
    w: (N,) net weight vector
    """
    N = len(characteristic)
    q_low = np.percentile(characteristic, 10)
    q_high = np.percentile(characteristic, 90)

    low_mask = characteristic <= q_low
    high_mask = characteristic >= q_high

    def vw_w(mask: np.ndarray) -> np.ndarray:
        w = np.zeros(N)
        if mask.sum() > 0:
            w[mask] = market_caps[mask] / market_caps[mask].sum()
        return w

    w_low = vw_w(low_mask)
    w_high = vw_w(high_mask)

    if long_low:
        return w_low - w_high
    else:
        return w_high - w_low


def investment_factor_weights(
    size_char: np.ndarray,
    bm_char: np.ndarray,
    inv_char: np.ndarray,
    market_caps: np.ndarray,
) -> np.ndarray:
    """Construct investment factor from triple sort (size x BM x investment).

    Creates 27 portfolios from low/mid/high breakpoints (30/40/30).
    Investment factor = avg(9 low-inv portfolios) - avg(9 high-inv portfolios).

    Parameters
    ----------
    size_char, bm_char, inv_char: (N,) arrays of characteristics
    market_caps: (N,) array of market capitalizations

    Returns
    -------
    w: (N,) net weight vector
    """
    N = len(size_char)
    breakpoints = (30, 70)  # percentage breakpoints for low/mid/high

    def group_labels(x: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(x, breakpoints)
        labels = np.where(x <= lo, 0, np.where(x >= hi, 2, 1))
        return labels

    s_labels = group_labels(size_char)
    b_labels = group_labels(bm_char)
    i_labels = group_labels(inv_char)

    def vw_w(mask: np.ndarray) -> np.ndarray:
        w = np.zeros(N)
        if mask.sum() > 0:
            w[mask] = market_caps[mask] / market_caps[mask].sum()
        return w

    low_inv_weights = []
    high_inv_weights = []

    for s in range(3):
        for b in range(3):
            mask_low = (s_labels == s) & (b_labels == b) & (i_labels == 0)
            mask_high = (s_labels == s) & (b_labels == b) & (i_labels == 2)
            low_inv_weights.append(vw_w(mask_low))
            high_inv_weights.append(vw_w(mask_high))

    w_low = np.mean(low_inv_weights, axis=0)
    w_high = np.mean(high_inv_weights, axis=0)
    return w_low - w_high


def build_factor_weight_matrix(
    K: int,
    market_caps: pd.Series,
    bm_ratios: Optional[pd.Series],
    accounting: Optional[pd.DataFrame],
) -> np.ndarray:
    """Build K x N factor weight matrix W for a given factor specification.

    Factors:
    K=1: market
    K=3: market, SMB, HML
    K=5: market, SMB, HML, gross profitability, investment
    K=7: market, SMB, HML, gross profitability, investment, asset growth, accruals

    Parameters
    ----------
    K: number of factors (1, 3, 5, or 7)
    market_caps: Series(N) market capitalizations
    bm_ratios: Series(N) book-to-market ratios (required for K>=3)
    accounting: DataFrame(N, cols) with accounting data (required for K>=5)

    Returns
    -------
    W: (K, N) weight matrix where each row is a factor portfolio
    """
    caps = market_caps.values.astype(float)

    # Factor 1: Market (value-weighted)
    w_mkt = caps / caps.sum()

    rows = [w_mkt]

    if K >= 3:
        if bm_ratios is None:
            raise ValueError("bm_ratios required for K>=3")
        w_smb, w_hml = double_sort_smb_hml(market_caps, bm_ratios, None)
        rows.extend([w_smb, w_hml])

    if K >= 5:
        if accounting is None:
            raise ValueError("accounting data required for K>=5")
        # Gross profitability: GP = gross_profit / total_assets
        gp = accounting['gross_profit'].values / np.maximum(accounting['AT'].values, 1e-6)
        w_gp = decile_factor_weights(gp, caps, long_low=True)  # long low GP, short high GP per paper definition

        # Investment factor: triple sort on size x BM x investment
        inv = (accounting['delta_PPEGT'].values + accounting['delta_INVT'].values) / np.maximum(accounting['AT_lag'].values, 1e-6)
        w_inv = investment_factor_weights(caps, bm_ratios.values, inv, caps)
        rows.extend([w_gp, w_inv])

    if K >= 7:
        if accounting is None:
            raise ValueError("accounting data required for K>=7")
        # Asset growth: AT/AT_lag
        ag = accounting['AT'].values / np.maximum(accounting['AT_lag'].values, 1e-6)
        w_ag = decile_factor_weights(ag, caps, long_low=True)  # factor = R_lowAG - R_highAG

        # Accruals
        acc = accounting['accruals'].values
        w_acc = decile_factor_weights(acc, caps, long_low=True)  # factor = R_lowAcc - R_highAcc
        rows.extend([w_ag, w_acc])

    W = np.vstack(rows)  # (K, N)
    return W
