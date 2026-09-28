import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional


class SimulatedMarketData:
    """Generate synthetic market data for testing the covariance forecasting pipeline.

    Simulates:
    - N stocks with K latent factors
    - S sectors with block residual structure
    - Realistic time-varying factor covariances using GARCH-like dynamics
    - Daily realized covariance matrices
    - Synthetic accounting data for factor construction
    """

    def __init__(
        self,
        N: int = 50,
        K: int = 7,
        S: int = 10,
        T: int = 1495,
        seed: int = 42,
    ):
        self.N = N
        self.K = K
        self.S = S
        self.T = T
        self.rng = np.random.default_rng(seed)

        # Sector assignment: distribute N stocks across S sectors
        self.sector_sizes = self._assign_sectors()
        self.sector_indices = self._build_sector_indices()

    def _assign_sectors(self) -> List[int]:
        """Assign stocks to 10 sectors with realistic sizes."""
        # Approximate the paper's sector distribution scaled to N
        paper_sizes = [31, 8, 65, 32, 61, 10, 45, 26, 36, 116]  # paper's 430 stock counts
        total = sum(paper_sizes)
        scaled = [max(1, round(s / total * self.N)) for s in paper_sizes]
        # Adjust to sum to N
        diff = self.N - sum(scaled)
        scaled[-1] += diff
        return scaled

    def _build_sector_indices(self) -> List[np.ndarray]:
        """Build index arrays for each sector."""
        indices = []
        start = 0
        for size in self.sector_sizes:
            indices.append(np.arange(start, start + size))
            start += size
        return indices

    def generate_factor_loadings(self) -> np.ndarray:
        """Generate fixed true factor loadings B: (K, N)."""
        B = self.rng.normal(0, 0.3, (self.K, self.N))
        # Market factor: all positive
        B[0, :] = np.abs(B[0, :]) + 0.5
        return B

    def simulate_realized_covariances(
        self,
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """Simulate T daily realized covariance matrices.

        Uses a factor model: Sigma_t = B.T @ Sigma_f_t @ B + Sigma_e_t
        where Sigma_f_t has GARCH-like dynamics and Sigma_e_t is block-diagonal.

        Returns
        -------
        Sigma_list: list of T (N, N) realized covariance matrices
        factor_returns: (T, K) factor return matrix
        stock_returns: (T, N) stock return matrix
        """
        N, K, T, S = self.N, self.K, self.T, self.S
        B_true = self.generate_factor_loadings()  # (K, N)

        # Simulate factor covariances with persistence
        Sigma_f_list = self._simulate_factor_covariances()

        # Simulate residual variances per sector (block diagonal)
        Sigma_e_list = self._simulate_residual_covariances()

        # Simulate daily returns
        factor_returns = np.zeros((T, K))
        stock_returns = np.zeros((T, N))
        Sigma_list = []

        for t in range(T):
            Sigma_f_t = Sigma_f_list[t]
            Sigma_e_t = Sigma_e_list[t]

            # Realized covariance
            Sigma_t = B_true.T @ Sigma_f_t @ B_true + Sigma_e_t
            Sigma_t = (Sigma_t + Sigma_t.T) / 2
            # Add small ridge for PSD
            Sigma_t += 1e-6 * np.eye(N)
            Sigma_list.append(Sigma_t)

            # Generate returns
            chol_f = np.linalg.cholesky(Sigma_f_t + 1e-8 * np.eye(K))
            f_t = chol_f @ self.rng.standard_normal(K)
            factor_returns[t, :] = f_t

            chol_e = np.linalg.cholesky(Sigma_e_t + 1e-8 * np.eye(N))
            e_t = chol_e @ self.rng.standard_normal(N)
            stock_returns[t, :] = B_true.T @ f_t + e_t

        return Sigma_list, factor_returns, stock_returns

    def _simulate_factor_covariances(self) -> List[np.ndarray]:
        """Simulate K x K factor covariance matrices with HAR-like persistence."""
        T, K = self.T, self.K
        # Base volatilities with persistence
        vol = np.ones((T, K)) * 0.02
        for t in range(1, T):
            shock = self.rng.standard_normal(K) * 0.005
            vol[t] = 0.95 * vol[t - 1] + 0.05 * np.abs(shock) + 0.002

        Sigma_f_list = []
        # Base correlation matrix for factors
        corr_f = np.eye(K) + 0.1 * (np.ones((K, K)) - np.eye(K))
        for t in range(T):
            D = np.diag(vol[t])
            Sigma_f_t = D @ corr_f @ D
            Sigma_f_t = (Sigma_f_t + Sigma_f_t.T) / 2
            Sigma_f_list.append(Sigma_f_t)
        return Sigma_f_list

    def _simulate_residual_covariances(self) -> List[np.ndarray]:
        """Simulate N x N block-diagonal residual covariance matrices."""
        T, N = self.T, self.N
        # Residual variances per stock
        res_var = np.ones((T, N)) * 0.01
        for t in range(1, T):
            shock = self.rng.standard_normal(N) * 0.002
            res_var[t] = 0.92 * res_var[t - 1] + 0.05 * np.abs(shock) + 0.001

        Sigma_e_list = []
        for t in range(T):
            Sigma_e_t = np.zeros((N, N))
            for sector_idx in self.sector_indices:
                ns = len(sector_idx)
                # Within-sector correlation
                corr_s = 0.3 * (np.ones((ns, ns)) - np.eye(ns)) + np.eye(ns)
                D_s = np.diag(np.sqrt(res_var[t, sector_idx]))
                block = D_s @ corr_s @ D_s
                block = (block + block.T) / 2
                Sigma_e_t[np.ix_(sector_idx, sector_idx)] = block
            Sigma_e_list.append(Sigma_e_t)
        return Sigma_e_list

    def generate_accounting_data(self) -> pd.DataFrame:
        """Generate synthetic annual accounting data for all N stocks."""
        N = self.N
        acct = pd.DataFrame({
            'AT': self.rng.uniform(1e8, 1e10, N),
            'AT_lag': self.rng.uniform(1e8, 1e10, N),
            'ACT': self.rng.uniform(1e7, 5e9, N),
            'CHE': self.rng.uniform(1e6, 1e9, N),
            'LCT': self.rng.uniform(1e7, 3e9, N),
            'DLC': self.rng.uniform(0, 5e8, N),
            'TXP': self.rng.uniform(0, 3e8, N),
            'DP': self.rng.uniform(0, 2e8, N),
            'PPEGT': self.rng.uniform(1e7, 5e9, N),
            'PPEGT_lag': self.rng.uniform(1e7, 5e9, N),
            'INVT': self.rng.uniform(1e6, 2e9, N),
            'INVT_lag': self.rng.uniform(1e6, 2e9, N),
        })

        # Derived fields
        acct['gross_profit'] = acct['AT'] * self.rng.uniform(0.1, 0.4, N)
        acct['delta_PPEGT'] = acct['PPEGT'] - acct['PPEGT_lag']
        acct['delta_INVT'] = acct['INVT'] - acct['INVT_lag']

        # Accruals
        delta_ACT = acct['ACT'] * 0.05
        delta_CHE = acct['CHE'] * 0.02
        delta_LCT = acct['LCT'] * 0.03
        delta_DLC = acct['DLC'] * 0.01
        delta_TXP = acct['TXP'] * 0.02
        avg_AT = (acct['AT'] + acct['AT_lag']) / 2
        acct['accruals'] = (delta_ACT - delta_CHE - delta_LCT + delta_DLC + delta_TXP - acct['DP']) / np.maximum(avg_AT, 1e-6)

        return acct

    def generate_market_caps(self) -> pd.Series:
        """Generate synthetic market capitalizations."""
        caps = self.rng.uniform(1e8, 1e11, self.N)
        return pd.Series(caps, name='market_cap')

    def generate_bm_ratios(self) -> pd.Series:
        """Generate synthetic book-to-market ratios."""
        bm = self.rng.uniform(0.1, 5.0, self.N)
        return pd.Series(bm, name='bm_ratio')
