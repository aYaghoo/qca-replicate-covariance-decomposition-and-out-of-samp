import numpy as np
import pandas as pd


class SimulatedMarketData:
    """Generate synthetic market data for testing the covariance forecasting pipeline.

    Simulates:
    - N stocks with K factors whose loadings are tied to firm characteristics
    - S sectors with block residual structure
    - Latent factor and residual variances following a HAR process in logs
    - Daily REALIZED covariance matrices built from M intraday returns
      (so the target contains measurement error, as real RCov does)
    - Synthetic accounting data for factor construction

    Units: returns are in percent, so daily variances are O(1).

    Factor order matches Alves et al. (2023): market, SMB, HML, gross
    profitability, investment, asset growth, accruals.
    """

    def __init__(
        self,
        N: int = 50,
        K: int = 7,
        S: int = 10,
        T: int = 1495,
        seed: int = 42,
        M: int = 78,  # intraday returns per day (78 = 5-min)
        alignment: float = 0.8,  # corr between characteristic and loading, in [0, 1]
        burn_in: int = 250,  # days simulated and discarded before t = 0
        har_coefs: tuple[float, float, float] = (0.35, 0.25, 0.35),  # daily, weekly, monthly
        vol_of_vol: float = 0.20,  # sd of daily shock to log variance
        market_vol: float = 1.0,  # median daily market-factor vol (%)
        style_vol: float = 0.5,  # median daily vol of the long-short factors (%)
        resid_vol: float = 1.4,  # median daily idiosyncratic vol (%)
        style_loading_sd: float = 0.35,  # cross-sectional SD of style-factor loadings
    ):
        if M <= N:
            raise ValueError(
                f"M={M} intraday returns cannot give a positive-definite {N}x{N} "
                "realized covariance; use M > N."
            )
        if sum(har_coefs) >= 1:
            raise ValueError("HAR coefficients must sum to < 1 for stationarity.")

        self.N = N
        self.K = K
        self.S = S
        self.T = T
        self.M = M
        self.alignment = alignment
        self.burn_in = burn_in
        self.har_coefs = har_coefs
        self.vol_of_vol = vol_of_vol
        self.market_vol = market_vol
        self.style_vol = style_vol
        self.resid_vol = resid_vol
        self.style_loading_sd = style_loading_sd
        self.rng = np.random.default_rng(seed)

        # Sector assignment: distribute N stocks across S sectors
        self.sector_sizes = self._assign_sectors()
        self.sector_indices = self._build_sector_indices()

        # Characteristics are drawn once and cached, because the true loadings
        # depend on them. The generate_* methods return these same draws.
        self._market_caps = self._draw_market_caps()
        self._bm_ratios = self._draw_bm_ratios()
        self._accounting = self._draw_accounting_data()

        self.B_true: np.ndarray | None = None
        self.Sigma_true_list: list[np.ndarray] | None = None

    def _assign_sectors(self) -> list[int]:
        """Assign stocks to 10 sectors with realistic sizes."""
        # Approximate the paper's sector distribution scaled to N
        paper_sizes = [31, 8, 65, 32, 61, 10, 45, 26, 36, 116]  # paper's 430 stock counts
        total = sum(paper_sizes)
        scaled = [max(1, round(s / total * self.N)) for s in paper_sizes]
        # Adjust to sum to N
        diff = self.N - sum(scaled)
        scaled[-1] += diff
        return scaled

    def _build_sector_indices(self) -> list[np.ndarray]:
        """Build index arrays for each sector."""
        indices = []
        start = 0
        for size in self.sector_sizes:
            indices.append(np.arange(start, start + size))
            start += size
        return indices

    # ------------------------------------------------------------------
    # Loadings
    # ------------------------------------------------------------------
    @staticmethod
    def _rank_score(x) -> np.ndarray:
        """Standardized cross-sectional ranks (robust to skewed ratios)."""
        r = np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)
        return (r - r.mean()) / r.std()

    def _characteristic_signals(self) -> list[np.ndarray]:
        """Signals for the six non-market factors, in paper order."""
        acct = self._accounting
        return [
            self._rank_score(np.log(self._market_caps.values)),  # SMB
            self._rank_score(self._bm_ratios.values),  # HML
            self._rank_score(acct["gross_profit"] / acct["AT"]),  # GP
            self._rank_score(
                (acct["delta_PPEGT"] + acct["delta_INVT"]) / acct["AT_lag"]
            ),  # Investment
            self._rank_score(acct["AT"] / acct["AT_lag"]),  # Asset growth
            self._rank_score(acct["accruals"]),  # Accruals
        ]

    def generate_factor_loadings(self) -> np.ndarray:
        """Generate fixed true factor loadings B: (K, N).

        Row 0 is market beta. Rows 1..6 load on the matching characteristic
        with correlation `alignment`; the sign is immaterial for spanning.
        Style rows are scaled to cross-sectional SD `style_loading_sd` so the
        market factor dominates systematic variance.
        """
        K, N, a, sd = self.K, self.N, self.alignment, self.style_loading_sd
        B = np.zeros((K, N))
        B[0] = np.clip(1.0 + 0.3 * self.rng.standard_normal(N), 0.3, None)

        signals = self._characteristic_signals()
        for k in range(1, K):
            noise = self.rng.standard_normal(N)
            if k - 1 < len(signals):
                B[k] = sd * (a * signals[k - 1] + np.sqrt(1 - a**2) * noise)
            else:
                B[k] = sd * noise  # factors beyond the paper's seven are unaligned
        return B

    # ------------------------------------------------------------------
    # Covariance dynamics
    # ------------------------------------------------------------------
    def _simulate_log_har(self, n: int, median_var: float) -> np.ndarray:
        """HAR(1,5,22) in log variance, burn-in included. Returns variances (T+burn_in, n)."""
        bd, bw, bm = self.har_coefs
        T_total = self.T + self.burn_in
        mu = np.log(median_var)
        x = np.full((T_total, n), mu)
        for t in range(22, T_total):
            dev_d = x[t - 1] - mu
            dev_w = x[t - 5 : t].mean(axis=0) - mu
            dev_m = x[t - 22 : t].mean(axis=0) - mu
            x[t] = (
                mu
                + bd * dev_d
                + bw * dev_w
                + bm * dev_m
                + self.vol_of_vol * self.rng.standard_normal(n)
            )
        return np.exp(x)

    def _simulate_factor_covariances(self) -> list[np.ndarray]:
        """Simulate K x K latent factor covariances (post burn-in)."""
        K = self.K
        h = np.empty((self.T + self.burn_in, K))
        h[:, :1] = self._simulate_log_har(1, self.market_vol**2)
        if K > 1:
            h[:, 1:] = self._simulate_log_har(K - 1, self.style_vol**2)
        h = h[self.burn_in :]

        corr_f = np.eye(K) + 0.1 * (np.ones((K, K)) - np.eye(K))
        Sigma_f_list = []
        for t in range(self.T):
            D = np.diag(np.sqrt(h[t]))
            Sigma_f_list.append(D @ corr_f @ D)
        return Sigma_f_list

    def _simulate_residual_covariances(self) -> list[np.ndarray]:
        """Simulate N x N block-diagonal latent residual covariances (post burn-in)."""
        N = self.N
        res_var = self._simulate_log_har(N, self.resid_vol**2)[self.burn_in :]

        Sigma_e_list = []
        for t in range(self.T):
            Sigma_e_t = np.zeros((N, N))
            for sector_idx in self.sector_indices:
                ns = len(sector_idx)
                corr_s = 0.3 * (np.ones((ns, ns)) - np.eye(ns)) + np.eye(ns)
                D_s = np.diag(np.sqrt(res_var[t, sector_idx]))
                Sigma_e_t[np.ix_(sector_idx, sector_idx)] = D_s @ corr_s @ D_s
            Sigma_e_list.append(Sigma_e_t)
        return Sigma_e_list

    def simulate_realized_covariances(
        self,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Simulate T daily realized covariance matrices.

        Latent: Sigma_t = B.T @ Sigma_f_t @ B + Sigma_e_t.
        Observed: RCov_t = sum_j r_{t,j} r_{t,j}' over M intraday returns
        r_{t,j} = B' f_{t,j} + e_{t,j}, f ~ N(0, Sigma_f_t/M), e ~ N(0, Sigma_e_t/M).
        The latent matrices are stored in self.Sigma_true_list.

        Returns:
        -------
        Sigma_list: list of T (N, N) realized covariance matrices
        factor_returns: (T, K) daily factor returns (sum of intraday)
        stock_returns: (T, N) daily stock returns (sum of intraday)
        """
        N, K, T, M = self.N, self.K, self.T, self.M
        B_true = self.generate_factor_loadings()  # (K, N)
        self.B_true = B_true

        Sigma_f_list = self._simulate_factor_covariances()
        Sigma_e_list = self._simulate_residual_covariances()

        factor_returns = np.zeros((T, K))
        stock_returns = np.zeros((T, N))
        Sigma_list, Sigma_true_list = [], []

        for t in range(T):
            Sigma_f_t, Sigma_e_t = Sigma_f_list[t], Sigma_e_list[t]
            Sigma_true_list.append(B_true.T @ Sigma_f_t @ B_true + Sigma_e_t)

            F = np.linalg.cholesky(Sigma_f_t) @ self.rng.standard_normal((K, M)) / np.sqrt(M)
            E = np.linalg.cholesky(Sigma_e_t) @ self.rng.standard_normal((N, M)) / np.sqrt(M)
            R = B_true.T @ F + E  # (N, M) intraday returns

            RCov = R @ R.T
            Sigma_list.append((RCov + RCov.T) / 2)
            factor_returns[t] = F.sum(axis=1)
            stock_returns[t] = R.sum(axis=1)

        self.Sigma_true_list = Sigma_true_list
        return Sigma_list, factor_returns, stock_returns

    # ------------------------------------------------------------------
    # Characteristics (drawn once in __init__, returned by generate_*)
    # ------------------------------------------------------------------
    def _draw_accounting_data(self) -> pd.DataFrame:
        """Draw synthetic annual accounting data for all N stocks."""
        N = self.N
        acct = pd.DataFrame(
            {
                "AT": self.rng.uniform(1e8, 1e10, N),
                "AT_lag": self.rng.uniform(1e8, 1e10, N),
                "ACT": self.rng.uniform(1e7, 5e9, N),
                "CHE": self.rng.uniform(1e6, 1e9, N),
                "LCT": self.rng.uniform(1e7, 3e9, N),
                "DLC": self.rng.uniform(0, 5e8, N),
                "TXP": self.rng.uniform(0, 3e8, N),
                "DP": self.rng.uniform(0, 2e8, N),
                "PPEGT": self.rng.uniform(1e7, 5e9, N),
                "PPEGT_lag": self.rng.uniform(1e7, 5e9, N),
                "INVT": self.rng.uniform(1e6, 2e9, N),
                "INVT_lag": self.rng.uniform(1e6, 2e9, N),
            }
        )

        # Derived fields
        acct["gross_profit"] = acct["AT"] * self.rng.uniform(0.1, 0.4, N)
        acct["delta_PPEGT"] = acct["PPEGT"] - acct["PPEGT_lag"]
        acct["delta_INVT"] = acct["INVT"] - acct["INVT_lag"]

        # Accruals
        delta_ACT = acct["ACT"] * 0.05
        delta_CHE = acct["CHE"] * 0.02
        delta_LCT = acct["LCT"] * 0.03
        delta_DLC = acct["DLC"] * 0.01
        delta_TXP = acct["TXP"] * 0.02
        avg_AT = (acct["AT"] + acct["AT_lag"]) / 2
        acct["accruals"] = (
            delta_ACT - delta_CHE - delta_LCT + delta_DLC + delta_TXP - acct["DP"]
        ) / np.maximum(avg_AT, 1e-6)

        return acct

    def _draw_market_caps(self) -> pd.Series:
        return pd.Series(self.rng.uniform(1e8, 1e11, self.N), name="market_cap")

    def _draw_bm_ratios(self) -> pd.Series:
        return pd.Series(self.rng.uniform(0.1, 5.0, self.N), name="bm_ratio")

    def generate_accounting_data(self) -> pd.DataFrame:
        """Return the cached synthetic accounting data."""
        return self._accounting.copy()

    def generate_market_caps(self) -> pd.Series:
        """Return the cached synthetic market capitalizations."""
        return self._market_caps.copy()

    def generate_bm_ratios(self) -> pd.Series:
        """Return the cached synthetic book-to-market ratios."""
        return self._bm_ratios.copy()
