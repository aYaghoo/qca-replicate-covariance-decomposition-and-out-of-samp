import numpy as np


def clean_covariance_matrices(
    Sigma_list: list[np.ndarray],
    sigma_threshold: float = 4.0,
    flag_fraction: float = 0.25,
    replacement_window: int = 10,
) -> tuple[list[np.ndarray], list[bool]]:
    """Clean realized covariance matrices by flagging extreme entries.

    For each day t:
    1. Compute historical mean and std of each unique element up to t.
    2. Flag entry if |x - mean| > sigma_threshold * std.
    3. Flag day if fraction of flagged entries > flag_fraction.
    4. Replace flagged matrix with average of nearest replacement_window preceding non-flagged matrices.

    Parameters
    ----------
    Sigma_list: list of N x N symmetric PSD matrices
    sigma_threshold: flag if deviation > sigma_threshold * historical_std
    flag_fraction: flag matrix if more than this fraction of unique entries are extreme
    replacement_window: number of preceding non-flagged matrices to average for replacement

    Returns
    -------
    cleaned_list: list of cleaned covariance matrices
    is_flagged: boolean list indicating which days were flagged
    """
    T = len(Sigma_list)
    N = Sigma_list[0].shape[0]
    idx = np.tril_indices(N)
    M = len(idx[0])  # number of unique elements

    # Stack all vech vectors
    vech_arr = np.array([Sigma_list[t][idx] for t in range(T)])  # T x M

    cleaned_list = list(Sigma_list)
    is_flagged = [False] * T
    non_flagged_indices = []

    for t in range(T):
        if t == 0:
            non_flagged_indices.append(t)
            continue

        # Historical mean and std up to t (exclusive)
        hist = vech_arr[:t, :]  # t x M
        hist_mean = hist.mean(axis=0)
        hist_std = hist.std(axis=0)

        # Current entries
        curr = vech_arr[t, :]

        # Flag entries beyond threshold (use std > 0 to avoid division by zero)
        valid_std = hist_std > 1e-10
        n_flagged = np.sum(valid_std & (np.abs(curr - hist_mean) > sigma_threshold * hist_std))
        frac_flagged = n_flagged / M

        if frac_flagged > flag_fraction:
            is_flagged[t] = True
            # Replace with average of nearest replacement_window preceding non-flagged
            preceding = [i for i in non_flagged_indices if i < t]
            if len(preceding) == 0:
                # No preceding non-flagged, keep original
                pass
            else:
                use = preceding[-replacement_window:]
                avg = np.mean([cleaned_list[i] for i in use], axis=0)
                avg = (avg + avg.T) / 2  # enforce symmetry
                cleaned_list[t] = avg
        else:
            non_flagged_indices.append(t)

    return cleaned_list, is_flagged
