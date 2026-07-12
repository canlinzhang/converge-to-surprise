import numpy as np
from scipy import stats
from scipy.special import rel_entr


class SurpriseScore:
    """
    Measures how surprising it is to observe M matches between two integer
    sequences seq_i and seq_j, under the null hypothesis that both sequences
    are i.i.d. draws from the same distribution.

    Parameters
    ----------
    seq_i : array-like of int
    seq_j : array-like of int
        Both sequences must have the same length N.
        Integers must be in {0, 1, ..., K-1}.
    K : int
        Number of possible integer values (alphabet size).
    """

    def __init__(self, seq_i, seq_j, K: int):
        self.seq_i = np.asarray(seq_i, dtype=int)
        self.seq_j = np.asarray(seq_j, dtype=int)
        self.K = K

        assert len(self.seq_i) == len(self.seq_j), \
            "seq_i and seq_j must have the same length."
        assert self.seq_i.ndim == 1 and self.seq_j.ndim == 1, \
            "Sequences must be 1-D arrays."

        self.N = len(self.seq_i)
        self._compute_stats()

    def _compute_stats(self):
        """Estimate p_k, collision probability q, and number of matches M."""
        # Count occurrences of each k in both sequences combined
        counts = np.zeros(self.K, dtype=float)
        for k in range(self.K):
            counts[k] = np.sum(self.seq_i == k) + np.sum(self.seq_j == k)

        total = counts.sum()  # = 2 * N
        self.p = counts / total  # shape (K,)

        # Collision probability: q = sum_k p_k^2
        self.q = np.sum(self.p ** 2)

        # Number of matches
        self.M = int(np.sum(self.seq_i == self.seq_j))

        # Empirical match rate
        self.q_hat = self.M / self.N

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def z_score(self) -> float:
        """
        Z-score under the normal approximation M ~ Binomial(N, q).

        Z = (M - N*q) / sqrt(N * q * (1 - q))

        Positive Z  → more matches than expected (sequences are similar).
        Negative Z  → fewer matches than expected.

        Returns
        -------
        float
            The Z-score.
        """
        mu = self.N * self.q
        sigma = np.sqrt(self.N * self.q * (1.0 - self.q))
        if sigma == 0:
            return float("inf") if self.M > mu else 0.0
        return (self.M - mu) / sigma

    def kl_surprise(self) -> float:
        """
        Surprise score via large-deviation / KL-divergence approximation.

        P(M = m) ≈ exp(-N * D(q_hat || q))

        where D(a || b) = a*log(a/b) + (1-a)*log((1-a)/(1-b))  (binary KL)

        Surprise = -log P(M = m) ≈ N * D(q_hat || q)

        A larger value means the observed match rate is more surprising.

        Returns
        -------
        float
            The KL-based surprise score (in nats).
        """
        a = self.q_hat
        b = self.q

        # Binary KL divergence D(a || b), with safe handling of edge cases
        kl = 0.0
        if a > 0 and b > 0:
            kl += a * np.log(a / b)
        if (1 - a) > 0 and (1 - b) > 0:
            kl += (1 - a) * np.log((1 - a) / (1 - b))

        #instead, we return the normalized KL divergence per sample
        return kl
    
    def contributing_positions(self, method: str = "kl", threshold: float = 0.005) -> np.ndarray:
        """
        Identify positions n where the matched pair (seq_i[n], seq_j[n]) = (k, k)
        contributes significantly to the surprise score.

        For each dimension k, we test whether the observed match count t_k is
        surprising under the null hypothesis (independence). If dimension k passes
        the threshold, all positions n where seq_i[n] == seq_j[n] == k are marked
        as contributing.

        Parameters
        ----------
        method : str
            "kl"      — per-dimension binary KL divergence (default).
            "z_score" — per-dimension Z-score under normal approximation.
        threshold : float
            For "kl"     : minimum per-dimension KL divergence to be considered contributing.
            For "z_score": minimum Z-score to be considered contributing.

        Returns
        -------
        np.ndarray of shape (N,), dtype int
            Binary mask: 1 if position n is a contributing position, 0 otherwise.
        """
        contributing = np.zeros(self.N, dtype=int)

        # t_k: number of matched pairs at dimension k
        match_mask = (self.seq_i == self.seq_j)                  # (N,) bool
        matched_symbols = self.seq_i[match_mask]                  # symbols at match positions
        t = np.bincount(matched_symbols, minlength=self.K)        # (K,) match counts per dim

        for k in range(self.K):
            t_k = t[k]
            if t_k == 0:
                continue  # dimension never matched — no contributing positions

            # Expected match rate for dimension k under independence: p_k^2
            p_k = self.p[k]
            q_k = p_k ** 2          # expected P(seq_i=k AND seq_j=k)
            q_hat_k = t_k / self.N  # observed rate for this dimension

            if method == "kl":
                # Binary KL: D(q_hat_k || q_k)
                kl_k = 0.0
                if q_hat_k > 0 and q_k > 0:
                    kl_k += q_hat_k * np.log(q_hat_k / q_k)
                r_hat = 1.0 - q_hat_k
                r     = 1.0 - q_k
                if r_hat > 0 and r > 0:
                    kl_k += r_hat * np.log(r_hat / r)
                significant = kl_k >= threshold

            elif method == "z_score":
                # Z-score: how many std devs above expected count?
                mu_k    = self.N * q_k
                sigma_k = np.sqrt(self.N * q_k * (1.0 - q_k))
                if sigma_k == 0:
                    significant = t_k > mu_k
                else:
                    z_k = (t_k - mu_k) / sigma_k
                    significant = z_k >= threshold

            else:
                raise ValueError(f"Unknown method '{method}'. Choose 'kl' or 'z_score'.")

            if significant:
                # Mark all positions where seq_i[n] == seq_j[n] == k
                pos_k = np.where((self.seq_i == k) & (self.seq_j == k))[0]
                contributing[pos_k] = 1

        return contributing

    def valid_dimensions_old(self, tau: int = 1) -> int:
        """
        Count how many symbol values k in {0,...,K-1} have at least tau matched pairs.

        Define t_k = #{n : seq_i[n] = seq_j[n] = k}.
        This returns #{k : t_k >= tau}.

        Notes
        -----
        - tau=1 counts how many dimensions ever appear as a matched pair.
        - tau=0 would return K (every dimension qualifies), so tau should
          typically be >= 1.

        Parameters
        ----------
        tau : int
            Threshold on matched-pair counts per dimension.

        Returns
        -------
        int
            Number of "valid" dimensions with t_k >= tau.
        """
        if tau < 0:
            raise ValueError("tau must be >= 0")

        # Mask of positions that are matches
        match_mask = (self.seq_i == self.seq_j)
        matched_symbols = self.seq_i[match_mask]  # equals seq_j[match_mask]

        # Count t_k efficiently for all k
        t = np.bincount(matched_symbols, minlength=self.K)

        return int(np.sum(t >= tau))

    def valid_dimensions(self, method: str = "kl", threshold: float = 0.005) -> int:
        """
        Count how many symbol values k in {0,...,K-1} are "valid" — i.e., their
        observed match count t_k is statistically surprising under the null
        hypothesis of independence.

        For each dimension k, we test whether the observed match rate q_hat_k = t_k / N
        is surprising relative to the expected rate q_k = p_k^2. A dimension is valid
        if it passes the significance threshold.

        Parameters
        ----------
        method : str
            "kl"      — per-dimension binary KL divergence (default).
            "z_score" — per-dimension Z-score under normal approximation.
        threshold : float
            For "kl"     : minimum per-dimension KL divergence to be considered valid.
            For "z_score": minimum Z-score to be considered valid.

        Returns
        -------
        int
            Number of "valid" dimensions that pass the significance threshold.
        """
        match_mask = (self.seq_i == self.seq_j)
        matched_symbols = self.seq_i[match_mask]
        t = np.bincount(matched_symbols, minlength=self.K)

        valid_count = 0

        for k in range(self.K):
            t_k = t[k]
            if t_k == 0:
                continue

            p_k = self.p[k]
            q_k = p_k ** 2
            q_hat_k = t_k / self.N

            if q_hat_k <= q_k:
                continue  # under-matching — not a valid dimension

            if method == "kl":
                kl_k = 0.0
                if q_hat_k > 0 and q_k > 0:
                    kl_k += q_hat_k * np.log(q_hat_k / q_k)
                r_hat = 1.0 - q_hat_k
                r     = 1.0 - q_k
                if r_hat > 0 and r > 0:
                    kl_k += r_hat * np.log(r_hat / r)
                significant = kl_k >= threshold

            elif method == "z_score":
                mu_k    = self.N * q_k
                sigma_k = np.sqrt(self.N * q_k * (1.0 - q_k))
                if sigma_k == 0:
                    significant = t_k > mu_k
                else:
                    z_k = (t_k - mu_k) / sigma_k
                    significant = z_k >= threshold

            else:
                raise ValueError(f"Unknown method '{method}'. Choose 'kl' or 'z_score'.")

            if significant:
                valid_count += 1

        return valid_count

    def view_agreement_rate(self) -> float:
        """
        Returns the fraction of positions where seq_i and seq_j agree.
        M / N where M = #{n : seq_i[n] == seq_j[n]}
        """
        return self.M / self.N

    def per_dim_kl_surprise(self) -> float:
        match_mask = (self.seq_i == self.seq_j)
        matched_symbols = self.seq_i[match_mask]
        t = np.bincount(matched_symbols, minlength=self.K)

        info_holder = list()
        for k in range(self.K):
            t_k = t[k]
            if t_k == 0:
                continue

            p_k = self.p[k]
            q_k = p_k ** 2
            q_hat_k = t_k / self.N

            if q_hat_k <= q_k:
                continue  # under-matching — not a valid dimension

            kl_k = 0.0
            if q_hat_k > 0 and q_k > 0:
                kl_k += q_hat_k * np.log(q_hat_k / q_k)
            r_hat = 1.0 - q_hat_k
            r     = 1.0 - q_k
            if r_hat > 0 and r > 0:
                kl_k += r_hat * np.log(r_hat / r)

            #this is the old format with log added
            #info_k = self.N * kl_k
            #log_info_k = np.log(info_k + 1.)
            #info_holder.append(log_info_k)

            #just linear
            info_holder.append(kl_k)

        return np.sum(info_holder)

    def score(self, method: str = "z_score") -> float:
        """
        Compute the surprise score.

        Parameters
        ----------
        method : str
            "z_score"    — Z-score under normal approximation (default).
            "kl"         — KL-divergence large-deviation surprise (nats).

        Returns
        -------
        float
            The requested surprise score.
        """
        if method == "z_score":
            return self.z_score()
        elif method == "kl":
            return self.kl_surprise()
        elif method == "valid_dims":
            return float(self.valid_dimensions())
        elif method == "per_dim_kl":
            return self.per_dim_kl_surprise()
        elif method == "view_agreement":
            return self.view_agreement_rate()
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'z_score', 'kl', 'valid_dims', 'per_dim_kl', or 'view_agreement'.")

    def summary(self) -> dict:
        """Return a dict with all key statistics."""
        return {
            "N": self.N,
            "K": self.K,
            "M": self.M,
            "q_expected": self.q,
            "q_hat": self.q_hat,
            "z_score": self.z_score(),
            "kl_surprise": self.kl_surprise(),
            "valid_dimensions": self.valid_dimensions(),
            "p_value_upper_tail": float(stats.norm.sf(self.z_score())),
        }



