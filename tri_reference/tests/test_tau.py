import numpy as np
from scipy import stats
from tri.tau import kendall_tau_b, systemic_tau, tau_b_matrix
from tri.null import circular_shift_surrogate


def test_tau_b_matches_scipy_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.integers(0, 4, 40); y = rng.integers(0, 4, 40)
        ref = stats.kendalltau(x, y, variant="b").statistic
        assert np.isclose(kendall_tau_b(x, y), ref)


def test_binary_tau_b_equals_phi():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, 200); y = (rng.random(200) < 0.3 + 0.4 * x).astype(int)
    phi = np.corrcoef(x, y)[0, 1]
    assert np.isclose(kendall_tau_b(x, y), phi)


def test_range_and_lower_bound():
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((6, 30))
    ts, d = systemic_tau(Z, return_details=True)
    assert -1 / 5 - 1e-12 <= ts <= 1
    T, _ = tau_b_matrix(Z)
    assert np.all(np.linalg.eigvalsh(T) > -1e-9)   # Gram matrix: PSD


def test_constant_series_convention():
    Z = np.vstack([np.zeros(20), np.arange(20), np.arange(20)])
    ts, d = systemic_tau(Z, return_details=True)
    assert np.isclose(ts, 1 / 3) and np.isclose(d["frac_undefined"], 2 / 3)


def test_surrogate_preserves_marginals():
    rng = np.random.default_rng(3)
    Z = (rng.random((5, 100)) < 0.2).astype(float)
    S = circular_shift_surrogate(Z, rng, 10)
    assert np.allclose(np.sort(S, axis=1), np.sort(Z, axis=1))


def test_binary_fast_path_equals_general_path():
    from tri.tau import sign_vectors
    rng = np.random.default_rng(4)
    Z = (rng.random((6, 80)) < 0.3).astype(float)
    T_fast, _ = tau_b_matrix(Z)
    S = sign_vectors(Z); G = S @ S.T; n = np.sqrt(np.diag(G))
    assert np.allclose(T_fast, G / np.outer(n, n))
