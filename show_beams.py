import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# reimplemented minimal helpers from hi.py for standalone beam visualization

def wavelength(f_hz, c=1500.0):
    return c / f_hz


def steering_vector_ula(N, theta_deg, f_hz, spacing_lambda=0.5, c=1500.0):
    lam = wavelength(f_hz, c)
    d = spacing_lambda * lam
    theta = np.deg2rad(theta_deg)
    n = np.arange(N)
    phase = -2j * np.pi * (n * d) * np.sin(theta) / lam
    return np.exp(phase)


def get_taylor_taper(N, nbar=4, sll_db=30.0):
    if N <= 1:
        return np.ones(N, dtype=float)
    nbar_clamped = max(1, min(nbar, (N - 1) // 2))
    try:
        from scipy.signal import windows
        tap = windows.taylor(N, nbar=nbar_clamped, sll=-abs(sll_db), norm=False)
        tap = np.abs(tap)
        if not np.all(np.isfinite(tap)) or np.max(np.abs(tap)) == 0:
            raise RuntimeError
    except Exception:
        if N == 1:
            tap = np.array([1.0])
        else:
            n = np.arange(N)
            tap = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
            tap = tap ** 0.7
            if np.max(np.abs(tap)) == 0:
                tap = np.ones(N)
    tap = tap / np.max(np.abs(tap))
    return tap


def lcmv_weights(R, C, desired=None, diag_load=1e-3):
    N = R.shape[0]
    M = C.shape[1]
    R = R.astype(complex)
    R = R + diag_load * np.trace(R) / N * np.eye(N, dtype=complex)
    if desired is None:
        f = np.ones((M, 1), dtype=complex)
    else:
        f = np.atleast_1d(desired).astype(complex).reshape(M, 1)
    RinvC = np.linalg.solve(R, C)
    CHRinvC = C.conj().T @ RinvC
    CHRinvC = CHRinvC + 1e-12 * np.eye(M)
    gamma = np.linalg.solve(CHRinvC, f)
    w = (RinvC @ gamma).ravel()
    return w


def beampattern_for_weights(w, N, f_hz, spacing_lambda=0.5, c=1500.0, angles_deg=None):
    if angles_deg is None:
        angles_deg = np.linspace(-90, 90, 721)
    A = np.column_stack([steering_vector_ula(N, th, f_hz, spacing_lambda, c) for th in angles_deg])
    resp = np.conjugate(w) @ A
    resp_db = 20 * np.log10(np.abs(resp) / np.max(np.abs(resp)) + 1e-12)
    return angles_deg, resp_db


def compute_broadband_lcmv_per_target(angles_deg, freqs, N=5, spacing_lambda=0.5, c=1500.0,
                                     nbar=4, sll_db=30.0, R=None, diag_load=1e-3):
    angles = np.atleast_1d(angles_deg)
    M = angles.size
    tap = get_taylor_taper(N, nbar=nbar, sll_db=sll_db)
    if R is None:
        R = np.eye(N, dtype=complex)
    Wt = np.zeros((M, len(freqs), N), dtype=complex)
    for i, f_hz in enumerate(freqs):
        C = np.column_stack([steering_vector_ula(N, th, f_hz, spacing_lambda, c) for th in angles])
        C_t = tap[:, None] * C
        for j in range(M):
            desired = np.zeros((M, 1), dtype=complex)
            desired[j, 0] = 1.0
            w = lcmv_weights(R, C_t, desired=desired, diag_load=diag_load)
            Wt[j, i, :] = w
    return freqs, Wt, tap


def plot_each_beam_average(freqs, Wt, desired_angles, N, spacing_lambda=0.5, c=1500.0,
                           angles_scan=None, title_prefix=None, save_plots=False):
    if angles_scan is None:
        angles_scan = np.linspace(-90, 90, 721)
    M = Wt.shape[0]
    for j in range(M):
        P_avg = np.zeros(len(angles_scan))
        for i, f_hz in enumerate(freqs):
            _, resp_db = beampattern_for_weights(Wt[j, i, :], N, f_hz, spacing_lambda, c, angles_scan)
            P_avg += 10 ** (resp_db / 10.0)
        P_avg = P_avg / len(freqs)
        P_avg_db = 10 * np.log10(P_avg + 1e-12)
        P_avg_db = P_avg_db - np.max(P_avg_db)
        plt.figure(figsize=(8, 4))
        plt.plot(angles_scan, P_avg_db)
        plt.ylim([-60, 0])
        plt.grid(True)
        plt.xlabel('Angle (deg)')
        plt.ylabel('Average Response (dB, normalized)')
        title = f"Beam {j}: target {desired_angles[j]} deg"
        if title_prefix:
            title = title_prefix + " — " + title
        plt.title(title)
        if save_plots:
            fname = f"beam_{j}_{int(np.round(desired_angles[j]))}deg.png"
            plt.savefig(fname, bbox_inches='tight')
            print('Saved', fname)
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    N = 5
    spacing_lambda = 0.5
    c = 1500.0
    desired_angles = [-30.0, 20.0]
    f_min, f_max = 10.0, 3000.0
    n_freqs = 128
    freqs = np.linspace(f_min, f_max, n_freqs)
    freqs_t, Wt, taper = compute_broadband_lcmv_per_target(desired_angles, freqs, N=N,
                                                             spacing_lambda=spacing_lambda, c=c,
                                                             nbar=4, sll_db=30.0, R=None,
                                                             diag_load=1e-3)
    plot_each_beam_average(freqs_t, Wt, desired_angles, N, spacing_lambda=spacing_lambda, c=c,
                           angles_scan=np.linspace(-90,90,721), title_prefix='Per-target beams', save_plots=True)
