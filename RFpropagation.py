import numpy as np

# Optional plotting — fall back gracefully if matplotlib isn't available
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


def wavelength(f_hz, c=1500.0):
    """Return wavelength (m) for frequency f_hz and speed c (m/s)."""
    return c / f_hz


def steering_vector_ula(N, theta_deg, f_hz, spacing_lambda=0.5, c=1500.0):
    """
    Narrowband ULA steering vector (elements along x, origin at element 0).
    N: number of sensors
    theta_deg: angle in degrees (broadside = 0, +ve to endfire)
    f_hz: frequency (Hz)
    spacing_lambda: spacing in wavelengths (default 0.5)
    c: propagation speed (m/s)
    Returns: (N,) complex steering vector a(theta)
    """
    lam = wavelength(f_hz, c)
    d = spacing_lambda * lam
    theta = np.deg2rad(theta_deg)
    n = np.arange(N)
    phase = -2j * np.pi * (n * d) * np.sin(theta) / lam
    return np.exp(phase)


def get_taylor_taper(N, nbar=4, sll_db=30.0):
    """
    Return a Taylor amplitude taper (real, positive) of length N.
    Tries scipy.signal.windows.taylor; falls back to a smooth cosine taper if unavailable
    or the SciPy routine returns invalid values (NaN) — this can happen for very small
    N or incompatible nbar/sll combinations.
    sll_db is the design sidelobe level in dB (positive number, e.g. 30).
    The returned taper is normalized so its maximum is 1.
    """
    # Guard for trivial lengths
    if N <= 1:
        return np.ones(N, dtype=float)

    # Clamp nbar to a sensible maximum for small N to avoid SciPy internal errors.
    # SciPy's Taylor implementation can produce invalid values for nbar >= (N-1)/2
    nbar_clamped = max(1, min(nbar, (N - 1) // 2))

    try:
        from scipy.signal import windows
        # Try to compute Taylor window; SciPy expects sll negative in some versions
        tried = False
        for sll_try in (-abs(sll_db), abs(sll_db)):
            try:
                tap = windows.taylor(N, nbar=nbar_clamped, sll=sll_try, norm=False)
                tap = np.abs(tap)
                tried = True
                break
            except Exception:
                # try the other sign or continue to fallback
                continue
        if not tried:
            raise RuntimeError('scipy.taylor returned error')

        # Validate output (no NaNs and non-zero max)
        if not np.all(np.isfinite(tap)) or np.max(np.abs(tap)) == 0:
            raise RuntimeError('scipy.taylor produced invalid output')

    except Exception:
        # SciPy unavailable or produced invalid result; use a robust fallback.
        # Use a Hann-like envelope raised slightly to better approximate Taylor for small N.
        if N == 1:
            tap = np.array([1.0])
        else:
            n = np.arange(N)
            tap = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
            # gently shape to approximate Taylor mainlobe emphasis
            tap = tap ** 0.7
            # If the Hann-like taper is too small (e.g., almost zeros), ensure numeric safety
            if np.max(np.abs(tap)) == 0:
                tap = np.ones(N)
        print(f"Warning: Taylor taper unavailable or invalid for N={N}, nbar={nbar}. Using fallback taper.")

    # normalize max to 1
    tap = tap / np.max(np.abs(tap))
    return tap


def lcmv_weights(R, C, desired=None, diag_load=1e-3):
    """
    Standard narrowband LCMV solver given covariance R and constraint matrix C (N x M).
    desired: (M,) or (M,1) desired complex responses (defaults to ones).
    Returns weight vector w (N,)
    """
    N = R.shape[0]
    M = C.shape[1]
    R = R.astype(complex)
    # diagonal loading proportional to average power
    R = R + diag_load * np.trace(R) / N * np.eye(N, dtype=complex)

    if desired is None:
        f = np.ones((M, 1), dtype=complex)
    else:
        f = np.atleast_1d(desired).astype(complex).reshape(M, 1)

    # Solve using linear solves for numerical stability
    RinvC = np.linalg.solve(R, C)            # N x M
    CHRinvC = C.conj().T @ RinvC             # M x M
    eps = 1e-12
    CHRinvC = CHRinvC + eps * np.eye(M)
    gamma = np.linalg.solve(CHRinvC, f)      # M x 1
    w = (RinvC @ gamma).ravel()              # N,
    return w


def compute_broadband_lcmv(angles_deg, freqs, N=5, spacing_lambda=0.5, c=1500.0,
                           nbar=4, sll_db=30.0, R=None, diag_load=1e-3):
    """
    Compute frequency-dependent LCMV weights for a broadband set of frequencies.
    - angles_deg: list of desired mainlobe directions (degrees)
    - freqs: 1D array of frequencies (Hz)
    - N: number of sensors (5)
    - spacing_lambda: spacing in wavelengths (0.5)
    - nbar, sll_db: Taylor taper parameters (nbar=4, sll=30)
    - R: covariance matrix (N x N) or None (identity used)
    Returns:
      freqs: array
      W: (len(freqs), N) complex weights for each frequency
      tap: (N,) taper amplitudes applied to the steering vectors when forming constraints
    """
    angles = np.atleast_1d(angles_deg)
    M = angles.size
    tap = get_taylor_taper(N, nbar=nbar, sll_db=sll_db)  # length N, max=1

    # Use identity covariance (white noise) if none provided
    if R is None:
        R = np.eye(N, dtype=complex)

    W = np.zeros((len(freqs), N), dtype=complex)
    for i, f_hz in enumerate(freqs):
        # steering matrix (N x M)
        C = np.column_stack([steering_vector_ula(N, th, f_hz, spacing_lambda, c) for th in angles])
        # incorporate taper by pre-multiplying steering vectors: C_t = diag(tap) @ C
        C_t = (tap[:, None] * C)
        # solve LCMV for tapered constraints (weights apply to raw sensors)
        w = lcmv_weights(R, C_t, desired=np.ones(M), diag_load=diag_load)
        W[i, :] = w
    return freqs, W, tap


def beampattern_for_weights(w, N, f_hz, spacing_lambda=0.5, c=1500.0, angles_deg=None):
    if angles_deg is None:
        angles_deg = np.linspace(-90, 90, 721)
    A = np.column_stack([steering_vector_ula(N, th, f_hz, spacing_lambda, c) for th in angles_deg])
    resp = np.conjugate(w) @ A
    resp_db = 20 * np.log10(np.abs(resp) / np.max(np.abs(resp)) + 1e-12)
    return angles_deg, resp_db


def plot_broadband_beampattern_average(freqs, W, N, spacing_lambda=0.5, c=1500.0,
                                       angles_scan=None, title=None):
    if not HAS_PLT:
        print("matplotlib not available — skipping plotting of beampattern.")
        return
    if angles_scan is None:
        angles_scan = np.linspace(-90, 90, 721)
    # average power across band (linear)
    P_avg = np.zeros(len(angles_scan))
    for i, f_hz in enumerate(freqs):
        _, resp_db = beampattern_for_weights(W[i, :], N, f_hz, spacing_lambda, c, angles_scan)
        P_avg += 10 ** (resp_db / 10.0)  # convert dB to linear (relative)
    P_avg = P_avg / len(freqs)
    # convert back to dB relative to peak
    P_avg_db = 10 * np.log10(P_avg + 1e-12)
    P_avg_db = P_avg_db - np.max(P_avg_db)
    plt.figure(figsize=(8, 4))
    plt.plot(angles_scan, P_avg_db)
    plt.ylim([-60, 0])
    plt.grid(True)
    plt.xlabel("Angle (deg)")
    plt.ylabel("Average Response (dB, normalized)")
    if title:
        plt.title(title)
    plt.show()


def plot_polar_beampattern(w, N, f_hz, spacing_lambda=0.5, c=1500.0, title=None, angles_deg=None):
    """
    Plot polar beampattern for weights w at frequency f_hz.
    angles_deg: sampling angles in degrees (default -180..180 mapped to polar plot)
    """
    if not HAS_PLT:
        print("matplotlib not available — skipping polar plot.")
        return
    if angles_deg is None:
        angles_deg = np.linspace(-180, 180, 721)
    # compute magnitude (linear)
    _, resp_db = beampattern_for_weights(w, N, f_hz, spacing_lambda, c, angles_deg=np.clip(angles_deg, -90, 90))
    # beampattern_for_weights expects -90..90; for polar we mirror to 180 range by padding
    # create full 360 by mirroring
    angles_half = np.linspace(-90, 90, len(resp_db))
    resp_linear = 10 ** (resp_db / 20.0)
    # mirror to get -180..180 (assume symmetric for ULA)
    resp_full = np.concatenate((resp_linear[::-1], resp_linear[1:]))
    angles_full = np.linspace(-180, 180, len(resp_full))

    # Normalize
    resp_full = resp_full / np.max(resp_full)

    theta = np.deg2rad(angles_full)
    r = resp_full

    ax = plt.subplot(projection='polar')
    ax.plot(theta, r)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rmax(1.0)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.grid(True)
    if title:
        ax.set_title(title, va='bottom')
    plt.show()



# DOA simulation and MUSIC estimator




def simulate_snapshots(angles_deg, N, f_hz, fs, T, spacing_lambda=0.5, c=1500.0, SNR_dB=0.0):
    """
    Simulate narrowband plane-wave snapshots for a ULA.
    - angles_deg: sequence of source DOAs in degrees
    - N: number of sensors
    - f_hz: carrier frequency (Hz)
    - fs: sampling frequency (Hz)
    - T: total duration (s)
    - spacing_lambda: spacing in wavelengths
    - c: propagation speed (m/s)
    - SNR_dB: per-sensor SNR in dB (signal power per sensor relative to noise)

    Returns:
      X: (N, L) complex array of snapshots
    """
    angles = np.atleast_1d(angles_deg)
    M = angles.size
    L = max(int(np.round(fs * T)), 16)
    t = np.arange(L) / fs

    # steering matrix (N x M)
    A = np.column_stack([steering_vector_ula(N, th, f_hz, spacing_lambda, c) for th in angles])

    # generate narrowband source signals (complex exponentials with random phases)
    phases = 2 * np.pi * np.random.rand(M)
    s = np.exp(2j * np.pi * f_hz * t[None, :] + 1j * phases[:, None])

    # scale sources to unit power per source per snapshot (mean power = 1)
    # compute signal at sensors
    Xsig = A @ s  # shape (N, L)

    # compute noise power needed for desired SNR per sensor
    sig_power = np.mean(np.abs(Xsig)**2)
    snr_lin = 10 ** (SNR_dB / 10.0)
    noise_power = sig_power / max(snr_lin, 1e-12)

    noise = (np.sqrt(noise_power/2) * (np.random.randn(N, L) + 1j * np.random.randn(N, L)))
    X = Xsig + noise
    return X


def sample_covariance(X):
    """Compute sample covariance matrix from data X (N x L)."""
    N, L = X.shape
    R = (X @ X.conj().T) / L
    return R


def music_spectrum(R, n_sources, angles_scan, N, f_hz, spacing_lambda=0.5, c=1500.0):
    """
    Compute MUSIC pseudospectrum over angles_scan.
    - R: covariance matrix (N x N)
    - n_sources: number of signal sources to estimate (M)
    - angles_scan: 1D array of angles (degrees) to evaluate
    Returns:
      angles_scan, P_dB (same length)
    """
    # eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(R)
    # sort ascending
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # noise subspace: eigenvectors corresponding to smallest N - n_sources eigenvalues
    En = eigvecs[:, :max(0, N - n_sources)]  # N x (N-n_sources)

    P = np.zeros(len(angles_scan), dtype=float)
    for k, th in enumerate(angles_scan):
        a = steering_vector_ula(N, th, f_hz, spacing_lambda, c)
        # projection of a onto noise subspace
        denom = np.linalg.norm(En.conj().T @ a) ** 2
        P[k] = 1.0 / (denom + 1e-12)
    # convert to dB
    P_db = 10 * np.log10(P / np.max(P) + 1e-12)
    return angles_scan, P_db


def find_peaks_simple(y, num_peaks=1):
    """
    Find local maxima in array y and return indices of top num_peaks peaks.
    Simple implementation without scipy.
    """
    N = len(y)
    peaks = []
    for i in range(1, N - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            peaks.append((y[i], i))
    if not peaks:
        # fallback: global max
        return [int(np.argmax(y))]
    peaks.sort(reverse=True)
    idxs = [p[1] for p in peaks[:num_peaks]]
    return idxs


def estimate_doa_music(X, n_sources, f_hz, spacing_lambda=0.5, c=1500.0, angles_scan=None):
    """
    Estimate DOAs from data X (N x L) using MUSIC at frequency f_hz.
    Returns estimated angles (degrees) and pseudospectrum (angles_scan, P_db).
    """
    N, L = X.shape
    R = sample_covariance(X)
    if angles_scan is None:
        angles_scan = np.linspace(-90, 90, 721)
    angles_scan, P_db = music_spectrum(R, n_sources, angles_scan, N, f_hz, spacing_lambda, c)
    # find peaks
    peak_idxs = find_peaks_simple(P_db, num_peaks=n_sources)
    est_angles = angles_scan[sorted(peak_idxs)]
    # sort by angle for readability
    est_angles = np.sort(est_angles)
    return est_angles, (angles_scan, P_db)


# Benchmarking utilities

import time
import sys

# optional psutil for accurate memory measurements
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

try:
    import resource
    HAS_RESOURCE = True
except Exception:
    HAS_RESOURCE = False


def get_current_memory_bytes():
    """
    Return current process memory usage in bytes. Uses psutil if available, else resource.
    If neither is available, returns None.
    """
    try:
        if HAS_PSUTIL:
            p = psutil.Process()
            return p.memory_info().rss
        if HAS_RESOURCE:
            # ru_maxrss is in kilobytes on Linux, bytes on some systems — convert conservatively
            ru = resource.getrusage(resource.RUSAGE_SELF)
            mrss = getattr(ru, 'ru_maxrss', None)
            if mrss is None:
                return None
            # On Linux, ru_maxrss is in kilobytes
            return int(mrss) * 1024
    except Exception:
        return None
    return None


def benchmark_pipeline(N=4, n_freqs=64, n_sources=2, f_demo=500.0, fs=8000.0, T=0.5,
                       spacing_lambda=0.5, c=1500.0, nbar=2, sll_db=30.0, iterations=3):
    """
    Run a small benchmark of the core operations:
    - compute_broadband_lcmv over n_freqs
    - simulate snapshots and run MUSIC once per iteration
    Measures elapsed time and memory before/after. Returns a summary dict.
    ``iterations`` repeats the per-iteration MUSIC step to give a small sample of runtime.
    """
    summary = {}

    # warm-up call (JIT-like cache effects for BLAS)
    freqs = np.linspace(10.0, 3000.0, n_freqs)
    desired_angles = np.linspace(-30, 30, n_sources)

    t0 = time.perf_counter()
    freqs_out, W, taper = compute_broadband_lcmv(desired_angles, freqs, N=N,
                                                 spacing_lambda=spacing_lambda, c=c,
                                                 nbar=nbar, sll_db=sll_db, R=None,
                                                 diag_load=1e-3)
    t1 = time.perf_counter()
    time_lcmv = t1 - t0

    # memory after LCMV
    mem_after_lcmv = get_current_memory_bytes()

    # now repeat MUSIC simulation and estimation iterations times
    times_music = []
    mem_before = get_current_memory_bytes()
    for i in range(iterations):
        X = simulate_snapshots(np.linspace(-25, 15, n_sources), N, f_demo, fs, T,
                               spacing_lambda=spacing_lambda, c=c, SNR_dB=0.0)
        t_before = time.perf_counter()
        est_angles, (angles_scan, P_db) = estimate_doa_music(X, n_sources, f_demo, spacing_lambda, c)
        t_after = time.perf_counter()
        times_music.append(t_after - t_before)
    mem_after = get_current_memory_bytes()

    summary['N'] = N
    summary['n_freqs'] = n_freqs
    summary['time_lcmv_sec'] = time_lcmv
    summary['time_music_avg_sec'] = float(np.mean(times_music))
    summary['time_music_std_sec'] = float(np.std(times_music))
    summary['mem_before_bytes'] = mem_before
    summary['mem_after_lcmv_bytes'] = mem_after_lcmv
    summary['mem_after_bytes'] = mem_after
    summary['taper_used'] = taper.tolist() if isinstance(taper, np.ndarray) else str(taper)
    summary['est_angles_last'] = est_angles.tolist()

    return summary


# Demo extension in main: simulate sources and run MUSIC
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Broadband LCMV + MUSIC demo for hydrophone array')
    parser.add_argument('--N', type=int, default=5, help='Number of hydrophones in the ULA (e.g., 4 or 5)')
    parser.add_argument('--freq', type=float, default=500.0, help='Demo narrowband frequency for MUSIC (Hz)')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting (useful for headless runs)')
    parser.add_argument('--benchmark', action='store_true', help='Run quick benchmark (timing+memory)')
    parser.add_argument('--bench-iterations', type=int, default=3, help='Iterations for MUSIC timing')
    parser.add_argument('--bench-nfreqs', type=int, default=64, help='Number of frequencies for broadband test')
    parser.add_argument('--bench-N', type=int, default=4, help='Number of sensors for benchmark')
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed for reproducible simulations')
    parser.add_argument('--beams', action='store_true', help='Compute and plot/save per-target beam beampatterns')
    parser.add_argument('--save-plots', action='store_true', help='When plotting beams, save plots to PNG files instead of showing')
    args = parser.parse_args()

    if getattr(args, 'benchmark', False):
        print('Running benchmark...')
        bench = benchmark_pipeline(N=args.bench_N, n_freqs=args.bench_nfreqs, n_sources=2,
                                  f_demo=args.freq, fs=8000.0, T=0.5, spacing_lambda=0.5,
                                  c=1500.0, nbar=2, sll_db=30.0, iterations=args.bench_iterations)
        # Print friendly results
        print('\nBenchmark summary:')
        print(f"Array sensors (N): {bench['N']}")
        print(f"Frequencies used: {bench['n_freqs']}")
        print(f"LCMV compute time: {bench['time_lcmv_sec']:.4f} s")
        print(f"MUSIC avg time per estimate: {bench['time_music_avg_sec']:.4f} s (std {bench['time_music_std_sec']:.4f})")
        if bench['mem_before_bytes'] is not None:
            print(f"Memory before (bytes): {bench['mem_before_bytes']}")
        if bench['mem_after_lcmv_bytes'] is not None:
            print(f"Memory after LCMV (bytes): {bench['mem_after_lcmv_bytes']}")
        if bench['mem_after_bytes'] is not None:
            print(f"Memory after MUSIC (bytes): {bench['mem_after_bytes']}")
        print('Taper used (amplitudes):', np.round(np.array(bench['taper_used']), 4))
        print('Last estimated angles (deg):', bench['est_angles_last'])

        # Provide a short recommendation for Raspberry Pi
        print('\nRecommendation for Raspberry Pi:')
        print('- Use --no-plot to disable matplotlib when running on Pi headless.')
        print('- Reduce bench n_freqs and snapshot lengths for real-time constraints.')
        print('- Install psutil on Pi for more accurate memory reporting: pip install psutil')

        # exit after benchmark
        sys.exit(0)

    # Allow user to override number of sensors (you mentioned you have 4 channels on Focusrite 18i8)
    N = args.N
    if N < 1:
        print('Invalid N, must be >= 1'); sys.exit(1)

    # Note about Focusrite mapping
    # If you use Focusrite 18i8 (2nd gen) with 4 channels connected to 4 hydrophones, map them to array elements
    # in physical order (e.g., channel 1 -> element 0, ch2 -> element1, ...). The code here assumes contiguous,
    # uniformly spaced ULA elements and that the input order matches physical order.

    spacing_lambda = 0.5
    c = 1500.0              # sound speed in water (m/s)
    # detection band 10 - 3000 Hz
    f_min, f_max = 10.0, 3000.0
    n_freqs = 128
    freqs = np.linspace(f_min, f_max, n_freqs)

    # Taylor taper parameters: nbar=4, sll=30 dB (will be clamped/fallback for small N)
    nbar = 4
    sll_db = 30.0

    # Example desired mainlobe angles (user can modify)
    desired_angles = [-30.0, 20.0]

    # If you have sample covariance matrix for the band, compute or load it and pass as R.
    R = None  # use identity (white noise) by default

    # If user provided a seed, set NumPy RNG for reproducible snapshots/results
    if getattr(args, 'seed', None) is not None:
        np.random.seed(int(args.seed))
        print(f"Using random seed: {int(args.seed)}")

    freqs_out, W, taper = compute_broadband_lcmv(desired_angles, freqs, N=N,
                                                 spacing_lambda=spacing_lambda, c=c,
                                                 nbar=nbar, sll_db=sll_db, R=R,
                                                 diag_load=1e-3)

    # If requested, compute per-target beamformers and show/save their averaged beampatterns
    if getattr(args, 'beams', False):
        freqs_t, Wt, taper_t = compute_broadband_lcmv_per_target(desired_angles, freqs, N=N,
                                                                 spacing_lambda=spacing_lambda, c=c,
                                                                 nbar=nbar, sll_db=sll_db, R=R,
                                                                 diag_load=1e-3)
        title_prefix = f"Broadband LCMV per-target (N={N})"
        plot_each_beam_average(freqs_t, Wt, desired_angles, N, spacing_lambda=spacing_lambda, c=c,
                               angles_scan=np.linspace(-90, 90, 721), title_prefix=title_prefix,
                               save_plots=getattr(args, 'save_plots', False))

    # Try to plot the average beampattern across the band (if matplotlib available and not disabled)
    title = f"Broadband LCMV with Taylor taper nbar={nbar}, sll={sll_db} dB (N={N})"
    if not args.no_plot:
        try:
            plot_broadband_beampattern_average(freqs_out, W, N, spacing_lambda, c,
                                               angles_scan=np.linspace(-90, 90, 721),
                                               title=title)
        except Exception as e:
            print("Plotting failed or not available:", e)

    # Provide a console summary so the script can be used headless
    center_idx = len(freqs_out) // 2
    w_center = W[center_idx, :]
    print("Center frequency (Hz):", freqs_out[center_idx])
    print("Taylor taper (amplitude):", np.round(taper, 4))
    print("Weights at center freq (complex):")
    np.set_printoptions(precision=4, suppress=True)
    print(w_center)

    # Simulation parameters for DOA demo
    fs = 8000.0        # sampling rate for simulation (Hz)
    T = 0.5            # seconds of data
    f_demo = args.freq     # narrowband frequency to run MUSIC at (Hz) — choose within band
    true_angles = [-25.0, 15.0]
    n_sources = len(true_angles)
    SNR_dB = 0.0

    print("\nRunning DOA simulation (MUSIC) demo...")
    X = simulate_snapshots(true_angles, N, f_demo, fs, T, spacing_lambda, c, SNR_dB)
    est_angles, (angles_scan, P_db) = estimate_doa_music(X, n_sources, f_demo, spacing_lambda, c)
    print("True angles:", true_angles)
    print("Estimated angles (deg):", np.round(est_angles, 2))

    if not args.no_plot and HAS_PLT:
        # plot pseudospectrum
        plt.figure(figsize=(8, 4))
        plt.plot(angles_scan, P_db)
        plt.xlabel('Angle (deg)')
        plt.ylabel('MUSIC Pseudospectrum (dB)')
        plt.title('MUSIC pseudospectrum')
        plt.grid(True)
        # mark true and estimated angles
        for a in true_angles:
            plt.axvline(a, color='g', linestyle='--', alpha=0.6)
        for a in est_angles:
            plt.axvline(a, color='r', linestyle=':', alpha=0.8)
        plt.show()

        # polar plot of center-frequency beampattern
        try:
            plot_polar_beampattern(w_center, N, freqs_out[center_idx], spacing_lambda, c,
                                    title=f'Polar beampattern at {freqs_out[center_idx]:.1f} Hz (N={N})')
        except Exception as e:
            print('Polar plot failed:', e)
    else:
        if not HAS_PLT:
            print('matplotlib not available — skipping plots')
        else:
            print('Plotting disabled by --no-plot')
