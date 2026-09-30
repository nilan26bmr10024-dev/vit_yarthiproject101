"""Curve fitting, distributions and summary statistics (NumPy only)."""
import math
import numpy as np


def summary_stats(data):
    a = np.asarray(data, dtype=float)
    if a.size == 0:
        raise ValueError("data is empty")
    return {
        "count": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "variance": float(a.var(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()), "max": float(a.max()),
        "q1": float(np.percentile(a, 25)), "q3": float(np.percentile(a, 75)),
    }


def fit_polynomial(xs, ys, degree=1):
    """Least-squares polynomial fit. Returns (coeffs high->low, r_squared)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) != len(ys) or len(xs) <= degree:
        raise ValueError("need equal-length data with more points than the degree")
    coeffs = np.polyfit(xs, ys, degree)
    ss_res = float(np.sum((ys - np.polyval(coeffs, xs)) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    return coeffs, (1.0 - ss_res / ss_tot if ss_tot else 1.0)


def fit_normal(data):
    """Return (mu, sigma) MLE for a normal distribution."""
    a = np.asarray(data, float)
    return float(a.mean()), float(a.std())


def normal_pdf(xs, mu=0.0, sigma=1.0):
    xs = np.asarray(xs, float)
    return np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def normal_cdf(v, mu=0.0, sigma=1.0):
    return 0.5 * (1 + math.erf((v - mu) / (sigma * math.sqrt(2))))


def correlation(xs, ys):
    return float(np.corrcoef(xs, ys)[0, 1])
