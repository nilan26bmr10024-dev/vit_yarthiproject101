
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import calculus


def evaluate(expression, xs):
    
    fn = sp.lambdify(calculus.x, calculus.parse(expression), modules="numpy")
    ys = fn(xs)
    return np.full_like(xs, ys, dtype=float) if np.isscalar(ys) else np.asarray(ys, float)


def get_points(expression, x_min=-3, x_max=3, num=7):
    
    xs = np.linspace(x_min, x_max, num)
    return list(zip(xs, evaluate(expression, xs)))


def plot_function(expression, x_min=-3, x_max=3, num=7, show=True, save_path=None,
                  with_derivative=False):
    
    points = get_points(expression, x_min, x_max, num)
    xs, ys = zip(*points)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", color="crimson", label=f"f(x) = {expression}")
    if with_derivative:
        d = calculus.differentiate(expression)
        plt.plot(xs, evaluate(str(d), np.array(xs)), "--", color="navy",
                 label=f"f'(x) = {d}")
    plt.title("Plotting Points Generated from a Function")
    plt.xlabel("X Axis"); plt.ylabel("Y Axis")
    plt.grid(True); plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return points


def plot_data_with_fit(xs, ys, coeffs, show=True, save_path=None):
    xs = np.asarray(xs, float)
    grid = np.linspace(xs.min(), xs.max(), 200)
    fig = plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, label="data")
    plt.plot(grid, np.polyval(coeffs, grid), color="crimson", label="fit")
    plt.title("Polynomial Fit"); plt.xlabel("X"); plt.ylabel("Y")
    plt.grid(True); plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)


def plot_histogram(data, bins=10, show=True, save_path=None):
    fig = plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, color="steelblue", edgecolor="black")
    plt.title("Histogram"); plt.xlabel("Value"); plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
