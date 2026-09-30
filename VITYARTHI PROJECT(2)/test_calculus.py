import unittest
import matplotlib
matplotlib.use("Agg")
import numpy as np
import sympy as sp
import calculus as c
import plotter
import stats

x = c.x


class TestDifferentiation(unittest.TestCase):
    def test_first_and_higher_order(self):
        self.assertEqual(c.differentiate("x**3"), 3 * x**2)
        self.assertEqual(c.differentiate("x**3", 2), 6 * x)

    def test_with_point(self):
        self.assertEqual(c.differentiate("x**2", 1, 3), 6)


class TestIntegration(unittest.TestCase):
    def test_indefinite(self):
        self.assertEqual(sp.simplify(c.integrate_indefinite("2*x") - x**2), 0)

    def test_definite_string_limits(self):
        self.assertEqual(c.integrate_definite("x", "0", "2"), 2)


class TestDiffEq(unittest.TestCase):
    def test_original_ic_y1_equals_8(self):
        sol = c.solve_ode("dydx - y", {1: 8})
        self.assertAlmostEqual(float(sol.rhs.subs(x, 1)), 8.0)


class TestPlotter(unittest.TestCase):
    def test_points_match_original_grid(self):
        pts = plotter.get_points("x**2 - 4")
        self.assertEqual(len(pts), 7)
        self.assertEqual((pts[0][0], pts[0][1]), (-3.0, 5.0))

    def test_constant_expression(self):
        self.assertEqual(list(plotter.evaluate("5", np.array([0., 1.]))), [5.0, 5.0])

    def test_plot_with_derivative_runs(self):
        self.assertEqual(len(plotter.plot_function("x**2", show=False, with_derivative=True)), 7)


class TestStats(unittest.TestCase):
    def test_summary(self):
        s = stats.summary_stats([1, 2, 3, 4, 5])
        self.assertEqual((s["mean"], s["median"]), (3.0, 3.0))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            stats.summary_stats([])

    def test_linear_fit(self):
        co, r2 = stats.fit_polynomial([0, 1, 2, 3], [1, 3, 5, 7], 1)
        self.assertAlmostEqual(co[0], 2.0); self.assertAlmostEqual(co[1], 1.0)
        self.assertAlmostEqual(r2, 1.0)

    def test_normal(self):
        self.assertAlmostEqual(stats.normal_cdf(0), 0.5)
        mu, sd = stats.fit_normal([1, 2, 3])
        self.assertAlmostEqual(mu, 2.0)

    def test_stat_plots_run(self):
        plotter.plot_histogram([1, 2, 2, 3], show=False)
        plotter.plot_data_with_fit([0, 1, 2], [1, 3, 5], [2, 1], show=False)


if __name__ == "__main__":
    unittest.main()
