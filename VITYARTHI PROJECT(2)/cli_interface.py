"""Menu workflow: same menus as calculus.py + functionplotter.py, in one CLI."""
import calculus
import plotter
import stats


def _safe(fn):
    try:
        fn()
    except Exception as e:
        print(f"Error: {e}")


def diff_eqn_menu():
    a = input("Enter The Differential Equation (use y, dydx, dy2dx2, d3ydx3) = 0: ")
    x0 = input("Initial x (blank for none): ").strip()
    ics = {x0: input(f"y({x0}) = ")} if x0 else None
    _safe(lambda: print(calculus.solve_ode(a, ics)))


def integration_menu():
    while True:
        print("1.Indefinite Integration\n2.Definite Integration\n3.Exit From Loop")
        ch = input("Enter Choice:")
        if ch == "1":
            a = input("Enter The Function To Integrate in terms of x:")
            _safe(lambda: print(calculus.integrate_indefinite(a)))
        elif ch == "2":
            a = input("Enter The Function To Integrate in terms of x:")
            lo, up = input("Enter Lower Limit:"), input("Enter Upper Limit:")
            _safe(lambda: print(calculus.integrate_definite(a, lo, up)))
        elif ch == "3":
            break


def differentiation_menu():
    while True:
        print("1.Differentiation Without Limits\n2.Differentiation With Limits\n3.Exit This Menu")
        ch = input("Enter The Choice:")
        if ch in ("1", "2"):
            a = input("Enter The Function In Terms Of x:")
            s = int(input("Order of differentiation:"))
            at = input("Evaluate at x = ") if ch == "2" else None
            _safe(lambda: print(calculus.differentiate(a, s, at)))
        elif ch == "3":
            break


def _floats(text):
    return [float(v) for v in text.split(",")]


def statistics_menu():
    while True:
        print("\n1.Summary Statistics\n2.Fit A Polynomial Curve\n3.Histogram")
        print("4.Normal Distribution (fit + probability)\n5.Exit This Menu")
        ch = input("Enter The Choice:")
        if ch == "1":
            d = input("Enter numbers (comma-separated): ")
            _safe(lambda: [print(f"{k}: {v:.4f}") for k, v in stats.summary_stats(_floats(d)).items()])
        elif ch == "2":
            xs = input("X values (comma-separated): "); ys = input("Y values (comma-separated): ")
            deg = int(input("Degree of polynomial: "))
            def run():
                co, r2 = stats.fit_polynomial(_floats(xs), _floats(ys), deg)
                print("Coefficients (highest power first):", co)
                print(f"R^2 = {r2:.4f}  Correlation = {stats.correlation(_floats(xs), _floats(ys)):.4f}")
                plotter.plot_data_with_fit(_floats(xs), _floats(ys), co)
            _safe(run)
        elif ch == "3":
            d = input("Enter numbers (comma-separated): ")
            _safe(lambda: plotter.plot_histogram(_floats(d)))
        elif ch == "4":
            d = input("Enter numbers (comma-separated): ")
            def run():
                mu, sigma = stats.fit_normal(_floats(d))
                print(f"Fitted mean = {mu:.4f}, std = {sigma:.4f}")
                v = float(input("Find P(X <= value), value = "))
                print(f"P(X <= {v}) = {stats.normal_cdf(v, mu, sigma):.4f}")
            _safe(run)
        elif ch == "5":
            break


def plot_menu():
    e = input("Enter Function (e.g., x**2 - 4): ")
    def run():
        pts = plotter.get_points(e)
        print("\nYour calculated (x, y) points are:")
        for px, py in pts:
            print(f"X: {px:.1f}, Y: {py:.1f}")
        d = input("Overlay derivative? (y/n): ").lower() == "y"
        plotter.plot_function(e, with_derivative=d)
    _safe(run)


def main():
    while True:
        print("\n1.Solve A Differential Equation\n2.Solve An Integration Equation")
        print("3.Solve A Derivative Equation\n4.Plot A Function\n5.Statistics\n6.Exit")
        ch = input("Enter The Choice:")
        if ch == "1": diff_eqn_menu()
        elif ch == "2": integration_menu()
        elif ch == "3": differentiation_menu()
        elif ch == "4": plot_menu()
        elif ch == "5": statistics_menu()
        elif ch == "6": break


if __name__ == "__main__":
    main()
