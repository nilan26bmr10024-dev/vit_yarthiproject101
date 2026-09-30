"""Symbolic calculus: differentiation, integration, ODEs (pure functions, no I/O)."""
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

x = sp.Symbol("x")
y = sp.Function("y")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_ODE_NAMES = {"x": x, "y": y(x), "dydx": y(x).diff(x),
              "dy2dx2": y(x).diff(x, 2), "d3ydx3": y(x).diff(x, 3)}


def parse(text, local_dict=None):
    """Parse a string into a SymPy expression (supports 2x, x^2, sin(x)...)."""
    return parse_expr(text, local_dict={"x": x, **(local_dict or {})},
                      transformations=_TRANSFORMS)


def differentiate(text, order=1, at=None):
    """d^order/dx^order of expression; optionally evaluated at x=at."""
    result = sp.diff(parse(text), x, order)
    return result if at is None else result.subs(x, sp.sympify(at))


def integrate_indefinite(text):
    return sp.integrate(parse(text), x)


def integrate_definite(text, lower, upper):
    return sp.integrate(parse(text), (x, sp.sympify(lower), sp.sympify(upper)))


def solve_ode(text, ics=None):
    """Solve an ODE written as an expression equal to 0, e.g. 'dydx - y'.
    ics: optional {x0: y0}, e.g. {1: 8}."""
    eqn = sp.Eq(parse(text, _ODE_NAMES), 0)
    ics = {y(sp.sympify(k)): sp.sympify(v) for k, v in (ics or {}).items()}
    return sp.dsolve(eqn, y(x), ics=ics or None)
