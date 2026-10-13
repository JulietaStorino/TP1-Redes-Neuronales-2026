import os
import numpy as np
import matplotlib.pyplot as plt


def euler(f, x, t, h, p):
    """
    Computes a single integration step using Euler's method.

    Parameters
    ----------
    f : callable
        Derivative function f(x, t, p) calculating dx/dt.
    x : float or array_like
        State vector at current time step.
    t : float
        Current time value.
    h : float
        Time step size.
    p : tuple, list, dict, or scalar
        Parameters to evaluate f.

    Returns
    -------
    x_next : float or ndarray
        Updated state vector after step size h.
    """
    return x + h * f(x, t, p)


def rk4(f, x, t, h, p):
    """
    Computes a single integration step using the 4th-order Runge-Kutta (RK4) method.

    Parameters
    ----------
    f : callable
        Derivative function f(x, t, p) calculating dx/dt.
    x : float or array_like
        State vector at current time step.
    t : float
        Current time value.
    h : float
        Time step size.
    p : tuple, list, dict, or scalar
        Parameters to evaluate f.

    Returns
    -------
    x_next : float or ndarray
        Updated state vector x(t + h) using the RK4 method.
    """
    k1 = f(x, t, p)
    k2 = f(x + 0.5 * h * k1, t + 0.5 * h, p)
    k3 = f(x + 0.5 * h * k2, t + 0.5 * h, p)
    k4 = f(x + h * k3, t + h, p)

    return x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def solve_ode(m, f, xa, a, b, k, p, c=lambda x, t, p: x):
    """
    Numerically integrates an ODE system dx/dt = f(x, t, p) over the interval
    [a, b], using k integration steps with method m under initial condition
    x(a) = xa. Note that a does not need to be less than b.

    Parameters
    ----------
    m : callable
        Single-step integration method function (e.g., euler, rk4).
    f : callable
        Derivative function f(x, t, p) calculating dx/dt.
    xa : float or array_like
        Initial state vector at initial time a.
    a : float
        Initial time.
    b : float
        Final time.
    k : int
        Number of integration steps.
    p : tuple, list, dict, or scalar, optional
        Parameters passed to f and c.
    c : callable, optional
        Conditioning or constraint function c(x, t, p) applied to the state
        at each step.

    Returns
    -------
    t : ndarray
        1D array of time points of length k + 1, where t[j] = a + j * h.
    w : ndarray
        Array of state vectors across all time points.
    """
    assert k > 0
    n = len(xa)
    h = (b - a) / k
    w = np.zeros((n, k + 1))

    t = np.zeros(k + 1)
    w[:, 0] = xa

    t[0] = a

    for j in range(k):
        t[j + 1] = t[j] + h
        w[:, j + 1] = m(f, w[:, j], t[j], h, p)
        w[:, j + 1] = c(w[:, j + 1], t[j + 1], p)

    return t, w

# ------------------------------------------------------------------
# Problem Set 4 - Exercise 4: exponential growth.
# Implemented only as a test of the code.
# ------------------------------------------------------------------

def _exponential_growth(x, t, p):
    """
    Derivative function for exponential growth, dx/dt = q * x.

    Parameters
    ----------
    x : array_like
        State vector at time t.
    t : float
        Current time value (unused, the system is autonomous).
    p : array_like
        Parameters, where q = p[0] is the growth rate.

    Returns
    -------
    dx : ndarray
        Time derivative of the state vector.
    """
    return p[0] * np.array(x[0])

def _exponential_growth_exact(xa, a, t, p):
    """
    Exact solution of dx/dt = q * x with x(a) = xa.

    Parameters
    ----------
    xa : float or array_like
        Initial state vector at initial time a.
    a : float
        Initial time.
    t : float or array_like
        Time value(s) at which the solution is evaluated.
    p : array_like
        Parameters, where q = p[0] is the growth rate.

    Returns
    -------
    x : float or ndarray
        Exact solution x(t) = xa * exp(q * (t - a)).
    """
    return xa * np.exp(p[0] * (t - a))

# ------------------------------------------------------------------
# Problem Set 4 - Exercise 6: harmonic oscillator.
# Implemented only as a test of the code.
# ------------------------------------------------------------------

def _harmonic_oscillator(x, t, p):
    """
    Derivative function for the harmonic oscillator y'' = -alpha * y,
    reduced to first order with x1 = y and x2 = y'.

    Parameters
    ----------
    x : array_like
        State vector [x1, x2] = [y, y'] at time t.
    t : float
        Current time value (unused, the system is autonomous).
    p : array_like
        Parameters, where alpha = p[0].

    Returns
    -------
    dx : ndarray
        Time derivative [x1', x2'] = [x2, -alpha * x1].
    """
    return np.array([x[1], -p[0] * x[0]])


def main():
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Exponential growth
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    a = 0
    b = 3
    h = 0.1

    k = int((b - a) / h)
    xa = np.array([1]) # x(a) = x(0) = 1
    p = np.array([1])

    t_euler, w_euler = solve_ode(euler, _exponential_growth, xa, a, b, k, p)
    t_rk4, w_rk4 = solve_ode(rk4, _exponential_growth, xa, a, b, k, p)

    # Continuous exact solution
    t_exact = np.linspace(a, b, 500)
    x_exact = _exponential_growth_exact(xa, a, t_exact, p)

    # Discrete exact solution for error calculation
    x_exact_euler = _exponential_growth_exact(xa, a, t_euler, p).squeeze()
    x_exact_rk4 = _exponential_growth_exact(xa, a, t_rk4, p).squeeze()

    ax = axes[0]

    ax.scatter(t_euler, w_euler[0,:], label="Euler's method", c="red", s=10)
    ax.scatter(t_rk4, w_rk4[0,:], label="RK4 method", c="blue", s=10)

    ax.plot(t_exact, x_exact.squeeze(), label="Exact", color="black", linestyle="-", linewidth=0.8)

    ax.set_xlabel(r"$t$", fontsize=12)
    ax.set_ylabel(r"$x(t)$", fontsize=12)

    ax.set_title(f"Exponential growth ($q={p[0]}, h={h}, x_0={xa[0]}$)", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")

    ax = axes[1]

    ax.scatter(t_euler, w_euler[0, :] - x_exact_euler, c="red", s=10, label="Euler err")
    ax.scatter(t_rk4, w_rk4[0, :] - x_exact_rk4, c="blue", s=10, label="RK4 err")

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)

    ax.set_xlabel(r"$t$", fontsize=12)
    ax.set_ylabel(r"$\Delta$", fontsize=10)

    plt.savefig(f"{output_dir}/exponential_growth.png", format="png")
    plt.close()

    # ------------------------------------------------------------------
    # Harmonic oscillator
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex="col", gridspec_kw={"height_ratios": [3, 1]})

    a = 0
    b = 10
    h = 0.1

    k = round((b - a) / h)
    xa = np.array([1.0, 0.0])  # x1(a) = y(a) = 1, x2(a) = y'(a) = 0
    p = np.array([3.0])  # alpha = 3

    # Continuous exact solution
    t_exact = np.linspace(a, b, 500)
    x1_exact = np.cos(np.sqrt(p[0]) * t_exact)
    x2_exact = -np.sqrt(p[0]) * np.sin(np.sqrt(p[0]) * t_exact)

    # Euler
    t_euler, w_euler = solve_ode(euler, _harmonic_oscillator, xa, a, b, k, p)
    x1_exact_euler = np.cos(np.sqrt(p[0]) * t_euler)
    x2_exact_euler = -np.sqrt(p[0]) * np.sin(np.sqrt(p[0]) * t_euler)

    ax = axes[0, 0]

    ax.scatter(t_euler, w_euler[0, :], label=r"$x_1(t)$", c="blue", s=8)
    ax.scatter(t_euler, w_euler[1, :], label=r"$x_2(t)$", c="red", s=8)

    ax.plot(t_exact, x1_exact, c="blue", linestyle="-", label="Exact")
    ax.plot(t_exact, x2_exact, c="red", linestyle="-", label="Exact")

    ax.set_title(f"Euler $\\alpha={p[0]}, h={h}, x(0)={list([float(v) for v in xa])}$", fontsize=13)

    ax.set_ylabel(r"$x(t)$", fontsize=12)

    ax = axes[1, 0]

    ax.scatter(t_euler, w_euler[0, :] - x1_exact_euler, c="blue", s=6, label=r"$\Delta x_1$")
    ax.scatter(t_euler, w_euler[1, :] - x2_exact_euler, c="red", s=6, label=r"$\Delta x_2$")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel(r"$\Delta x(t)$", fontsize=12)
    ax.set_xlabel(r"$t$", fontsize=12)

    ax.set_ylim(-5, 5)

    # RK4
    t_rk4, w_rk4 = solve_ode(rk4, _harmonic_oscillator, xa, a, b, k, p)
    x1_exact_rk4 = np.cos(np.sqrt(p[0]) * t_rk4)
    x2_exact_rk4 = -np.sqrt(p[0]) * np.sin(np.sqrt(p[0]) * t_rk4)

    ax = axes[0, 1]

    ax.scatter(t_rk4, w_rk4[0, :], label=r"$x_1(t)$", c="blue", s=8)
    ax.scatter(t_rk4, w_rk4[1, :], label=r"$x_2(t)$", c="red", s=8)

    ax.plot(t_exact, x1_exact, c="blue", linestyle="-", label="Exact")
    ax.plot(t_exact, x2_exact, c="red", linestyle="-", label="Exact")

    ax.set_ylabel(r"$x(t)$", fontsize=12)
    ax.set_title(f"RK4 $\\alpha={p[0]}, h={h}, x(0)={list([float(v) for v in xa])}$", fontsize=13)
    ax.legend(fontsize=12, loc="lower right", framealpha=0.9)

    ax = axes[1, 1]

    ax.scatter(t_rk4, w_rk4[0, :] - x1_exact_rk4, c="blue", s=8)
    ax.scatter(t_rk4, w_rk4[1, :] - x2_exact_rk4, c="red", s=8)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel(r"$\Delta x(t)$", fontsize=12)
    ax.set_xlabel(r"$t$", fontsize=12)

    ax.set_ylim(-0.0003, 0.0003)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/harmonic_oscillator.png", format="png")
    plt.close()


if __name__ == "__main__":
    main()