import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class IzhikevichParams:
    """
    Parameters of the Izhikevich model.

    Attributes
    ----------
    I : callable
        Input current I(t).
    g : array_like
        Coefficients [g_0, g_1, g_2] of the quadratic term.
    v_minus : float
        Post-spike reset value of v.
    v_plus : float
        Spike threshold for v.
    b : float
        Time scale of the recovery variable u.
    c : float
        Sensitivity of u to subthreshold fluctuations of v.
    delta_u : float
        Post-spike increment of u.
    """
    I: Callable
    g: np.ndarray
    v_minus: float
    v_plus: float
    b: float
    c: float
    delta_u: float


def izhikevich_model(x, t, p):
    """
    Derivative function of the Izhikevich model:
        v'(t) = g_2 v^2(t) + g_1 v(t) + g_0 - u(t) + I(t)
        u'(t) = b (c v(t) - u(t))

    Parameters
    ----------
    x : array_like
        State [v, u].
        - v: membrane potential [mV].
        - u: membrane recovery variable.
    t : float
        Current time.
    p : IzhikevichParams
        Model parameters.

    Returns
    -------
    dx : ndarray
        Time derivative [v'(t), u'(t)].
    """
    v, u = x

    dv = p.g[2] * v**2 + p.g[1] * v + p.g[0] - u + p.I(t)
    du = p.b * (p.c * v - u)

    return np.array([dv, du])


def reset(x, t, p):
    """
    Potential reset mechanism (neuronal firing):
        if v >= v_+ :  v <- v_-,  u <- u + delta_u
    
    Parameters
    ----------
    x : array_like
        State [v, u].
        - v: membrane potential [mV].
        - u: membrane recovery variable.
    t : float
        Current time (unused here).
    p : IzhikevichParams
        Model parameters.

    Returns
    -------
    x : ndarray
        State [v, u] after applying the reset.
    """
    v, u = x

    if v >= p.v_plus:
        v = p.v_minus
        u = u + p.delta_u

    return np.array([v, u])
