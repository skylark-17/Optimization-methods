import numpy as np
from scipy.interpolate import interp1d


def one_dim(a, c, f, eps=1e-6, max_steps=100):
    K = (3 - np.sqrt(5)) / 2
    x = w = v = a + K * (c - a)
    fx = fw = fv = f(x)
    d = e = c - a
    for i in range(max_steps):
        g = e
        e = d
        tol = eps * np.abs(x) + eps / 10
        if np.abs(x - (a + c) / 2) + (c - a) / 2 <= 2 * tol:
            break
        accept = False
        if x != w and x != v and w != v and fx != fw and fx != fv and fw != fv:
            parabola = interp1d(x=[x, w, v], y=[fx, fw, fv], kind='quadratic', fill_value='extrapolate')
            A = (parabola(1) + parabola(-1)) / 2
            B = (parabola(1) - parabola(-1)) / 2
            u = -B / (2 * A)
            if u >= a and u <= c and np.abs(u - x) < g / 2:
                accept = True
                if u - a < 2 * tol or c - u < 2 * tol:
                    u = x - np.sign(x - (a + c) / 2) * tol
        if accept == False:
            if x < (a + c) / 2:
                u = x + K * (c - x)
                e = c - x
            else:
                u = x - K * (x - a)
                e = x - a
        if np.abs(u - x) < tol:
            u = x + np.sign(u - x) * tol
        d = np.abs(u - x)
        fu = f(u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                c = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        else:
            if u >= x:
                c = u
            else:
                a = u
            if fu <= fw or w == x:
                v = w
                w = u
                fv = fw
                fw = fu
            else:
                if fu <= fv or v == x or v == w:
                    v = w
                    w = u
                    fv = fw
                    fw = fu
    return [x, f(x)]


def one_dim_with_derivative(a, c, f, df, eps, max_steps):
    x = w = v = (a + c) / 2
    fx = fw = fv = f(x)
    dfx = dfw = dfv = df(x)
    d = e = c - a
    for i in range(max_steps):
        g = e
        e = d
        if abs(x - (a + c) / 2 + (c - a) / 2 <= 2 * (eps * abs(x) + eps / 10)):
            break
        if x != w and dfx != dfw:
            pass
    pass
