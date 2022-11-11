import numpy as np

from result import Result


def find_u(x1, x2, x3, y1, y2, y3):
    a = (y2 * x3 + y1 * x2 + y3 * x1 - y1 * y3 - y2 * x1 - y3 * x2) / (
            (x1 - x2) * (x3 * x3 + x1 * x2 - x1 * x3 - x2 * x3))
    b = (y1 - y3 - a * (x1 * x1 - x3 * x3)) / (x1 - x3)
    return -b / (2 * a)


def one_dim(a, c, f, eps=1e-6, max_steps=100):
    res = Result()
    K = (3 - np.sqrt(5)) / 2
    x = w = v = a + K * (c - a)
    fx = fw = fv = f(x)
    res.add_point(x, fx)
    d = e = c - a
    for i in range(max_steps):
        g = e
        e = d
        tol = eps * np.abs(x) + eps / 10
        if np.abs(x - (a + c) / 2) + (c - a) / 2 <= 2 * tol:
            break
        accept = False
        if x != w and x != v and w != v and fx != fw and fx != fv and fw != fv:
            u = find_u(x, w, v, fx, fw, fv)
            if u >= a and u <= c and np.abs(u - x) < g / 2:
                accept = True
                if u - a < 2 * tol or c - u < 2 * tol:
                    u = x - np.sign(x - (a + c) / 2) * tol
        if not accept:
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
            res.add_point(x, fx)
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
    res.set(x, fx)
    return res


def find_u_with_derivative(x1, x2, df1, df2):
    a = (df1 - df2) / (x1 - x2)
    b = df1 - a * x1
    return -b / (2 * a)


def one_dim_with_derivative(a, c, f, df, eps=1e-6, max_steps=100):
    res = Result()
    x = w = v = (a + c) / 2
    fx = fw = fv = f(x)
    dfx = dfw = dfv = df(x)
    res.add_point(x, fx)
    d = e = c - a
    for i in range(max_steps):
        g = e
        e = d
        accept_u1 = False
        accept_u2 = False
        if abs(x - (a + c) / 2 + (c - a) / 2 <= 2 * (eps * abs(x) + eps / 10)):
            break
        if x != w and dfx != dfw:
            u1 = find_u_with_derivative(x, w, dfx, dfw)
            if u1 >= a and u1 <= c and (u1 - x) * dfx <= 0 and np.abs(u1 - x) < g / 2:
                accept_u1 = True
        if x != v and dfx != dfv:
            u2 = find_u_with_derivative(x, v, dfx, dfv)
            if u2 >= a and u2 <= c and (u2 - x) * dfx <= 0 and np.abs(u2 - x) < g / 2:
                accept_u2 = True

        if accept_u1 and not accept_u2:
            u = u1
        if accept_u2 and not accept_u1:
            u = u2
        if accept_u1 and accept_u2:
            if np.abs(u1 - x) < np.abs(u2 - x):
                u = u1
            else:
                u = u2
        if not (accept_u1 or accept_u2):
            if dfx > 0:
                u = (a + x) / 2
                e = x - a
            else:
                u = (x + c) / 2
                e = c - x
        if np.abs(u - x) < eps:
            u = x + np.sign(u - x) * eps
        d = np.abs(u - x)
        fu = f(u)
        dfu = df(u)
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
            dfv = dfw
            dfw = dfx
            dfx = dfu
            res.add_point(x, fx)
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
                dfv = dfw
                dfw = dfu
            else:
                if fu <= fv or v == x or v == w:
                    v = u
                    fv = fu
                    dfv = dfu
    res.set(x, fx, dfx)
    return res
