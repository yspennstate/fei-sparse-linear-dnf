"""Exact check of the sparse linear DNF bound (Theorem 1) and of the three intermediate inequalities.

F is a monotone DNF in 0/1 variables (bit j of the index is x_j), all terms of width w, two terms share at most one
variable, every variable in at most D terms; f = 1 - 2F. For every coordinate i where the hypotheses hold and every
k = 1..L, check
  (1) d a theta <= delta / A <= d a,
  (2) mass of hhat^2 outside Omega_k <= min(1, d gamma_k) delta,
  (3) mass of Nhat^2 on Omega_k <= kappa delta,
  (4) |E_i / delta - 2| <= the right side of (LIN),
with exact Walsh-Hadamard transforms."""
import math
import sys
import numpy as np

LN2 = math.log(2)


def wht(v):
    a = np.asarray(v, float).copy()
    h = 1
    while h < len(a):
        a = a.reshape(-1, 2 * h)
        x, y = a[:, :h].copy(), a[:, h:].copy()
        a[:, :h], a[:, h:] = x + y, x - y
        a = a.reshape(-1)
        h *= 2
    return a / len(a)


def eta(x):
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    m = x > 1e-300
    out[m] = -x[m] * np.log2(x[m])
    return out


def Phi(p, q):
    return eta(p) + eta(q) - eta(np.asarray(p) + np.asarray(q))


def random_linear(n, w, D, tries, rng):
    terms = []
    deg = np.zeros(n, int)
    for _ in range(tries):
        t = frozenset(rng.choice(n, w, replace=False).tolist())
        if any(len(t & s) > 1 for s in terms) or any(deg[j] >= D for j in t):
            continue
        terms.append(t)
        for j in t:
            deg[j] += 1
    return terms


def star(D, w):
    return [frozenset([0] + list(range(1 + t * (w - 1), 1 + (t + 1) * (w - 1)))) for t in range(D)]


def grid(m):
    """rows and columns of an m x m grid: width m, D = 2, linear."""
    return [frozenset(r * m + c for c in range(m)) for r in range(m)] + \
           [frozenset(r * m + c for r in range(m)) for c in range(m)]


def popcount(x):
    x = np.asarray(x, np.int64).copy()
    c = np.zeros(x.shape, np.int64)
    while x.any():
        c += x & 1
        x >>= 1
    return c


def check(name, terms, n, out):
    w = len(next(iter(terms)))
    assert all(len(t) == w for t in terms)
    assert all(len(s & t) <= 1 for s in terms for t in terms if s != t)
    L, a = w - 1, 2.0 ** -(w - 1)
    D = max(sum(1 for t in terms if j in t) for j in range(n))
    x = np.arange(1 << n, dtype=np.int64)
    masks = [sum(1 << j for j in t) for t in terms]
    fire = [(x & m) == m for m in masks]
    F = np.zeros(1 << n, bool)
    for fi in fire:
        F |= fi
    fhat = wht(1.0 - 2.0 * F)
    worst = None
    for i in range(n):
        inc = [t for t in terms if i in t]
        d = len(inc)
        if d == 0:
            continue
        theta = 1 - L * (D - 1) * a - (d - 1) * a / 2
        if not (theta > 0 and d * a <= 0.25):
            continue
        idx = np.arange(1 << n, dtype=np.int64)
        lo = idx[(idx >> i) & 1 == 0]
        p, q = fhat[lo] ** 2, fhat[lo | (1 << i)] ** 2
        delta = q.sum()
        E = Phi(p, q).sum()
        N = np.ones(1 << n, bool)
        for t, fi in zip(terms, fire):
            if i not in t:
                N &= ~fi
        A = N.mean()
        Nhat2 = wht(N.astype(float))[lo] ** 2
        r = delta / A
        ok1 = d * a * theta - 1e-12 <= r <= d * a + 1e-12
        mate_masks = [sum(1 << j for j in t if j != i) for t in inc]
        cnt = np.array([popcount(lo & mm) for mm in mate_masks])   # mates of each incident term in T
        for k in range(1, L + 1):
            if not k * (D - 1) * a < 2:
                continue
            gam = a * sum(math.comb(L, j) for j in range(k))
            eps = min(1.0, d * gam)
            alpha = math.comb(L, k) * 2.0 ** (math.comb(k, 2) - 2 * k) * ((D - 1) * a) ** k / (1 - k * (D - 1) * a / 2)
            kappa = alpha / (a * theta)
            inO = (cnt >= k).any(axis=0)
            h_out = q[~inO].sum()
            N_in = Nhat2[inO].sum()
            ok2 = h_out <= eps * delta + 1e-12
            ok3 = N_in <= kappa * delta + 1e-12
            rhs = 2 * eps + (8 / LN2) * math.sqrt(kappa) + d * a * math.log2(math.e * (1 + (d * a) ** -2))
            if eps > 0:
                rhs += eps * math.log2(math.e * (3 + 8 / (eps * d * a * theta)))
            dev = abs(E / delta - 2)
            ok4 = dev <= rhs + 1e-12
            row = (name, i, d, D, k, E / delta, rhs, h_out / delta, eps, N_in / delta, kappa, ok1 and ok2 and ok3 and ok4)
            out.append(row)
            if not row[-1]:
                print("FAIL", row, flush=True)
            if worst is None or dev / rhs > worst[0]:
                worst = (dev / rhs, row)
    return worst


def main():
    rng = np.random.default_rng(11)
    out = []
    cases = [("star D=2 w=4", star(2, 4), 7), ("star D=3 w=5", star(3, 5), 13), ("star D=4 w=5", star(4, 5), 17), ("star D=4 w=4", star(4, 4), 13),
             ("grid 4x4", grid(4), 16)]
    for (n, w, D) in [(16, 4, 2), (18, 4, 3), (18, 5, 3), (18, 5, 4), (17, 4, 3), (18, 5, 2)]:
        for rep in range(2):
            cases.append((f"random n={n} w={w} D={D} #{rep}", random_linear(n, w, D, 400, rng), n))
    for name, terms, n in cases:
        wst = check(name, terms, n, out)
        if wst:
            r = wst[1]
            print(f"{name:28s} terms {len(terms):3d}  worst dev/rhs {wst[0]:.4f} (i={r[1]} d={r[2]} k={r[4]}: "
                  f"E/Inf {r[5]:.4f}, rhs {r[6]:.3f}; h-out {r[7]:.4f} <= {r[8]:.4f}; N-in {r[9]:.4f} <= {r[10]:.4f})",
                  flush=True)
        else:
            print(f"{name:28s} terms {len(terms):3d}  no coordinate meets the hypotheses", flush=True)
    bad = [r for r in out if not r[-1]]
    print(f"checked {len(out)} (coordinate, k) pairs; failures {len(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
