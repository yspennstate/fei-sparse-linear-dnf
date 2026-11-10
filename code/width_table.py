"""E_i/Inf_i at every coordinate type of the star of D terms of width w (all terms share one centre variable,
otherwise disjoint) and of the m x m grid DNF (rows and columns), computed exactly with the Walsh-Hadamard transform."""
import numpy as np
from check_lin import wht, Phi, star, grid


def ratios(terms, n, coords):
    x = np.arange(1 << n, dtype=np.int64)
    F = np.zeros(1 << n, bool)
    for t in terms:
        m = sum(1 << j for j in t)
        F |= (x & m) == m
    fhat2 = wht(1.0 - 2.0 * F) ** 2
    out = []
    for i in coords:
        lo = x[(x >> i) & 1 == 0]
        p, q = fhat2[lo], fhat2[lo | (1 << i)]
        out.append((q.sum(), Phi(p, q).sum() / q.sum()))
    return out


def main():
    print("star: D terms of width w through a centre variable")
    print(f"{'D':>2} {'w':>3} {'n':>3}   {'Inf centre':>10} {'E/Inf centre':>12}   {'Inf leaf':>10} {'E/Inf leaf':>10}")
    for D, ws in [(2, range(3, 11)), (3, range(3, 8)), (4, range(3, 6))]:
        for w in ws:
            n = 1 + D * (w - 1)
            (ic, rc), (il, rl) = ratios(star(D, w), n, [0, 1])
            print(f"{D:>2} {w:>3} {n:>3}   {ic:10.6f} {rc:12.4f}   {il:10.6f} {rl:10.4f}", flush=True)
    print()
    print("grid: the m rows and m columns of an m x m array (width m, every variable in two terms)")
    for m in [3, 4]:
        (inf, r), = ratios(grid(m), m * m, [0])
        print(f"m={m}  Inf {inf:.6f}  E/Inf {r:.4f}")


if __name__ == "__main__":
    main()
