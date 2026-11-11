# Erasure entropies of sparse linear DNFs

Yitzchak Shmalo

This repository contains a paper on the Fourier entropy-influence (FEI) conjecture of Friedgut and Kalai, together
with the scripts behind every numerical value in it.

Paper: [main.pdf](main.pdf) (source [main.tex](main.tex)).

## Summary

For a Boolean function f and a coordinate i, the erasure entropy E_i(f) is the conditional entropy of the event
"i is in U" given U minus i, where U is drawn from the spectral distribution. FEI is equivalent to every non-constant
f having a coordinate with E_i(f) = O(Inf_i(f)).

* A separation inequality for every monotone f: write h for the indicator that i is pivotal and N for the indicator
  that F vanishes at x_i = 0. If the spectrum of h puts weight at most eps Inf_i outside a family Omega of sets and the
  spectrum of N puts weight at most kappa Inf_i on Omega, then E_i/Inf_i = 2 + O(eps log(1/eps) + sqrt(kappa)) with
  explicit constants.
* For a monotone DNF whose terms all have width w, any two of which share at most one variable, and in which every
  variable lies in at most D terms, an explicit bound on |E_i/Inf_i - 2| at each coordinate, in terms of w, D, the
  number of terms through i and a parameter k. Omega is the family of sets containing at least k of the other
  variables of some term through i.
* Consequently E_i/Inf_i = 2 + o(1) at every coordinate, uniformly in the number of variables and terms, as w grows
  with D at most 2^((1 - eta)(w - 1)), and the sum of the E_i is (2 + o(1)) times the total influence.

## Code

The scripts need Python 3 and NumPy. Their saved outputs are next to them.

* `code/check_lin.py` (output `check_lin_out.txt`): the three lemmas and the main inequality, with exact
  Walsh-Hadamard transforms, at every coordinate and every admissible k of 17 linear DNFs (stars, the rows and columns
  of a 4 x 4 array, and 12 random ones with w = 4, 5 and D = 2, 3, 4): 784 cases.
* `code/width_table.py` (output `width_table_out.txt`): E_i/Inf_i for stars of D terms of width w and for the grid,
  the table of Section 4.

## Related papers

* [The one-coordinate step in the Fourier entropy-influence conjecture](https://github.com/yspennstate/fourier-entropy-influence-one-coordinate)
* [Good coordinates in the Fourier entropy-influence conjecture](https://github.com/yspennstate/fei-good-coordinates)
* [A variance form of Han's bound on the Fourier entropy](https://github.com/yspennstate/fei-variance-han-bound)
* [Four equivalent forms of the Fourier entropy-influence conjecture](https://github.com/yspennstate/fei-equivalent-forms)
* [A filtration bound for the erasure entropies of Boolean functions](https://github.com/yspennstate/fei-filtration-witness)
* [The Fourier entropy of linear monotone DNFs on few variables](https://github.com/yspennstate/fei-linear-dnf-census)

## License

MIT, see [LICENSE](LICENSE).
