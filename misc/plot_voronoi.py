# Implementation of the requested functions, plus a quick demo plots.
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.pyplot import legend


def P_vi(K: int, a: float) -> float:
    """
    Compute P_{v_i}(x) = sum_{k=0}^{K-1} [ (-1)^k * C(n, k) / (k+1)^{alpha_t - 1} ]

    Parameters
    ----------
    n : int
        The 'n' in the binomial coefficient C(n, k).
    K : int
        Number of terms in the sum (upper limit is K-1).
    alpha_t : float
        The alpha_t parameter in the exponent.

    Returns
    -------
    float
        The value of the finite sum.
    """
    total = 0.0
    n = K - 1
    for k in range(K):
        print(k)
        term = ((-1) ** k) * math.comb(n, k) / ((k + 1) ** a)
        total += term
    return total

from math import comb, isclose

def P_vi_inter(K: int, a: float) -> float:
    """
    Probability that X = a*e1 + (1-a)*U (with U ~ Dirichlet(1,...,1) on the K-simplex)
    lies in the Voronoi region of e1 (i.e., X1 >= Xi for all i>=2),
    as a function of scalar a in [0,1].

    Special cases:
      - If a >= 0.5, the probability is 1.
      - If a == 0, the probability is 1/K.
    """
    if K < 2:
        raise ValueError("K must be >= 2.")
    if not (0.0 <= a <= 1.0):
        raise ValueError("a must be in [0, 1].")

    # Handle easy edges & numerical safety
    if a >= 0.5:
        return 1.0
    if a == 0.0:
        return 1.0 / K

    delta = a / (1.0 - a)  # in (0, 1) when a in (0, 0.5)

    s = 0.0
    for r in range(0, K):  # r = 0..K-1
        base = 1.0 - r * delta
        if base <= 0.0 and r > 0:
            # For r>=1, terms become non-positive and stay that way as r increases
            break
        term = ((-1.0)**r) * comb(K-1, r) * (base ** (K-1)) / (r + 1.0)
        s += max(term, 0.0) if base < 0 else term  # max(...) guard is redundant after break, but harmless
    return float(s)



def plot_P_vi_over_alpha(n: int, N: int = 200):
    """
    Plot P_{v_i}(x) as a function of alpha_t for N values between 0 and 1 (inclusive).

    Parameters
    ----------
    n : int
        The 'n' in C(n, k).
    K : int
        Number of terms in the sum (upper limit is K-1).
    N : int, optional
        Number of sample points for alpha_t in [0, 1], by default 200.
    """
    t = np.linspace(0.0, 1.0,  int(N))
    K=7
    for a in [1, 2, 3, 4, 5, 6, 8, 10]:
        alphas = -(np.log(np.array([1.0]) - t) * a)+1  # scale=1.0
        # alphas = (n-2) * alphas

        # values = np.array([P_vi_inter(n, a) for a in t])
        values = np.array([P_vi(n, a) for a in alphas])


        cmap = 'inferno'
        cm = get_cmap(cmap)
        plt.plot(t, values, linewidth=2, color=cm(float(a/(K+4))), label=f"$a={a}$")
    plt.xlabel(r"$t$", size=16)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.legend(prop={'size': 14})
    # plt.ylabel(r"$P_{v_i}$")
    # plt.title(r"Voronoi probability $P_{v_i}$")
        # plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# --- Demo (you can delete this block) ---
# A quick example so you can see the plots right away.
# Feel free to call plot_P_vi_over_alpha with your own n, K, N.
plot_P_vi_over_alpha(n=3, N=300)
