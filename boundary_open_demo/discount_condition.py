"""The sufficient condition on the discount, computed on the demonstration grid.

Eq. (5) holds for an undiscounted return. At a discount the shaping term of Eq. (4), which is the
UNDISCOUNTED difference of potentials, no longer telescopes to a constant. Abel summation over a
completing route of k transitions from origin o to destination g, with Phi(g)=0, gives

    G_da(p) - G_tt(p) = beta*Phi(o) + D(p)
    D(p) = gamma^{k-1} R_g - beta (1-gamma) sum_{t=1}^{k-1} gamma^{t-1} Phi(n_t)

beta*Phi(o) is constant over routes from o, D(p) is not. The travel-time order is preserved
whenever the spread of D over the completing routes is smaller than the travel-time return gap
between the optimum and the next best. Using 1 - gamma^m <= m(1-gamma) on both terms:

    sup |D(p) - D(p')|  <=  (1-gamma) [ (K - k_min) R_g + (K-1) beta Phi_max ]

so a sufficient condition is

    gamma  >  1 - Delta / [ (K - k_min) R_g + (K-1) beta Phi_max ]

which tends to 1 - Delta/inf = the undiscounted case as gamma -> 1, recovering Eq. (6).
"""
import itertools, sys

N = 5                      # env.N_SIDE
C_MIN, C_MAX = 1.0, 1.6    # traversal cost range at amplitude 0.6
BETA, R_G = 1.0, 10.0      # Section V-A settings

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

nodes = [(r, c) for r in range(N) for c in range(N)]

# Phi is the Manhattan distance to the destination. Its maximum over the grid depends on where the
# destination sits; the condition must hold for the worst destination.
phi_max = max(manhattan(n, g) for g in nodes for n in nodes)

# The evaluation set draws destinations at graph distances 3 to 8 (Section V-A).
K_MIN, K = 3, 8

# The travel-time return gap. On a four-neighbour grid every completing route has the parity of the
# Manhattan distance, so the next-best completing route is at least two transitions longer and
# costs at least 2*c_min more.
DELTA = 2.0 * C_MIN

# The bound has to cover every route the comparison admits, not the optimal ones alone. The route
# defining DELTA runs two transitions past the longest completing route of the evaluation set, so
# the longest route entering the comparison is K + 2. The derivation reads "across the routes the
# evaluation set admits" and K was substituted at the optimal length only.
K_CMP = K + 2

den = (K_CMP - K_MIN) * R_G + (K_CMP - 1) * BETA * phi_max
gamma_star = 1.0 - DELTA / den

print("  Phi_max (worst destination on the %dx%d grid) = %d" % (N, N, phi_max))
print("  k in [%d, %d] and the comparison reaches %d, beta = %.1f, R_g = %.1f, c_min = %.1f" % (K_MIN, K, K_CMP, BETA, R_G, C_MIN))
print("  Delta (two extra transitions at c_min)        = %.1f" % DELTA)
print("  denominator (K-k_min)R_g + (K-1) beta Phi_max = %.1f" % den)
print("  gamma*  = 1 - Delta/denominator               = %.4f" % gamma_star)
print()
print("  the learner of Section V-A uses gamma = 0.99")
print("  condition satisfied: %s" % (0.99 > gamma_star))
print("  also satisfied at the 0.90 reported in the supplement: %s" % (0.90 > gamma_star))
