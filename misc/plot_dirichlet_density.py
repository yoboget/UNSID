import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from math import lgamma
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import get_cmap

def plot_dirichlet_simplex(alpha1, alpha2, alpha3, resolution=200, cmap="viridis",
                           ax=None, show_colorbar=True, annotate=True, relative=False,
                           blackout_voronoi=True, blackout_alpha=0.5, vmin=None, vmax=None):
    """
    Plot the Dirichlet(alpha1, alpha2, alpha3) density as a color map on the 2-simplex.

    Parameters
    ----------
    alpha1, alpha2, alpha3 : float
        Positive parameters of the Dirichlet distribution.
    resolution : int, optional
        Controls the sampling density on the simplex. Larger -> smoother, slower.
    cmap : str or Colormap, optional
        Matplotlib colormap to use.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates a new figure and axes.
    show_colorbar : bool, optional
        Whether to add a colorbar.
    annotate : bool, optional
        Whether to label the triangle corners.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plots.
    """
    a = np.array([alpha1, alpha2, alpha3], dtype=float)
    if np.any(a <= 0):
        raise ValueError("Dirichlet parameters must be strictly positive.")

    # Equilateral triangle vertices for (1,0,0), (0,1,0), (0,0,1)
    v1 = np.array([0.0, 0.0])                 # corresponds to x1=1
    v2 = np.array([1.0, 0.0])                 # corresponds to x2=1
    v3 = np.array([0.5, np.sqrt(3.0)/2.0])    # corresponds to x3=1

    # Sample a uniform grid on the simplex in barycentric coordinates
    pts = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            x1 = i / resolution
            x2 = j / resolution
            x3 = 1.0 - x1 - x2
            pts.append((x1, x2, x3))
    P = np.array(pts)  # shape (N, 3)

    # Map barycentric coordinates to 2D coordinates
    XY = P[:, 0, None] * v1 + P[:, 1, None] * v2 + P[:, 2, None] * v3
    x, y = XY[:, 0], XY[:, 1]

    # Compute log-density up to a constant and then exponentiate in a stable way
    # log pdf = lgamma(sum a) - sum lgamma(a_i) + sum (a_i - 1) * log x_i
    const = lgamma(a.sum()) - np.sum([lgamma(ai) for ai in a])
    logP = np.where(P > 0.0, np.log(P), -np.inf)
    logpdf = const + ( (a - 1.0) * logP ).sum(axis=1)

    # Stabilize: scale so that max density = 1 (good for plotting and avoids overflow)

    if relative:
        finite_mask = np.isfinite(logpdf)
        if not np.any(finite_mask):
            raise RuntimeError("Numerical issue: all evaluated log-densities are non-finite.")
        max_logpdf = np.max(logpdf[finite_mask])
        z = np.exp(logpdf - max_logpdf)
        z[~finite_mask] = 0.0  # points on edges with zero coordinates when alpha<1
    else:
        pdf = np.zeros_like(logpdf)
        finite_mask = np.isfinite(logpdf)
        pdf[finite_mask] = np.exp(logpdf[finite_mask])
        z=np.exp(logpdf-const)


        vmin_auto = np.min(pdf) if vmin is None else float(vmin)
        vmax_auto = np.max(pdf) if vmax is None else float(vmax)
        norm = Normalize(vmin=vmin_auto, vmax=vmax_auto)
        z_for_plot = pdf
        z = pdf


    # Triangulate and color
    tri = mtri.Triangulation(x, y)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
    tpc = ax.tripcolor(tri, z, shading="gouraud", cmap=cmap, norm=norm)

    if blackout_voronoi:
        m12 = 0.5 * (v1 + v2)  # midpoint of edge (v1,v2)
        m13 = 0.5 * (v1 + v3)  # midpoint of edge (v1,v3)
        center = (v1 + v2 + v3) / 3.0  # centroid = circumcenter for equilateral
        poly = np.vstack([v1, m12, center, m13])
        ax.fill(poly[:, 0], poly[:, 1], color="w", alpha=blackout_alpha, zorder=6)

    # Draw the triangle boundary
    ax.plot([v1[0], v2[0], v3[0], v1[0]],
            [v1[1], v2[1], v3[1], v1[1]], color="k", lw=1)

    # Labels and cosmetics
    if annotate:
        ax.text(v1[0] - 0.02, v1[1] - 0.02, "(1,0,0)", ha="right", va="top", fontsize=9)
        ax.text(v2[0] + 0.02, v2[1] - 0.02, "(0,1,0)", ha="left", va="top", fontsize=9)
        ax.text(v3[0], v3[1] + 0.02, "(0,0,1)", ha="center", va="bottom", fontsize=9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3)/2 + 0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Dirichlet({alpha1:g}, {alpha2:g}, {alpha3:g})\n")

    if show_colorbar:
        cbar = plt.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
        if relative:
            cbar.set_label("Relative density")
        else:
            cbar.set_label("Density")
    plt.tight_layout()
    ax.set_axis_off()

    return ax

def _colors_from_cmap(cmap="viridis", positions=None):
    """
    Return three RGBA colors sampled from a Matplotlib colormap.

    Parameters
    ----------
    cmap : str or Colormap
        Colormap name or object (e.g., "viridis", "plasma", etc.).
    positions : iterable of 3 floats in [0,1], optional
        Where to sample colors along the colormap. If None, uses evenly
        spaced positions chosen to maximize contrast: [0.15, 0.55, 0.9].
    """
    cm = get_cmap(cmap)
    if positions is None:
        positions = (0.15, 0.55, 0.90)
    if len(positions) != 3:
        raise ValueError("positions must contain exactly 3 values in [0, 1].")
    return [cm(float(p)) for p in positions]


def _colors_from_cmap(cmap="viridis", positions=None):
    cm = get_cmap(cmap)
    if positions is None:
        positions = (0.15, 0.55, 0.90)
    if len(positions) != 3:
        raise ValueError("positions must contain exactly 3 values in [0, 1].")
    return [cm(float(p)) for p in positions]

def plot_dirichlet_corner_samples(
    n, a, b, c, *,
    ax=None, seed=None, s=14, alpha=0.85,
    palette="viridis", color_positions=None, colors=None,
    annotate=False, show_triangle=True, title=False,
    draw_voronoi=True, voronoi_lw=1.5, voronoi_alpha=1.0
):
    """
    Scatter n samples from Dir(a,1,1), Dir(1,b,1), Dir(1,1,c) on the 2-simplex,
    color the three groups with colors sampled from `palette`, and (optionally)
    draw the Voronoi region borders for each vertex in the corresponding color.

    Parameters
    ----------
    n : int
        Number of samples per distribution.
    a, b, c : float
        Positive Dirichlet parameters for the emphasized corners.
    palette : str or Colormap
        Colormap to extract three colors from (e.g., "viridis").
    color_positions : 3-tuple of floats in [0,1], optional
        Where to sample colors along the colormap. Defaults to (0.15, 0.55, 0.90).
    colors : iterable of 3 Matplotlib colors, optional
        If given, overrides palette sampling.
    draw_voronoi : bool
        If True, draw the Voronoi borders of the three vertices.
    voronoi_lw : float
        Line width for Voronoi borders.
    voronoi_alpha : float
        Alpha for Voronoi borders.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    for name, val in zip(("a","b","c"), (a,b,c)):
        if val <= 0:
            raise ValueError(f"Parameter {name} must be > 0.")

    # Equilateral triangle embedding
    v1 = np.array([0.0, 0.0])                 # (1,0,0)
    v2 = np.array([1.0, 0.0])                 # (0,1,0)
    v3 = np.array([0.5, np.sqrt(3.0)/2.0])    # (0,0,1)

    # Helpers
    def bary_to_xy(P):
        return P[:, [0]] * v1 + P[:, [1]] * v2 + P[:, [2]] * v3

    # Samples
    rng = np.random.default_rng(seed)
    X1 = rng.dirichlet([a, 1.0, 1.0], size=n)
    X2 = rng.dirichlet([1.0, b, 1.0], size=n)
    X3 = rng.dirichlet([1.0, 1.0, c], size=n)
    XY1, XY2, XY3 = bary_to_xy(X1), bary_to_xy(X2), bary_to_xy(X3)

    # Colors
    if colors is None:
        c1, c2, c3 = _colors_from_cmap(palette, color_positions)
    else:
        if len(colors) != 3:
            raise ValueError("colors must be a sequence of exactly 3 colors.")
        c1, c2, c3 = colors

    # Axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 5.0))

    # Scatter points
    ax.scatter(XY1[:,0], XY1[:,1], s=s, alpha=alpha, label=f"Dir({a:g},1,1)",
               color=c1, edgecolors="none")
    ax.scatter(XY2[:,0], XY2[:,1], s=s, alpha=alpha, label=f"Dir(1,{b:g},1)",
               color=c2, edgecolors="none")
    ax.scatter(XY3[:,0], XY3[:,1], s=s, alpha=alpha, label=f"Dir(1,1,{c:g})",
               color=c3, edgecolors="none")

    # Triangle border (optional)
    if show_triangle:
        ax.plot([v1[0], v2[0], v3[0], v1[0]],
                [v1[1], v2[1], v3[1], v1[1]],
                color="k", lw=1.1, zorder=5)

    # Voronoi borders (colored)
    if draw_voronoi:
        center = (v1 + v2 + v3) / 3.0             # centroid = circumcenter (equilateral)
        m12 = 0.5 * (v1 + v2)
        m23 = 0.5 * (v2 + v3)
        m13 = 0.5 * (v1 + v3)

        # Region polygons (each: vertex -> adjacent midpoint -> center -> other midpoint -> back)
        poly1 = np.vstack([v1, m12, center, m13, v1])
        poly2 = np.vstack([v2, m23, center, m12, v2])
        poly3 = np.vstack([v3, m13, center, m23, v3])

        # Draw outlines in corresponding colors
        ax.plot(poly1[:,0], poly1[:,1], color=c1, lw=voronoi_lw,
                alpha=voronoi_alpha, zorder=6, solid_joinstyle="round")
        ax.plot(poly2[:,0], poly2[:,1], color=c2, lw=voronoi_lw,
                alpha=voronoi_alpha, zorder=6, solid_joinstyle="round")
        ax.plot(poly3[:,0], poly3[:,1], color=c3, lw=voronoi_lw,
                alpha=voronoi_alpha, zorder=6, solid_joinstyle="round")

    # Cosmetics
    if annotate:
        ax.text(v1[0]-0.02, v1[1]-0.02, "(1,0,0)", ha="right", va="top", fontsize=9)
        ax.text(v2[0]+0.02, v2[1]-0.02, "(0,1,0)", ha="left", va="top", fontsize=9)
        ax.text(v3[0],      v3[1]+0.02, "(0,0,1)", ha="center", va="bottom", fontsize=9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3)/2 + 0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()

    if title:
        ax.set_title(f"{n} samples each from Dir({a:g},1,1), Dir(1,{b:g},1), Dir(1,1,{c:g})")
    # ax.legend(loc="upper right", frameon=False, handlelength=1.2, fontsize=9)
    plt.tight_layout()
    return ax


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap

# def _colors_from_cmap(cmap="viridis", positions=None):
#     """
#     Return three RGBA colors sampled from a Matplotlib colormap.
#     positions defaults to (0.15, 0.55, 0.90) for good contrast.
#     """
#     cm = get_cmap(cmap)
#     if positions is None:
#         positions = (0.15, 0.55, 0.90)
#     if len(positions) != 3:
#         raise ValueError("positions must contain exactly 3 values in [0, 1].")
#     return [cm(float(p)) for p in positions]

def plot_vertex_mixture_samples(
    n, a, *,                      # <-- n per vertex, mixing weight a
    ax=None, seed=None, s=14, alpha=0.85,
    palette="viridis", color_positions=None, colors=None,
    annotate=False, show_triangle=True, title=False,
    draw_voronoi=True, voronoi_lw=1.5, voronoi_alpha=1.0
):
    """
    Scatter n samples per vertex of y = a * x1 + (1-a) * x0 on the 2-simplex,
    where x1 is a vertex ((1,0,0), (0,1,0), (0,0,1)) and x0 ~ Dir(1,1,1).

    Parameters
    ----------
    n : int
        Number of samples per vertex (total points = 3n).
    a : float in [0, 1]
        Mixing weight toward the vertex. a=1 -> points at vertices; a=0 -> uniform Dir(1,1,1).
    ax : matplotlib.axes.Axes or None
        Axes to draw on; if None, a new figure/axes are created.
    seed : int or None
        RNG seed for reproducibility.
    s : float
        Marker size for scatter points.
    alpha : float
        Marker transparency.
    palette : str or Colormap
        Base colormap to draw the three colors from (e.g., "viridis").
    color_positions : tuple/list of 3 floats in [0,1] or None
        Where to sample the three colors from `palette`. Defaults to (0.15, 0.55, 0.90).
    colors : list/tuple of 3 Matplotlib colors, optional
        If provided, overrides `palette`/`color_positions`.
    annotate, show_triangle, title : bool
        Usual plots cosmetics.
    draw_voronoi : bool
        If True, draw the Voronoi region borders of the three vertices in the three colors.
    voronoi_lw : float
        Line width for Voronoi borders.
    voronoi_alpha : float
        Alpha for Voronoi borders.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if not (0.0 <= a <= 1.0):
        raise ValueError("a must be in [0, 1] to keep points inside the simplex.")

    # Equilateral triangle embedding
    v1 = np.array([0.0, 0.0])                 # (1,0,0)
    v2 = np.array([1.0, 0.0])                 # (0,1,0)
    v3 = np.array([0.5, np.sqrt(3.0)/2.0])    # (0,0,1)

    # Convert barycentric -> 2D
    def bary_to_xy(P):
        return P[:, [0]] * v1 + P[:, [1]] * v2 + P[:, [2]] * v3

    # Unit vectors for vertices in barycentric coords
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])

    # Random base points x0 ~ Dir(1,1,1)
    rng = np.random.default_rng(seed)
    X0_1 = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    X0_2 = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    X0_3 = rng.dirichlet([1.0, 1.0, 1.0], size=n)

    # Mixtures toward each vertex: y = a*e_i + (1-a)*x0
    Y1 = a * e1 + (1.0 - a) * X0_1
    Y2 = a * e2 + (1.0 - a) * X0_2
    Y3 = a * e3 + (1.0 - a) * X0_3

    XY1, XY2, XY3 = bary_to_xy(Y1), bary_to_xy(Y2), bary_to_xy(Y3)

    # Colors
    if colors is None:
        c1, c2, c3 = _colors_from_cmap(palette, color_positions)
    else:
        if len(colors) != 3:
            raise ValueError("colors must be a sequence of exactly 3 colors.")
        c1, c2, c3 = colors

    # Axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 5.0))

    # Scatter
    ax.scatter(XY1[:,0], XY1[:,1], s=s, alpha=alpha,
               label=f"a·(1,0,0) + (1−a)·Dir(1,1,1)", color=c1, edgecolors="none")
    ax.scatter(XY2[:,0], XY2[:,1], s=s, alpha=alpha,
               label=f"a·(0,1,0) + (1−a)·Dir(1,1,1)", color=c2, edgecolors="none")
    ax.scatter(XY3[:,0], XY3[:,1], s=s, alpha=alpha,
               label=f"a·(0,0,1) + (1−a)·Dir(1,1,1)", color=c3, edgecolors="none")

    # Triangle border
    if show_triangle:
        ax.plot([v1[0], v2[0], v3[0], v1[0]],
                [v1[1], v2[1], v3[1], v1[1]],
                color="k", lw=1.1, zorder=5)

    # Voronoi borders in corresponding colors
    if draw_voronoi:
        center = (v1 + v2 + v3) / 3.0
        m12 = 0.5 * (v1 + v2)
        m23 = 0.5 * (v2 + v3)
        m13 = 0.5 * (v1 + v3)
        poly1 = np.vstack([v1, m12, center, m13, v1])
        poly2 = np.vstack([v2, m23, center, m12, v2])
        poly3 = np.vstack([v3, m13, center, m23, v3])
        ax.plot(poly1[:,0], poly1[:,1], color=c1, lw=voronoi_lw,
                alpha=voronoi_alpha, zorder=6, solid_joinstyle="round")
        ax.plot(poly2[:,0], poly2[:,1], color=c2, lw=voronoi_lw,
                alpha=voronoi_alpha, zorder=6, solid_joinstyle="round")
        ax.plot(poly3[:,0], poly3[:,1], color=c3, lw=voronoi_lw,
                alpha=voronoi_alpha, zorder=6, solid_joinstyle="round")

    # Cosmetics
    if annotate:
        ax.text(v1[0]-0.02, v1[1]-0.02, "(1,0,0)", ha="right", va="top", fontsize=9)
        ax.text(v2[0]+0.02, v2[1]-0.02, "(0,1,0)", ha="left", va="top", fontsize=9)
        ax.text(v3[0],      v3[1]+0.02, "(0,0,1)", ha="center", va="bottom", fontsize=9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3)/2 + 0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()

    if title:
        ax.set_title(f"{n} samples per vertex of a·vertex + (1−a)·Dir(1,1,1),  a={a:g}")
    # ax.legend(loc="upper right", frameon=False, handlelength=1.2, fontsize=9)
    plt.tight_layout()
    return ax

def _gumbel_softmax_sample(logits, temperature, shape, rng):
    """
    Draws samples from a Gumbel-Softmax distribution.
    `logits` should be a 1D array of log-probabilities.
    """
    gumbel_noise = rng.gumbel(loc=0.0, scale=1.0, size=shape + logits.shape)
    y = gumbel_noise + logits
    return np.exp(y) / np.sum(np.exp(y), axis=-1, keepdims=True)

def plot_gumbel_softmax_samples(
    n, t, temperature, stationary_dist, *,
    ax=None, seed=None, s=14, alpha=0.85,
    palette="viridis", color_positions=None, colors=None,
    annotate=False, show_triangle=True, title=True,
    draw_voronoi=True, voronoi_lw=1.5, voronoi_alpha=1.0
):
    """
    Scatter n samples per vertex using a Gumbel-Softmax noising process.

    The process interpolates from a vertex `x_0` to a stationary distribution `pi_inf`
    based on a time parameter `t`.

    Parameters
    ----------
    n : int
        Number of samples per vertex (total points = 3n).
    t : float in [0, 1]
        Time parameter. t=0 is clean, t=1 is stationary.
    temperature : float
        Gumbel-Softmax temperature. Lower values are "harder" (closer to vertices).
    stationary_dist : list or np.ndarray
        The stationary distribution, e.g., [1/3, 1/3, 1/3].
    ax, seed, s, alpha, palette, etc. :
        Standard plotting and cosmetic parameters.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if not (0.0 <= t <= 1.0):
        raise ValueError("t must be in [0, 1].")
    pi_inf = np.array(stationary_dist, dtype=float)
    if not np.isclose(pi_inf.sum(), 1.0) or np.any(pi_inf < 0):
        raise ValueError("stationary_dist must be a valid probability distribution.")

    # Equilateral triangle embedding
    v1 = np.array([0.0, 0.0])                 # (1,0,0)
    v2 = np.array([1.0, 0.0])                 # (0,1,0)
    v3 = np.array([0.5, np.sqrt(3.0)/2.0])    # (0,0,1)

    def bary_to_xy(P):
        return P[:, [0]] * v1 + P[:, [1]] * v2 + P[:, [2]] * v3

    # Vertices in barycentric coords
    e1, e2, e3 = np.eye(3)

    # Interpolate probabilities
    pi1 = (1 - t) * e1 + t * pi_inf
    pi2 = (1 - t) * e2 + t * pi_inf
    pi3 = (1 - t) * e3 + t * pi_inf

    # Sample using Gumbel-Softmax
    rng = np.random.default_rng(seed)
    Y1 = _gumbel_softmax_sample(np.log(pi1), temperature, shape=(n,), rng=rng)
    Y2 = _gumbel_softmax_sample(np.log(pi2), temperature, shape=(n,), rng=rng)
    Y3 = _gumbel_softmax_sample(np.log(pi3), temperature, shape=(n,), rng=rng)

    XY1, XY2, XY3 = bary_to_xy(Y1), bary_to_xy(Y2), bary_to_xy(Y3)

    # --- Plotting (similar to other functions) ---
    if colors is None:
        c1, c2, c3 = _colors_from_cmap(palette, color_positions)
    else:
        c1, c2, c3 = colors

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 5.0))

    ax.scatter(XY1[:,0], XY1[:,1], s=s, alpha=alpha, color=c1, edgecolors="none")
    ax.scatter(XY2[:,0], XY2[:,1], s=s, alpha=alpha, color=c2, edgecolors="none")
    ax.scatter(XY3[:,0], XY3[:,1], s=s, alpha=alpha, color=c3, edgecolors="none")

    if show_triangle:
        ax.plot([v1[0], v2[0], v3[0], v1[0]], [v1[1], v2[1], v3[1], v1[1]], color="k", lw=1.1, zorder=5)

    if draw_voronoi:
        center = (v1 + v2 + v3) / 3.0
        m12, m23, m13 = 0.5 * (v1 + v2), 0.5 * (v2 + v3), 0.5 * (v1 + v3)
        ax.plot(np.vstack([v1, m12, center, m13, v1])[:,0], np.vstack([v1, m12, center, m13, v1])[:,1], color=c1, lw=voronoi_lw, alpha=voronoi_alpha, zorder=6)
        ax.plot(np.vstack([v2, m23, center, m12, v2])[:,0], np.vstack([v2, m23, center, m12, v2])[:,1], color=c2, lw=voronoi_lw, alpha=voronoi_alpha, zorder=6)
        ax.plot(np.vstack([v3, m13, center, m23, v3])[:,0], np.vstack([v3, m13, center, m23, v3])[:,1], color=c3, lw=voronoi_lw, alpha=voronoi_alpha, zorder=6)

    if annotate:
        ax.text(v1[0]-0.02, v1[1]-0.02, "e₁", ha="right", va="top", fontsize=11)
        ax.text(v2[0]+0.02, v2[1]-0.02, "e₂", ha="left", va="top", fontsize=11)
        ax.text(v3[0],      v3[1]+0.02, "e₃", ha="center", va="bottom", fontsize=11)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, np.sqrt(3)/2 + 0.05)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_axis_off()

    if title:
        ax.set_title(f"Gumbel-Softmax (t={t:g}, τ={temperature:g}), π∞={list(np.round(pi_inf,2))}")
    plt.tight_layout()
    return ax

# 400 samples per distribution, reproducible:
a = -np.log(1.0 - 0.0) * 3 + 1
# ax = plot_dirichlet_corner_samples(1000, a, a, a, seed=1,
#                                    palette="inferno",
#                                    color_positions=(0.15, 0.5, 0.85))

# ax = plot_vertex_mixture_samples(1000, a=0.5, seed=7,
#                                  palette="inferno",
#                                  color_positions=(0.15, 0.5, 0.85),
#                                  voronoi_lw=1.5)

ax = plot_gumbel_softmax_samples(n=1000, t=0.5, temperature=1,
                                 stationary_dist=[5/12, 3/12, 4/12],
                                 seed=42, palette="inferno")
plt.show()
# ax = plot_dirichlet_simplex(5, 1.0, 1.0, resolution=250, cmap="inferno",
#                             show_colorbar=True, annotate=False, blackout_alpha=0.7, vmin=0, vmax=6)
# plt.show()