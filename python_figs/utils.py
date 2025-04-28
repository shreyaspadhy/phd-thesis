import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def bivariate_gaussian_pdf(X, Y, mu, Sigma):
    """
    Compute the PDF of a bivariate normal distribution on a grid.
    X and Y are 2D arrays created by np.meshgrid.
    mu is a 1D array-like (length 2).
    Sigma is a 2x2 covariance matrix.
    Returns a 2D array of the same shape as X and Y.
    """
    # Stack X, Y so that pos = [ [x0, y0], [x1, y1], ... ] for all grid points
    pos = np.dstack((X, Y))

    # Invert covariance matrix and compute determinant
    Sigma_inv = np.linalg.inv(Sigma)
    det_Sigma = np.linalg.det(Sigma)

    # Center the grid by subtracting the mean
    centered = pos - mu

    # Mahalanobis term
    exponent = np.einsum("...k,kl,...l->...", centered, Sigma_inv, centered)

    # PDF expression
    denom = 2 * np.pi * np.sqrt(det_Sigma)
    pdf = np.exp(-0.5 * exponent) / denom
    return pdf


def plot_bivariate_gaussian(
    ax, rho, title, x_range=(-3, 3), y_range=(-3, 3), grid_size=100
):
    """
    Plot a 3D surface of the bivariate normal distribution with correlation rho
    on the given Axes3D object (ax).
    """
    # Define mean and covariance (sigma_x = sigma_y = 1)
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, rho], [rho, 1.0]])

    # Create a grid
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)

    # Compute PDF on the grid
    Z = bivariate_gaussian_pdf(X, Y, mu, Sigma)

    # Plot the surface
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cm.Greys,
        # edgecolor="none",
        alpha=1.0,
        # antialiased=True,
        rcount=200,
        ccount=200,
        shade=True,
    )

    # Adjust view angle: feel free to tweak
    ax.view_init(elev=45, azim=-45)

    # Remove axis panes for a cleaner look
    ax.set_box_aspect((1, 1, 0.5))  # Slight flattening in z for aesthetics
    ax.grid(False)


def plot_gaussian_mixture(
    ax,
    components,
    weights=None,
    title="Gaussian Mixture",
    x_range=(-3, 3),
    y_range=(-3, 3),
    grid_size=100,
):
    """
    Plot a 3D surface of a mixture of bivariate Gaussian distributions.

    Parameters:
    -----------
    ax : Axes3D object
        The 3D axes to plot on
    components : list of tuples
        Each tuple contains (mu, Sigma) where:
        - mu is a 1D array-like of length 2 (mean vector)
        - Sigma is a 2x2 covariance matrix
    weights : list or array, optional
        Mixture weights for each component. If None, equal weights are used.
    title : str, optional
        Title for the plot
    x_range, y_range : tuple, optional
        Range for x and y axes
    grid_size : int, optional
        Number of points in each dimension of the grid
    cmap : matplotlib colormap, optional
        Colormap for the surface
    """
    # Create a grid
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)

    # If weights not provided, use equal weights
    if weights is None:
        weights = np.ones(len(components)) / len(components)

    # Compute the mixture PDF
    Z = np.zeros_like(X)
    for (mu, Sigma), weight in zip(components, weights):
        Z += weight * bivariate_gaussian_pdf(X, Y, mu, Sigma)

    from matplotlib.colors import LightSource

    ls = LightSource(0, 90)  # Head-on light source (azimuth=0, elevation=90)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    # rgb = ls.shade(Z, cmap=cm.Grays, vert_exag=0.1, blend_mode='soft')
    # Plot the surface
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.Greys(norm(Z))
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        # rstride=1,
        # cstride=1,
        cmap=cm.Greys,
        linewidth=0,
        edgecolor="none",
        alpha=1.0,
        antialiased=False,
        rcount=10000,
        ccount=10000,
        shade=True,
        # facecolors=rgb
    )

    # surf2 = ax.plot_surface(
    #     X,
    #     Y,
    #     Z,
    #     # rstride=1,
    #     # cstride=1,
    #     # cmap=cm.Greys,
    #     # linewidth=0,
    #     # edgecolor="none",
    #     # alpha=1.0,
    #     # antialiased=False,
    #     rcount=100,
    #     ccount=100,
    #     shade=False,
    #     facecolors=colors
    # )
    # surf2.set_facecolor((0,0,0,0))

    # Adjust view angle
    ax.view_init(elev=45, azim=-45)

    # Remove axis panes for a cleaner look
    ax.set_box_aspect((1, 1, 0.5))  # Slight flattening in z for aesthetics
    ax.grid(False)

    return surf
