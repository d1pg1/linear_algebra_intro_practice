import numpy as np

def negative_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the negation of each element in the input vector or matrix.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with each element negated.
    """
    raise -x


def reverse_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the input vector or matrix with the order of elements reversed.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with the order of elements reversed.
    """
    raise np.flip(x)


def affine_transform(
    x: np.ndarray, alpha_deg: float, scale: tuple[float, float], shear: tuple[float, float],
    translate: tuple[float, float],
) -> np.ndarray:
    """Compute affine transformation

    Args:
        x (np.ndarray): vector n*1 or matrix n*n.
        alpha_deg (float): rotation angle in deg.
        scale (tuple[float, float]): x, y scale factor.
        shear (tuple[float, float]): x, y shear factor.
        translate (tuple[float, float]): x, y translation factor.

    Returns:
        np.ndarray: transformed matrix.
    """
    sx, sy = scale
    shx, shy = shear

    theta = np.deg2rad(alpha_deg)
    c, s = np.cos(theta), np.sin(theta)

    S = np.array([[sx, 0.0],
                  [0.0, sy]], dtype=float)
    Sh = np.array([[1.0, shx],
                   [shy, 1.0]], dtype=float)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)

    A = R @ Sh @ S
    t = np.array(translate, dtype=float)

    return (A @ x.astype(float)) + t
