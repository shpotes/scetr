from typing import Tuple
from einops import rearrange
import torch

def corner_with_angle_to_points(bbox: torch.Tensor) -> Tuple[torch.Tensor]:
    """
    Compute the corners of a bounding box in corner angle notation
    TODO: review copilot code
    """
    x, y, w, h, c0, s0 = (bbox[:, i] for i in range(6))

    ### COPILOT:
    x1 = x + (- (w * c0) + (h * s0)) / 2; y1 = y + (- (w * s0) - (h * c0)) / 2
    x2 = x + ((w * c0) + (h * s0)) / 2; y2 = y + ( (w * s0) - (h * c0)) / 2
    x3 = x + ((w * c0) - (h * s0)) / 2; y3 = y + ((w * s0) + (h * c0)) / 2
    x4 = x + (- (w * c0) - (h * s0)) / 2; y4 = y + ((w * s0) - (h * c0)) / 2
    ###

    return x1, y1, x2, y2, x3, y3, x4, y4

def convex_hull(points: torch.Tensor) -> torch.Tensor:
    """Computes the convex hull of a set of 2D points.

    Args:
        points: A 2D tensor of shape (N, 2) representing the points.

    Returns:
        A tensor of shape (M, 2) representing the points of the convex hull.
    """
    # Sort the points by x-coordinate.
    xs = points[:, 0]
    ys = points[:, 1]
    sorted_indices = torch.argsort(xs + ys / 1000) # add a small number to avoid ties
    xs = xs[sorted_indices]
    ys = ys[sorted_indices]

    # Compute the lower hull.
    lower = []
    for i in range(xs.size(0)):
        while len(lower) >= 2 and torch.cross(lower[-2], lower[-1], dim=0) <= 0:
            lower.pop()
        lower.append(torch.tensor([xs[i], ys[i]]))

    # Compute the upper hull.
    upper = []
    for i in reversed(range(xs.size(0))):
        while len(upper) >= 2 and torch.cross(upper[-2], upper[-1], dim=0) <= 0:
            upper.pop()
        upper.append(torch.tensor([xs[i], ys[i]]))

    # Concatenate the lower and upper hull points.
    return torch.cat([torch.stack(lower, dim=0), torch.stack(upper[:-1], dim=0)], dim=0)

def poly_area(poly: torch.Tensor) -> torch.Tensor:
    """
    Compute the area of a polygon using the Shoelace formula.
    TODO: test this!!!

    Args:
        poly: A 2D tensor of shape (N, 2) representing the polygon.

    Returns:
        A tensor of shape (1,) representing the area of the polygon.
    """

    x = poly[:, 0]
    y = poly[:, 1]

    S1 = torch.sum(x * torch.roll(y, -1))
    S2 = torch.sum(y * torch.roll(x, -1))

    return 0.5 * (S1 - S2).abs()

def box_area(bbox: torch.Tensor) -> torch.Tensor:
    """
    compute the area of a given bbox in corner angle notation
    """

    x1, y1, x2, y2, x3, y3, _, _ = corner_with_angle_to_points(bbox)
    return torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * torch.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)

def box_intersection_area(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    Compute Area(b1 intersect b2)
    """
    pass # TODO: implement this function

def box_union_area(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    Compute Area(b1 U b2) = Area(b1) + Area(b2) - Area(b1 intersect b2)
    """
    return box_area(bbox1) + box_area(bbox2) - box_intersection_area(bbox1, bbox2)

def generalized_intersection_over_union(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    Compute the (GIoU) Generalized Intersction over Union between bbox1 and bbox2
    """
    eps = torch.finfo(bbox1.dtype)

    union_area = box_union_area(bbox1, bbox2)
    intersection_area = box_intersection_area(bbox1, bbox2)

    IoU = union_area / (intersection_area + eps)
    convex_hull_area = poly_area(
        convex_hull(torch.cat([bbox1, bbox2], dim=0))
    )

    return IoU - (convex_hull_area - union_area) / (convex_hull_area + eps)