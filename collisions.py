import torch
from typing import Tuple

@torch.no_grad()
def sdf_plane(points: torch.Tensor, plane_normal: torch.Tensor, plane_point: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the Signed Distance Field (SDF) to an infinite plane.

    Args:
        points: Tensor of shape (B, N, 3) representing the coordinates of the rope nodes.
        plane_normal: Tensor of shape (3,) representing the normal of the plane.
        plane_point: Tensor of shape (3,) representing a point on the plane.
        
    Returns:
        distances: Tensor of shape (B, N) containing the signed distances to the plane.
        normals: Tensor of shape (B, N, 3) containing the normals at the closest points.
    """
    plane_normal = plane_normal.to(dtype=points.dtype, device=points.device)
    plane_normal = plane_normal / torch.norm(plane_normal)
    plane_point = plane_point.to(dtype=points.dtype, device=points.device)
    
    # Vector from plane point to current point
    p_to_point = points - plane_point
    
    # Distance is the projection of p_to_point onto the plane normal
    distances = torch.einsum('bni,i->bn', p_to_point, plane_normal)
    
    # Normals are constant for an infinite plane
    normals = plane_normal.view(1, 1, 3).expand_as(points)
    
    return distances, normals

@torch.no_grad()
def sdf_sphere(points: torch.Tensor, center: torch.Tensor, radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the Signed Distance Field (SDF) to a sphere.

    Args:
        points: Tensor of shape (B, N, 3) representing the coordinates of the rope nodes.
        center: Tensor of shape (3,) representing the center of the sphere.
        radius: Float representing the sphere's radius.
        
    Returns:
        distances: Tensor of shape (B, N) containing the signed distances to the sphere.
        normals: Tensor of shape (B, N, 3) containing the normals at the closest points.
    """
    center = center.to(dtype=points.dtype, device=points.device)
    
    p_to_center = points - center
    
    # Euclidean distance from center
    distances_to_center = torch.norm(p_to_center, dim=-1)
    
    # SDF is distance to center minus the radius
    distances = distances_to_center - radius
    
    # Normal is the normalized vector pointing from center to point
    distances_to_center_safe = torch.clamp(distances_to_center, min=1e-8)
    normals = p_to_center / distances_to_center_safe.unsqueeze(-1)
    
    return distances, normals

@torch.no_grad()
def sdf_box(points: torch.Tensor, center: torch.Tensor, extents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the Signed Distance Field (SDF) to an axis-aligned bounding box (AABB).

    Args:
        points: Tensor of shape (B, N, 3) representing the coordinates of the rope nodes.
        center: Tensor of shape (3,) representing the center of the box.
        extents: Tensor of shape (3,) representing the half-widths (hx, hy, hz) of the box.
        
    Returns:
        distances: Tensor of shape (B, N) containing the signed distances to the box.
        normals: Tensor of shape (B, N, 3) containing the normals at the closest points.
    """
    center = center.to(dtype=points.dtype, device=points.device)
    extents = extents.to(dtype=points.dtype, device=points.device)
    
    # Transform points to local box space
    p_local = points - center
    
    # Calculate difference from box extents in all dimensions
    q = torch.abs(p_local) - extents
    
    q_max = torch.max(q, dim=-1).values
    q_out = torch.clamp(q, min=0.0)
    
    # Distance to closest point on the surface from outside
    dist_out = torch.norm(q_out, dim=-1)
    # Penetration depth if completely inside (all components of q < 0)
    dist_in = torch.min(q_max, torch.zeros_like(q_max))
    
    distances = dist_out + dist_in
    
    # Normal calculation
    signs = torch.sign(p_local)
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs) # Handle exact center zeroes
    
    # Normal when point is outside the bounds
    normal_out = q_out * signs
    normal_out_norm = torch.clamp(torch.norm(normal_out, dim=-1, keepdim=True), min=1e-8)
    normal_out = normal_out / normal_out_norm
    
    # Normal when point is inside (project out through the least penetrating face)
    q_max_idx = torch.argmax(q, dim=-1)
    normal_in = torch.nn.functional.one_hot(q_max_idx, num_classes=3).to(points.dtype) * signs
    
    is_outside = (q_max > 0.0).unsqueeze(-1)
    normals = torch.where(is_outside, normal_out, normal_in)
    
    return distances, normals
