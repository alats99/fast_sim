import math
import torch
from typing import Tuple, Union

@torch.no_grad()
def generate_horizontal(B: int, N: int, length: float, device: Union[str, torch.device]) -> torch.Tensor:
    """
    Generate ropes laid straight along the X-axis, slightly elevated above Z=0.

    Args:
        B: Batch size.
        N: Number of nodes per rope.
        length: Total expected length of the rope.
        device: PyTorch device to allocate the tensors on.

    Returns:
        Tensor of shape (B, N, 3) with node positions.
    """
    positions = torch.zeros((B, N, 3), device=device)
    
    # Array of node coordinates stretching from x=0 to x=length
    x_coords = torch.linspace(0, length, N, device=device)
    
    positions[..., 0] = x_coords.unsqueeze(0).expand(B, N)
    
    # Elevate slightly
    positions[..., 2] = 0.1
    
    return positions

@torch.no_grad()
def generate_hanging(B: int, N: int, length: float, device: Union[str, torch.device], anchor_point: Tuple[float, float, float] = (0.0, 0.0, 1.0)) -> torch.Tensor:
    """
    Generate ropes hanging straight down along the Z-axis from a specified anchor.

    Args:
        B: Batch size.
        N: Number of nodes per rope.
        length: Total expected length of the rope.
        device: PyTorch device to allocate the tensors on.
        anchor_point: Spatial point (x, y, z) to hang the rope down from.

    Returns:
        Tensor of shape (B, N, 3) with node positions.
    """
    positions = torch.zeros((B, N, 3), device=device)
    anchor_tensor = torch.tensor(anchor_point, device=device, dtype=torch.float32)
    
    # Z-coordinates extending downwards from the anchor
    z_coords = torch.linspace(anchor_tensor[2].item(), anchor_tensor[2].item() - length, N, device=device)
    
    positions[..., 0] = anchor_tensor[0]
    positions[..., 1] = anchor_tensor[1]
    positions[..., 2] = z_coords.unsqueeze(0).expand(B, N)
    
    return positions

@torch.no_grad()
def generate_coiled_spiral(B: int, N: int, length: float, device: Union[str, torch.device]) -> torch.Tensor:
    """
    Generate ropes arranged in a generic Archimedean spiral on the XY plane.
    Points will automatically scale such that total piecewise arc length equals `length`.

    Args:
        B: Batch size.
        N: Number of nodes per rope.
        length: Total expected length of the rope curve.
        device: PyTorch device to allocate the tensors on.

    Returns:
        Tensor of shape (B, N, 3) with node positions.
    """
    # Create the baseline spiral extending to 3 full rotations
    theta = torch.linspace(0, 6 * math.pi, N, device=device)
    x = theta * torch.cos(theta)
    y = theta * torch.sin(theta)
    z = torch.zeros_like(x)
    
    spiral_points = torch.stack([x, y, z], dim=-1)
    
    # Calculate the current total polyline length 
    deltas = spiral_points[1:] - spiral_points[:-1]
    current_length = torch.sum(torch.norm(deltas, dim=-1))
    
    # Scale all coordinates so the total segment length equals `length` exactly
    scale_factor = length / torch.clamp(current_length, min=1e-8)
    spiral_points = spiral_points * scale_factor
    
    # Assign and return a cloned array memory slice
    positions = spiral_points.unsqueeze(0).expand(B, N, 3).clone()
    
    return positions

@torch.no_grad()
def generate_random_walk(B: int, N: int, length: float, device: Union[str, torch.device]) -> torch.Tensor:
    """
    Generate batched non-intersection-checked 3D random walks.
    Enforces that consecutive nodes are separated by exactly `length / (N - 1)`.

    Args:
        B: Batch size.
        N: Number of nodes per rope.
        length: Total expected length of the rope.
        device: PyTorch device to allocate the tensors on.

    Returns:
        Tensor of shape (B, N, 3) with node positions.
    """
    positions = torch.zeros((B, N, 3), device=device)
    segment_length = length / (N - 1)
    
    # Node 0 remains at the origin (0, 0, 0) for each sample. 
    # Build cumulatively outwards across sequence N
    for i in range(1, N):
        # Sample B directions natively in 3D
        directions = torch.randn((B, 3), device=device)
        directions = directions / torch.clamp(torch.norm(directions, dim=-1, keepdim=True), min=1e-8)
        
        step = directions * segment_length
        positions[:, i, :] = positions[:, i-1, :] + step
        
    return positions
