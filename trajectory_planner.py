import torch
import numpy as np

class ForceTrajectoryPlanner:
    """
    Trajectory planner for computing time-varying external actuation forces.
    Generates dynamic force vectors (e.g., circular, figure-eight) to feed into the simulation loop.
    """
    def __init__(
        self, 
        trajectory_type: str = 'circular', 
        omega: float = 0.02, 
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            trajectory_type: 'circular', 'constant', 'square_wave'
            omega: Base angular velocity controlling the speed of the trajectory
            device: PyTorch device
        """
        self.trajectory_type = trajectory_type
        self.omega = omega
        self.device = device
        
        # Persistent random direction for 'constant' mode
        self.fixed_dir = torch.randn(3, device=device)
        self.fixed_dir /= torch.norm(self.fixed_dir)
        
    def get_direction(self, t: int) -> torch.Tensor:
        """
        Returns a normalized (3,) direction vector for the given simulation step t.
        """
        angle = self.omega * t
        
        if self.trajectory_type == 'circular':
            dir_x = np.cos(angle)
            dir_y = np.sin(angle)
            dir_z = 0.5
            
        elif self.trajectory_type == 'square_wave':
            # Jumps between two directions
            if (int(angle / np.pi) % 2) == 0:
                dir_x, dir_y, dir_z = 1.0, 0.0, 1.0
            else:
                dir_x, dir_y, dir_z = -1.0, 0.0, 1.0
            
        else: # 'constant' or Default
            return self.fixed_dir
            
        direction = torch.tensor([dir_x, dir_y, dir_z], device=self.device, dtype=torch.float32)
        direction = direction / torch.norm(direction)
        return direction
        
    def get_force_tensor(self, t: int, B: int, N: int, actuated_nodes: list, force_magnitude: float) -> torch.Tensor:
        """
        Builds the (B, N, 3) external forces tensor required by the simulator for timestep t.
        """
        direction = self.get_direction(t)
        
        forces = torch.zeros((B, N, 3), dtype=torch.float32, device=self.device)
        for node_idx in actuated_nodes:
            forces[:, node_idx, :] = force_magnitude * direction
            
        return forces
