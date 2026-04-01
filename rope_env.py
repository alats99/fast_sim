import torch
from typing import Dict, Any, Union
from simulator.base_env import BaseDeformableEnv

class BatchedRopeEnv(BaseDeformableEnv):
    """
    Rope-specific simulator with 1D linear connectivity.
    """
    def __init__(
        self, 
        B: int, 
        N: int, 
        length: float, 
        device: Union[str, torch.device] = 'cpu', 
        **physics_kwargs: Any
    ):
        super().__init__(B, N, device, **physics_kwargs)
        self.length = length
        self.stretch_stiffness = physics_kwargs.get('stretch_stiffness', 10000.0)
        self.bending_stiffness = physics_kwargs.get('bending_stiffness', 1000.0)
        
        self.l0_stretch = self.length / (self.N_nodes - 1)
        self.l0_bend = 2.0 * self.l0_stretch

    def _compute_internal_forces(self) -> torch.Tensor:
        forces = torch.zeros_like(self.positions)
        
        # Stretch (1st neighbor)
        diff_s = self.positions[:, 1:] - self.positions[:, :-1]
        dist_s = torch.norm(diff_s, dim=-1, keepdim=True)
        dir_s = diff_s / torch.clamp(dist_s, min=1e-8)
        f_s = self.stretch_stiffness * (dist_s - self.l0_stretch) * dir_s
        forces[:, :-1] += f_s
        forces[:, 1:] -= f_s
        
        # Bend (2nd neighbor)
        if self.N_nodes > 2:
            diff_b = self.positions[:, 2:] - self.positions[:, :-2]
            dist_b = torch.norm(diff_b, dim=-1, keepdim=True)
            dir_b = diff_b / torch.clamp(dist_b, min=1e-8)
            f_b = self.bending_stiffness * (dist_b - self.l0_bend) * dir_b
            forces[:, :-2] += f_b
            forces[:, 2:] -= f_b
            
        return forces

    # Override step to include specialized 1D damping
    def step(self, actuation_forces: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Perform standard integration from base
        state = super().step(actuation_forces)
        
        # Apply 1D adjacent damping (dashpot)
        v_diff = self.velocities[:, 1:] - self.velocities[:, :-1]
        damping_impulse = (self.damping * self.dt / self.mass_per_node) * v_diff
        self.velocities[:, :-1] += damping_impulse
        self.velocities[:, 1:] -= damping_impulse
        
        return self._get_state_dict()

