import os
import sys
import torch
import random

# Add parent directory to path to enable absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.rope_env import BatchedRopeEnv
from simulator.initial_states import (
    generate_horizontal, 
    generate_hanging, 
    generate_random_walk
)
from simulator.trajectory_planner import ForceTrajectoryPlanner

def generate_dataset(num_samples=500, B=50, N=50, length=2.0, output_file='data/thesis/train.pt', trajectory_mode='mixed'):
    """
    Generates a complete dataset split with randomized physics, initial states,
    and trajectories. 
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n--- Generating Split: {os.path.basename(output_file)} ---")
    print(f"Samples: {num_samples}, N: {N}, L: {length}, Device: {device}")
    
    T = 1000
    save_interval = 10
    num_saved_steps = T // save_interval
    num_batches = num_samples // B
    
    all_positions = []
    all_forces = []
    all_node_types = []
    all_metadata = []

    for batch_idx in range(num_batches):
        print(f"Processing batch {batch_idx+1}/{num_batches}...")
        
        physics_kwargs = {
            'stretch_stiffness': random.uniform(5000.0, 25000.0),
            'bend_stiffness': random.uniform(100.0, 500.0),
            'damping': random.uniform(5.0, 20.0),
            'mass_per_node': random.uniform(0.1, 0.4),
            'gravity': [0.0, 0.0, -9.81],
            'dt': 0.001
        }
        
        env = BatchedRopeEnv(B, N, length, device=device, **physics_kwargs)
        
        # 2. Randomize Initial State (Diversity & Grounding)
        init_pos = torch.zeros((B, N, 3), device=device)
        for b in range(B):
            state_choice = random.choice(['horizontal', 'random'])
            if state_choice == 'horizontal':
                z_start = random.choice([0.0, random.uniform(0.0, 0.5)])
                sample_pos = generate_horizontal(1, N, length, device)[0]
                sample_pos[:, 2] = z_start
                init_pos[b] = sample_pos
            else:
                sample_pos = generate_random_walk(1, N, length, device)[0]
                z_min_target = random.choice([0.0, random.uniform(0.0, 0.3)])
                current_z_min = torch.min(sample_pos[:, 2])
                sample_pos[:, 2] += (z_min_target - current_z_min)
                init_pos[b] = sample_pos
            
        env.reset(init_pos)
        
        # 3. Randomize Node Types & Trajectories
        node_types = torch.zeros((B, N, 1), dtype=torch.long, device=device)
        actuated_nodes_list = []
        planners = []
        magnitudes = torch.zeros(B, device=device)
        
        total_mass = N * physics_kwargs['mass_per_node']
        gravity_mag = total_mass * 9.81

        for b in range(B):
            num_fixed = random.randint(0, 2)
            fixed_indices = random.sample(range(N), num_fixed)
            node_types[b, fixed_indices, 0] = 2
            
            available = [i for i in range(N) if i not in fixed_indices]
            act_idx = random.sample(available, 1)
            node_types[b, act_idx, 0] = 1
            actuated_nodes_list.append(act_idx)
            
            if trajectory_mode == 'balanced':
                sample_global_idx = batch_idx * B + b
                traj_choice = 'circular' if sample_global_idx < (num_samples // 2) else 'constant'
            else:
                traj_choice = random.choice(['circular', 'constant'])
                
            planners.append(ForceTrajectoryPlanner(trajectory_type=traj_choice, 
                                                 omega=random.uniform(0.05, 0.2), 
                                                 device=device))
            magnitudes[b] = random.uniform(0.5, 2.0) * gravity_mag
            
        env.node_types = node_types
        
        # 4. Simulation Loop (T=1000)
        positions_traj = torch.zeros((num_saved_steps, B, N, 3), device=device)
        forces_traj = torch.zeros((num_saved_steps, B, N, 3), device=device)
        
        saved_idx = 0
        for t in range(1000):
            batch_forces = torch.zeros((B, N, 3), device=device)
            if t >= 200:
                for b in range(B):
                    direction = planners[b].get_direction(t - 200)
                    batch_forces[b, actuated_nodes_list[b][0]] = direction * magnitudes[b]
            
            state = env.step(batch_forces)
            
            if (t + 1) % save_interval == 0:
                positions_traj[saved_idx] = state['positions']
                forces_traj[saved_idx] = batch_forces
                saved_idx += 1
                
        # Collect results
        all_positions.append(positions_traj.cpu())
        all_forces.append(forces_traj.cpu())
        all_node_types.append(env.node_types.cpu())
        all_metadata.append({
            'physics': physics_kwargs, 
            'trajectories': [p.trajectory_type for p in planners],
            'magnitudes': magnitudes.cpu().tolist()
        })

    # Combine and Save
    final_dict = {
        'positions': torch.cat(all_positions, dim=1), # (T_save, Num_Samples, N, 3)
        'actuation_forces': torch.cat(all_forces, dim=1),
        'node_types': torch.cat(all_node_types, dim=0), # (Num_Samples, N, 1)
        'metadata_list': all_metadata
    }
    torch.save(final_dict, output_file)
    print(f"Saved {num_samples} samples to {output_file}")

if __name__ == '__main__':
    # ── Large-Scale Dataset Generation for Thesis ──────────────────────────────
    # All splits use N=50, L=1.0m (same geometry as the trained model).
    # Outputs go to data/thesis/large_split/ to keep original data intact.

    # 1. Training Set: 8,000 samples — 10x the old size
    generate_dataset(num_samples=8000, B=50, N=50, length=1.0,
                     output_file='data/thesis/large_split/train.pt',
                     trajectory_mode='mixed')

    # 2. Validation Set: 500 samples
    generate_dataset(num_samples=500, B=50, N=50, length=1.0,
                     output_file='data/thesis/large_split/val.pt',
                     trajectory_mode='mixed')

    # 3. Test Set: 500 samples
    generate_dataset(num_samples=500, B=50, N=50, length=1.0,
                     output_file='data/thesis/large_split/test.pt',
                     trajectory_mode='mixed')

    print(f"\n--- Thesis Large-Split Data Generation Complete! ---")