# Rope Simulator

This repository contains a custom offline simulator for a 1D deformable object (rope) in 3D space, written entirely in PyTorch.
The simulator is fully batched, meaning it executes multiple simulations simultaneously using vectorized operations without loops over the batch dimension.

## Files and Usage

### `base_env.py`
Contains the base class `BaseDeformableEnv`. All common physics logic is here:
- Semi-implicit Euler integration.
- Gravity management.
- Collisions with the Z=0 plane.
- Static and kinetic friction.

### `rope_env.py`
Extends `BaseDeformableEnv` for 1D objects (ropes). Computes tensile and bending forces along a linear chain of nodes.

### `collisions.py`
Contains purely vectorized mathematical functions for computing Signed Distance Fields (SDFs).
Calculates distances and normal vectors between rope nodes and various geometries:
- `sdf_plane` (Infinite plane)
- `sdf_sphere` (Sphere)
- `sdf_box` (Box)

### `initial_states.py`
Responsible for generating the initial positional coordinates of all ropes in a batch. Supports creating:
- Horizontal alignments
- Vertically hanging ropes
- 3D Random Walks
- Archimedean spirals

### `trajectory_planner.py`
Trajectory planner for computing time-varying external actuation forces. Generates dynamic force vectors (e.g., circular, square wave) to feed into the simulation loop.

### `data_generator.py`
The executable external script that:
- Sets up and generates the static offline dataset using the Simulator.
- Runs a loop dynamically randomizing physical parameters (stiffness, friction, damping, etc.).
- Simulates the ropes over time (with random actuation forces) for multiple steps.
- Saves the batched results (trajectories) directly to standalone `.pt` files.

These files constitute an extremely fast, fully parallelized data generation engine.

## Data Generation

To generate new data with the simulator, run the `data_generator.py` script from the project root directory:

```bash
python data_generator.py
```

The script contains ready-to-use configurations at the bottom (`__main__` block), where you can set up calls to the `generate_dataset()` function.

### Basic Parameters for `generate_dataset()`
- `num_samples`: Total number of trajectories you want in the `.pt` file.
- `B` (Batch Size): How many ropes run simultaneously. Choose a value that fits your GPU memory (e.g., 50).
- `N`: Number of nodes for the dataset. (e.g., `N=50`)
- `length`: Total length of the rope in meters. (e.g., `length=1.0`)
- `output_file`: Path to save the tensor dictionary.
- `trajectory_mode`: Options like `'balanced'`, `'circular'`, etc.

**Example** (Generating a Test Set):
```python
generate_dataset(
    num_samples=100, 
    B=20, 
    N=50, 
    length=1.0, 
    output_file='data/test.pt'
)
```
