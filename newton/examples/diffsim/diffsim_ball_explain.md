# Differentiable Simulation Ball Example Analysis

## Overview
This example demonstrates Newton's differentiable simulation capabilities by optimizing a particle's initial velocity to hit a target after bouncing off walls and the ground. The key innovation is using automatic differentiation (AD) to compute gradients of the simulation outcome with respect to initial conditions.

## Physics Setup

### Scene Configuration
- **Particle**: Single ball starting at position (0, -0.5, 1.0) with initial velocity (0, 5.0, -5.0)
- **Target**: Located at (0, -2.0, 1.5) - the goal position the ball should reach
- **Obstacles**: 
  - Box wall at position (0, 2.0, 1.0) with dimensions 1.0 × 0.25 × 1.0
  - Ground plane at z=0
- **Contact Parameters**: 
  - Stiffness (ke): 10,000 N/m
  - Damping (kd): 10 N·s/m
  - Friction (mu): 0.2
  - Restitution: 1.0 (perfectly elastic collisions)

### Simulation Parameters
- **Time Integration**: 36 simulation steps, 8 substeps each (288 total substeps)
- **Time Step**: 1/60 seconds per frame, divided by 8 substeps = ~2.08ms per substep
- **Solver**: Semi-implicit integrator for stability with contact forces

## Differentiability Implementation

### 1. Gradient-Enabled Model Creation
```python
self.model = scene.finalize(requires_grad=True)
```
The `requires_grad=True` flag enables automatic differentiation throughout the simulation pipeline.

### 2. Warp Tape for Automatic Differentiation
```python
self.tape = wp.Tape()
with self.tape:
    self.forward()
self.tape.backward(self.loss)
```
Warp's `Tape` object records all operations during the forward pass and enables reverse-mode automatic differentiation (backpropagation).

### 3. Loss Function
```python
@wp.kernel
def loss_kernel(pos: wp.array(dtype=wp.vec3), target: wp.vec3, loss: wp.array(dtype=float)):
    delta = pos[0] - target
    loss[0] = wp.dot(delta, delta)
```
The loss is the squared Euclidean distance between the final particle position and the target.

### 4. Gradient Flow
The gradient flows backwards through:
1. **Loss computation** → position error
2. **Final state** → particle position at end of simulation  
3. **Integration steps** → velocity and position updates through physics
4. **Contact forces** → collision responses with walls and ground
5. **Initial conditions** → initial velocity (optimization parameter)

## Optimization Process

### Gradient Descent Update
```python
@wp.kernel  
def step_kernel(x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), alpha: float):
    x[tid] = x[tid] - grad[tid] * alpha
```
Simple gradient descent with learning rate α = 0.02 to update initial velocity.

### Optimization Loop
1. **Forward Simulation**: Run physics for 288 timesteps
2. **Loss Computation**: Calculate distance to target
3. **Backward Pass**: Compute gradients via automatic differentiation
4. **Parameter Update**: Adjust initial velocity using gradients
5. **Repeat**: Continue until convergence

## Key Technical Features

### GPU Acceleration with Graph Capture
```python
if wp.get_device().is_cuda:
    with wp.ScopedCapture() as capture:
        self.forward_backward()
    self.graph = capture.graph
```
On CUDA devices, the entire forward-backward pass is captured as a computational graph for maximum performance.

### Gradient Verification
The `check_grad()` method implements finite difference gradient checking:
- Perturbs each parameter by small epsilon (1e-3)
- Computes numerical gradient: `(f(x+ε) - f(x-ε)) / (2ε)`  
- Compares against analytical gradient from automatic differentiation
- Validates correctness with tolerance of 5e-2

### Contact Handling
```python
self.contacts = self.model.collide(self.states[0], soft_contact_margin=10.0)
```
Uses soft contact model with:
- Pre-computed contact points for efficiency
- Differentiable contact forces
- Smooth force gradients even during contact transitions

## Expected Behavior

### Optimization Trajectory
- **Initial**: Ball follows ballistic path, likely missing target
- **Iteration 1-20**: Large adjustments to initial velocity based on gradients
- **Iteration 20+**: Fine-tuning with smaller velocity changes
- **Convergence**: Ball trajectory passes through or very close to target

### Loss Evolution  
- **Decreasing trend**: Each iteration should reduce distance to target
- **Validation**: All losses < 10.0, consecutive improvements > 1e-3
- **Convergence**: Loss approaches near-zero as ball hits target

## Applications

This example demonstrates the foundation for:
- **Trajectory Optimization**: Planning paths through complex environments
- **Control Design**: Learning control policies for robotic systems  
- **Inverse Problems**: Inferring initial conditions from observed outcomes
- **Parameter Identification**: Estimating material properties from motion data

## Technical Advantages

1. **Exact Gradients**: AD provides machine-precision gradients vs. noisy finite differences
2. **Computational Efficiency**: Single forward+backward pass vs. multiple forward evaluations
3. **Scalability**: Gradient computation cost independent of parameter count
4. **GPU Acceleration**: Full pipeline runs on GPU with graph optimization
5. **Contact Differentiability**: Handles complex contact scenarios smoothly

The example showcases Newton's ability to seamlessly integrate physics simulation with modern machine learning optimization techniques, enabling sophisticated applications in robotics, computer graphics, and scientific computing.