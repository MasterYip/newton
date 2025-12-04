# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Diffsim ANYmal C
#
# Shows how to use MPC (Model Predictive Control) with differentiable
# simulation to control the ANYmal C quadruped robot for walking.
#
# Command: python -m newton.examples diffsim_anymal_c
#
###########################################################################

import numpy as np
import warp as wp
import warp.optim

import newton
import newton.examples
import newton.utils
from newton import State

# Joint ordering conversions between Lab and MuJoCo conventions
lab_to_mujoco = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
mujoco_to_lab = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]


@wp.kernel
def increment_seed(seed: wp.array(dtype=int)):
    """Increments the random seed for trajectory sampling."""
    seed[0] += 1


@wp.kernel
def sample_gaussian(
    mean_trajectory: wp.array(dtype=float, ndim=3),
    noise_scale: float,
    num_control_points: int,
    control_dim: int,
    control_limits: wp.array(dtype=float, ndim=2),
    seed: wp.array(dtype=int),
    rollout_trajectories: wp.array(dtype=float, ndim=3),
):
    """
    Samples trajectory variations using Gaussian noise for MPC exploration.
    Generates diverse control candidates around the mean trajectory.
    """
    world_id, point_id, control_id = wp.tid()
    unique_id = (world_id * num_control_points + point_id) * control_dim + control_id
    r = wp.rand_init(seed[0], unique_id)
    mean = mean_trajectory[0, point_id, control_id]
    lo, hi = control_limits[control_id, 0], control_limits[control_id, 1]
    sample = mean + noise_scale * wp.randn(r)
    for _i in range(10):
        if sample < lo or sample > hi:
            sample = mean + noise_scale * wp.randn(r)
        else:
            break
    rollout_trajectories[world_id, point_id, control_id] = wp.clamp(sample, lo, hi)


@wp.kernel
def replicate_states(
    joint_q_in: wp.array(dtype=float),
    joint_qd_in: wp.array(dtype=float),
    dofs_per_world: int,
    joint_q_out: wp.array(dtype=float),
    joint_qd_out: wp.array(dtype=float),
):
    """
    Copies the current robot state to all rollout simulation worlds.
    Initializes parallel MPC rollouts from identical starting conditions.
    """
    tid = wp.tid()
    world_offset = tid * dofs_per_world
    for i in range(dofs_per_world):
        joint_q_out[world_offset + i] = joint_q_in[i]
        joint_qd_out[world_offset + i] = joint_qd_in[i]


@wp.kernel
def quat_rotate_inverse(q: wp.array(dtype=float), v: wp.array(dtype=float), result: wp.array(dtype=float)):
    """Rotate a vector by the inverse of a quaternion (quat is XYZW format)."""
    tid = wp.tid()
    quat = wp.vec4(q[tid * 4 + 0], q[tid * 4 + 1], q[tid * 4 + 2], q[tid * 4 + 3])
    vec = wp.vec3(v[tid * 3 + 0], v[tid * 3 + 1], v[tid * 3 + 2])
    
    q_w = quat[3]
    q_vec = wp.vec3(quat[0], quat[1], quat[2])
    
    a = vec * (2.0 * q_w * q_w - 1.0)
    b = wp.cross(q_vec, vec) * q_w * 2.0
    c = q_vec * wp.dot(q_vec, vec) * 2.0
    
    res = a - b + c
    result[tid * 3 + 0] = res[0]
    result[tid * 3 + 1] = res[1]
    result[tid * 3 + 2] = res[2]


@wp.kernel
def anymal_cost(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    command: wp.vec3,
    step: int,
    horizon_length: int,
    weighting: float,
    default_joint_pos: wp.array(dtype=float),
    cost: wp.array(dtype=float),
):
    """
    Computes multi-objective cost function for ANYmal C walking trajectory evaluation.
    
    Cost Components:
    1. Velocity Tracking: Match commanded linear (xy) and angular (z) velocities
    2. Orientation: Penalize non-flat base orientation (gravity vector in base frame)
    3. Base Height: Maintain target height above ground
    4. Joint Positions: Stay near default standing pose
    5. Joint Velocities: Minimize excessive joint motion
    6. Torques: Minimize control effort
    7. Angular Velocity: Penalize pitch/roll rotation
    """
    world_id = wp.tid()
    dof_offset = world_id * 18
    
    # Extract base pose (first 7 DOFs: xyz position + xyzw quaternion)
    base_pos = wp.vec3(joint_q[dof_offset + 0], joint_q[dof_offset + 1], joint_q[dof_offset + 2])
    base_quat = wp.vec4(joint_q[dof_offset + 3], joint_q[dof_offset + 4], 
                        joint_q[dof_offset + 5], joint_q[dof_offset + 6])
    
    # Extract base velocities (first 6 DOF velocities: xyz linear + xyz angular)
    base_lin_vel = wp.vec3(joint_qd[dof_offset + 0], joint_qd[dof_offset + 1], joint_qd[dof_offset + 2])
    base_ang_vel = wp.vec3(joint_qd[dof_offset + 3], joint_qd[dof_offset + 4], joint_qd[dof_offset + 5])
    
    # Transform velocities to base frame
    q_w = base_quat[3]
    q_vec = wp.vec3(base_quat[0], base_quat[1], base_quat[2])
    a = base_lin_vel * (2.0 * q_w * q_w - 1.0)
    b = wp.cross(q_vec, base_lin_vel) * q_w * 2.0
    c = q_vec * wp.dot(q_vec, base_lin_vel) * 2.0
    vel_base = a - b + c
    
    a_ang = base_ang_vel * (2.0 * q_w * q_w - 1.0)
    b_ang = wp.cross(q_vec, base_ang_vel) * q_w * 2.0
    c_ang = q_vec * wp.dot(q_vec, base_ang_vel) * 2.0
    ang_vel_base = a_ang - b_ang + c_ang
    
    # Gravity vector in base frame (for orientation penalty)
    gravity_world = wp.vec3(0.0, 0.0, -1.0)
    a_g = gravity_world * (2.0 * q_w * q_w - 1.0)
    b_g = wp.cross(q_vec, gravity_world) * q_w * 2.0
    c_g = q_vec * wp.dot(q_vec, gravity_world) * 2.0
    gravity_base = a_g - b_g + c_g
    
    # 1. Velocity tracking cost (primary objective)
    lin_vel_error = (command[0] - vel_base[0]) * (command[0] - vel_base[0]) + \
                    (command[1] - vel_base[1]) * (command[1] - vel_base[1])
    ang_vel_error = (command[2] - ang_vel_base[2]) * (command[2] - ang_vel_base[2])
    tracking_sigma = 0.25
    tracking_cost = (2.0 - wp.exp(-lin_vel_error / tracking_sigma) - wp.exp(-ang_vel_error / tracking_sigma))
    
    # 2. Orientation cost (keep base flat)
    orientation_cost = gravity_base[0] * gravity_base[0] + gravity_base[1] * gravity_base[1]
    
    # 3. Base height cost (maintain 0.5m height)
    target_height = 0.5
    height_cost = (base_pos[2] - target_height) * (base_pos[2] - target_height)
    
    # 4. Joint position cost (stay near default pose)
    joint_pos_cost = 0.0
    for i in range(12):
        diff = joint_q[dof_offset + 7 + i] - default_joint_pos[i]
        joint_pos_cost += diff * diff
    
    # 5. Joint velocity cost
    joint_vel_cost = 0.0
    for i in range(12):
        vel = joint_qd[dof_offset + 6 + i]
        joint_vel_cost += vel * vel
    
    # 6. Torque/control effort cost
    torque_cost = 0.0
    for i in range(12):
        torque = joint_target[world_id * 12 + i]
        torque_cost += torque * torque
    
    # 7. Angular velocity xy cost (pitch/roll)
    ang_vel_xy_cost = ang_vel_base[0] * ang_vel_base[0] + ang_vel_base[1] * ang_vel_base[1]
    
    # 8. Z-axis linear velocity cost
    lin_vel_z_cost = vel_base[2] * vel_base[2]
    
    # Temporal discounting
    discount = 0.9 ** wp.float(horizon_length - step - 1) / wp.float(horizon_length) ** 2.0
    
    # Weight the costs (from anymal_c_flat_config.py)
    tracking_weight = 1.5  # Combined lin + ang tracking
    orientation_weight = 5.0
    height_weight = 0.0  # Not used in flat config
    joint_pos_weight = 0.0
    joint_vel_weight = 0.0
    torque_weight = 0.000025 * 1000.0  # Scale up for visibility
    ang_vel_xy_weight = 0.05
    lin_vel_z_weight = 2.0
    
    total_cost = (
        tracking_cost * tracking_weight +
        orientation_cost * orientation_weight +
        height_cost * height_weight +
        joint_pos_cost * joint_pos_weight +
        joint_vel_cost * joint_vel_weight +
        torque_cost * torque_weight +
        ang_vel_xy_cost * ang_vel_xy_weight +
        lin_vel_z_cost * lin_vel_z_weight
    ) * weighting * discount
    
    wp.atomic_add(cost, world_id, total_cost)


@wp.kernel
def distribute_joint_targets(
    actuated_targets: wp.array(dtype=float),
    full_targets: wp.array(dtype=float),
    num_robots: int,
):
    """Distribute actuated joint targets (12 per robot) to full DOF array (18 per robot, first 6 are floating base)."""
    robot_id, joint_id = wp.tid()
    # Source: 12 actuated joints per robot
    src_idx = robot_id * 12 + joint_id
    # Destination: skip first 6 DOFs (floating base) per robot
    dst_idx = robot_id * 18 + 6 + joint_id
    full_targets[dst_idx] = actuated_targets[src_idx]


@wp.kernel
def enforce_control_limits(
    control_limits: wp.array(dtype=float, ndim=2),
    control_points: wp.array(dtype=float, ndim=3),
):
    """Enforces physical constraints on control parameters after optimization."""
    world_id, t_id, control_id = wp.tid()
    lo, hi = control_limits[control_id, 0], control_limits[control_id, 1]
    control_points[world_id, t_id, control_id] = wp.clamp(control_points[world_id, t_id, control_id], lo, hi)


@wp.kernel
def pick_best_trajectory(
    rollout_trajectories: wp.array(dtype=float, ndim=3),
    lowest_cost_id: int,
    best_traj: wp.array(dtype=float, ndim=3),
):
    """Selects the lowest-cost trajectory from all rollout evaluations."""
    t_id, control_id = wp.tid()
    best_traj[0, t_id, control_id] = rollout_trajectories[lowest_cost_id, t_id, control_id]


@wp.kernel
def interpolate_control_linear(
    control_points: wp.array(dtype=float, ndim=3),
    control_dofs: wp.array(dtype=int),
    default_joint_pos: wp.array(dtype=float),
    action_scale: float,
    t: float,
    joint_target: wp.array(dtype=float),
):
    """
    Converts sparse control waypoints into continuous joint position targets.
    Control points represent action deltas, not absolute positions.
    """
    world_id, control_id = wp.tid()
    t_id = int(t)
    frac = t - wp.floor(t)
    
    control_left = control_points[world_id, t_id, control_id]
    control_right = control_points[world_id, t_id + 1, control_id]
    
    action = control_left * (1.0 - frac) + control_right * frac
    joint_id = control_dofs[control_id]
    
    # Convert action to joint target: default + scaled_action
    joint_target[world_id * 12 + joint_id] = default_joint_pos[joint_id] + action_scale * action


class ANYmalC:
    def __init__(
        self,
        name: str,
        trajectory_shape: tuple[int, int],
        variation_count: int = 1,
        requires_grad: bool = False,
        state_count: int | None = None,
    ) -> None:
        self.variation_count = variation_count
        self.requires_grad = requires_grad
        self.sim_tick = 0
        
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        
        # Joint and shape configs matching the walking example
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        
        # Load ANYmal C URDF
        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        stage_path = str(asset_path / "urdf" / "anymal.urdf")
        
        for i in range(variation_count):
            builder.add_urdf(
                stage_path,
                xform=wp.transform(
                    wp.vec3(float(i) * 2.0, 0.0, 0.62),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)
                ),
                floating=True,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                ignore_inertial_definitions=False,
            )
        
        # Set initial joint positions (default standing pose)
        initial_q = {
            "RH_HAA": 0.0, "RH_HFE": -0.4, "RH_KFE": 0.8,
            "LH_HAA": 0.0, "LH_HFE": -0.4, "LH_KFE": 0.8,
            "RF_HAA": 0.0, "RF_HFE": 0.4, "RF_KFE": -0.8,
            "LF_HAA": 0.0, "LF_HFE": 0.4, "LF_KFE": -0.8,
        }
        
        for key, value in initial_q.items():
            joint_idx = builder.joint_key.index(key) + 6
            for i in range(variation_count):
                builder.joint_q[i * 18 + joint_idx] = value
        
        # Set joint PD gains
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5
        
        builder.add_ground_plane()
        
        self.model = builder.finalize(requires_grad=requires_grad)
        
        # Store default joint positions (12 joints, skip base 6 DOF velocities)
        self.default_joint_pos = wp.array([
            initial_q["LF_HAA"], initial_q["LF_HFE"], initial_q["LF_KFE"],
            initial_q["RF_HAA"], initial_q["RF_HFE"], initial_q["RF_KFE"],
            initial_q["LH_HAA"], initial_q["LH_HFE"], initial_q["LH_KFE"],
            initial_q["RH_HAA"], initial_q["RH_HFE"], initial_q["RH_KFE"],
        ], dtype=float)
        
        # Initialize states
        if requires_grad:
            self.states = tuple(self.model.state() for _ in range(state_count + 1))
            self.controls = tuple(self.model.control() for _ in range(state_count))
        else:
            self.states = [self.model.state(), self.model.state()]
            self.controls = (self.model.control(),)
        
        # Create joint target arrays
        for control in self.controls:
            control.joint_targets = wp.zeros(variation_count * 12, dtype=float, requires_grad=requires_grad)
        
        # Define trajectories (actions in joint space)
        self.trajectories = wp.zeros(
            (variation_count, trajectory_shape[0], trajectory_shape[1]),
            dtype=float,
            requires_grad=requires_grad,
        )
        
        self.body_count = 13 * variation_count  # 13 bodies per robot
        self.dof_count = 18  # 6 (base) + 12 (joints)
    
    @property
    def state(self) -> newton.State:
        return self.states[self.sim_tick if self.requires_grad else 0]
    
    @property
    def next_state(self) -> newton.State:
        return self.states[self.sim_tick + 1 if self.requires_grad else 1]
    
    @property
    def control(self) -> newton.Control:
        return self.controls[min(len(self.controls) - 1, self.sim_tick) if self.requires_grad else 0]


class Example:
    def __init__(self, viewer, verbose=False, num_rollouts=16, args=None):
        self.fps = 50
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        
        self.verbose = verbose
        self.rollout_count = num_rollouts
        self.viewer = viewer
        
        # Walking command: [vel_x, vel_y, ang_vel_z]
        self.command = wp.vec3(1.0, 0.0, 0.0)  # Walk forward at 1 m/s
        
        # MPC parameters
        self.optim_step_count = 6
        self.control_point_step = 5  # Steps between control waypoints
        self.control_point_count = 4  # Number of waypoints
        self.control_point_data_count = self.control_point_count + 1
        
        # Control configuration (12 joint positions)
        self.control_dofs = wp.array(list(range(12)), dtype=int)
        self.control_dim = 12
        self.action_scale = 0.5  # Scale for action -> joint position
        
        # Control limits (action space, not joint limits)
        self.control_limits = wp.array([(-10.0, 10.0)] * self.control_dim, dtype=float)
        
        # Create reference robot
        self.robot = ANYmalC(
            "anymal",
            (self.control_point_data_count, self.control_dim),
        )
        
        # Create rollout robots
        self.rollout_step_count = self.control_point_step * self.control_point_count
        self.rollouts = ANYmalC(
            "rollout",
            (self.control_point_data_count, self.control_dim),
            variation_count=self.rollout_count,
            requires_grad=True,
            state_count=self.rollout_step_count * self.sim_substeps,
        )
        
        self.seed = wp.zeros(1, dtype=int)
        self.rollout_costs = wp.zeros(self.rollout_count, dtype=float, requires_grad=True)
        self.cost_history = []
        
        # Create collision pipeline from command-line args
        self.collision_pipeline_robot = newton.examples.create_collision_pipeline(self.robot.model, args)
        self.collision_pipeline_rollouts = newton.examples.create_collision_pipeline(self.rollouts.model, args)
        
        # Solvers
        self.solver_rollouts = newton.solvers.SolverMuJoCo(
            self.rollouts.model,
            use_mujoco_contacts=args.use_mujoco_contacts if args else False,
            ls_parallel=True,
            njmax=16384,
            nconmax=6144,
        )
        
        self.solver_robot = newton.solvers.SolverMuJoCo(
            self.robot.model,
            use_mujoco_contacts=args.use_mujoco_contacts if args else False,
            ls_parallel=True,
            njmax=16384,
            nconmax=6144,
        )
        
        self.optimizer = warp.optim.SGD(
            [self.rollouts.trajectories.flatten()],
            lr=5e-3,
            nesterov=False,
            momentum=0.0,
        )
        
        # Evaluate FK to update body poses
        newton.eval_fk(self.robot.model, self.robot.state.joint_q, self.robot.state.joint_qd, self.robot.state)
        
        # Initialize contacts using collision pipeline
        self.contacts_robot = self.robot.model.collide(self.robot.state, collision_pipeline=self.collision_pipeline_robot)
        self.contacts_rollouts = self.rollouts.model.collide(self.rollouts.state, collision_pipeline=self.collision_pipeline_rollouts)
        
        self.viewer.set_model(self.robot.model)
        self.capture()
    
    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.forward_backward()
            self.graph = capture.graph
        else:
            self.graph = None
    
    def forward_backward(self):
        self.tape = wp.Tape()
        with self.tape:
            self.forward()
        self.rollout_costs.grad.fill_(1.0)
        self.tape.backward()
    
    def update_robot(self, robot: ANYmalC, solver) -> None:
        robot.state.clear_forces()
        
        wp.launch(
            interpolate_control_linear,
            dim=(robot.variation_count, self.control_dim),
            inputs=(
                robot.trajectories,
                self.control_dofs,
                robot.default_joint_pos,
                self.action_scale,
                robot.sim_tick / (self.sim_substeps * self.control_point_step),
            ),
            outputs=(robot.control.joint_targets,),
        )
        
        # Distribute actuated joint targets (12 DOF) to full state including floating base (18 DOF)
        wp.launch(
            distribute_joint_targets,
            dim=(robot.variation_count, 12),
            inputs=(robot.control.joint_targets, robot.control.joint_target_pos, robot.variation_count),
        )
        
        # Compute contacts using collision pipeline
        contacts = robot.model.collide(robot.state, collision_pipeline=self.collision_pipeline_rollouts if robot.requires_grad else self.collision_pipeline_robot)
        
        solver.step(robot.state, robot.next_state, robot.control, contacts, self.sim_dt)
        robot.sim_tick += 1
    
    def forward(self):
        self.rollouts.sim_tick = 0
        self.rollout_costs.zero_()
        
        wp.launch(
            replicate_states,
            dim=self.rollout_count,
            inputs=(
                self.robot.state.joint_q,
                self.robot.state.joint_qd,
                self.robot.dof_count,
            ),
            outputs=(
                self.rollouts.state.joint_q,
                self.rollouts.state.joint_qd,
            ),
        )
        
        for i in range(self.rollout_step_count):
            for _ in range(self.sim_substeps):
                self.update_robot(self.rollouts, self.solver_rollouts)
            
            wp.launch(
                anymal_cost,
                dim=self.rollout_count,
                inputs=(
                    self.rollouts.state.joint_q,
                    self.rollouts.state.joint_qd,
                    self.rollouts.control.joint_targets,
                    self.command,
                    i,
                    self.rollout_step_count,
                    1e2,
                    self.rollouts.default_joint_pos,
                ),
                outputs=(self.rollout_costs,),
            )
    
    def step_optimizer(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.forward_backward()
        
        self.optimizer.step([self.rollouts.trajectories.grad.flatten()])
        
        wp.launch(
            enforce_control_limits,
            dim=self.rollouts.trajectories.shape,
            inputs=(self.control_limits,),
            outputs=(self.rollouts.trajectories,),
        )
        self.tape.zero()
    
    def step(self):
        # Sample trajectories
        noise_scale = 0.2
        wp.launch(
            sample_gaussian,
            dim=(
                self.rollouts.trajectories.shape[0] - 1,
                self.rollouts.trajectories.shape[1],
                self.rollouts.trajectories.shape[2],
            ),
            inputs=(
                self.robot.trajectories,
                noise_scale,
                self.control_point_data_count,
                self.control_dim,
                self.control_limits,
                self.seed,
            ),
            outputs=(self.rollouts.trajectories,),
        )
        
        wp.launch(increment_seed, dim=1, inputs=(), outputs=(self.seed,))
        
        # Optimize
        for _ in range(self.optim_step_count):
            self.step_optimizer()
        
        # Pick best trajectory
        wp.synchronize()
        lowest_cost_id = np.argmin(self.rollout_costs.numpy())
        wp.launch(
            pick_best_trajectory,
            dim=(self.control_point_data_count, self.control_dim),
            inputs=(self.rollouts.trajectories, lowest_cost_id),
            outputs=(self.robot.trajectories,),
        )
        self.rollouts.trajectories[-1].assign(self.robot.trajectories[0])
        
        # Simulate reference robot
        self.robot.sim_tick = 0
        for _ in range(self.sim_substeps):
            self.update_robot(self.robot, self.solver_robot)
            self.robot.states[0], self.robot.states[1] = self.robot.states[1], self.robot.states[0]
        
        loss = np.min(self.rollout_costs.numpy())
        if self.verbose:
            print(f"[{(self.frame + 1):3d}] loss={loss:.6f}")
        self.cost_history.append(loss)
    
    def render(self):
        self.viewer.begin_frame(self.frame * self.frame_dt)
        self.viewer.log_state(self.robot.state)
        self.viewer.end_frame()
        self.frame += 1


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print status messages.")
    parser.add_argument("--num_rollouts", type=int, default=8, help="Number of rollouts for MPC.")
    
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args.verbose, args.num_rollouts, args)
    newton.examples.run(example, args)
