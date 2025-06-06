#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, Twist
import numpy as np
import math
from scipy.optimize import minimize
import threading

class InverseKinematicsController(Node):
    def __init__(self):
        super().__init__('ik_controller')
        
        # Publishers - Changed to publish to /joint_states
        self.joint_states_pub = self.create_publisher(
            JointState, 
            '/joint_states', 
            10
        )
        
        # Publisher for base movement commands
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # Subscribers
        self.target_poses_sub = self.create_subscription(
            PoseArray,
            '/target_end_effector_poses',
            self.target_poses_callback,
            10
        )
        
        # Store current joint states
        self.current_joint_states = None
        self.joint_state_lock = threading.Lock()
        
        # IK solver parameters
        self.max_iterations = 100
        self.position_tolerance = 0.05  # 5cm
        self.orientation_tolerance = 0.1  # radians
        self.damping_factor = 0.01
        
        # Base movement parameters
        self.base_xy_tolerance = 0.05  # 5cm tolerance for base positioning
        self.max_arm_reach_xy = 0.8  # Maximum XY reach of arms from base (adjust based on your robot)
        self.base_movement_gain = 0.3  # Proportional gain for base movement
        self.max_base_velocity = 0.2  # Maximum base velocity (m/s)
        self.base_position = [0.0, 0.0]  # Current estimated base position in world frame
        
        # Joint limits (in radians) - adjust these based on your robot's specifications
        self.joint_limits = {
            'left': [
                (-np.pi, np.pi),    # LeftARMJoint1
                (-np.pi/2, np.pi/2), # LeftARMJoint02
                (-np.pi, np.pi),    # LeftARMJoint03
                (-np.pi, np.pi)     # LeftEndJoint
            ],
            'right': [
                (-np.pi, np.pi),    # RightARMJoint1
                (-np.pi/2, np.pi/2), # RihgtARMJoint2
                (-np.pi, np.pi),    # RightARMJoint3
                (-np.pi, np.pi)     # RightEndJOINT
            ]
        }
        
        # Complete joint names as per your robot structure
        self.all_joint_names = [
            'leftbackwheel01',
            'leftbackorient01',
            'Rightbackorient-1',
            'Rightbackwheel-1',
            'LeftFrontWheelOrient2-1',
            'LeftFrontWheelOrient1-1',
            'LeftFrontwheel-1',
            'RightFrontWheelOrient2-1',
            'RightFrontWheelOrient01-1',
            'RightFrontWheel',
            'LeftARMJoint1',
            'LeftARMJoint02', 
            'LeftARMJoint03',
            'LeftEndJoint',
            'RightARMJoint1',
            'RihgtARMJoint2',
            'RightARMJoint3',
            'RightEndJOINT'
        ]
        
        # Initialize joint positions (all zeros initially)
        self.current_positions = [0.0] * len(self.all_joint_names)
        
        # Arm joint indices in the complete joint array
        self.left_arm_indices = [10, 11, 12, 13]   # LeftARMJoint1, LeftARMJoint02, LeftARMJoint03, LeftEndJoint
        self.right_arm_indices = [14, 15, 16, 17]  # RightARMJoint1, RihgtARMJoint2, RightARMJoint3, RightEndJOINT
        
        # Wheel joint indices for base movement
        self.wheel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # All wheel joints
        
        # Base arm positions relative to robot base (adjust based on your robot's geometry)
        self.left_arm_base_offset = [0.0, -0.195]   # Left arm base position relative to robot center
        self.right_arm_base_offset = [0.0, 0.195]   # Right arm base position relative to robot center
        
        self.get_logger().info('Inverse Kinematics Controller with Base Movement initialized')

    def joint_state_callback(self, msg):
        """Store the latest joint states"""
        with self.joint_state_lock:
            self.current_joint_states = msg

    def target_poses_callback(self, msg):
        """Handle target poses and solve IK with base movement if needed"""
        if len(msg.poses) < 2:
            self.get_logger().warn("Expected 2 target poses (left and right arms)")
            return
            
        left_target = msg.poses[0]
        right_target = msg.poses[1]
        
        # Get current joint positions
        current_left, current_right = self.get_current_arm_positions()
        if current_left is None or current_right is None:
            self.get_logger().warn("No current joint states available, using current stored positions")
            current_left = [self.current_positions[i] for i in self.left_arm_indices]
            current_right = [self.current_positions[i] for i in self.right_arm_indices]
        
        # First, try to solve IK without base movement
        left_solution = self.solve_ik_arm(left_target, current_left, 'left')
        right_solution = self.solve_ik_arm(right_target, current_right, 'right')
        
        # Check if we need base movement
        base_movement_needed = False
        base_command = Twist()
        
        if left_solution is None or right_solution is None:
            # Calculate required base movement
            base_movement_needed, base_command = self.calculate_base_movement(left_target, right_target)
            
            if base_movement_needed:
                self.get_logger().info(
                    f"Target poses out of reach, commanding base movement: "
                    f"linear.x={base_command.linear.x:.3f}, linear.y={base_command.linear.y:.3f}"
                )
                self.cmd_vel_pub.publish(base_command)
                
                # Update estimated base position (simplified odometry)
                self.base_position[0] += base_command.linear.x * 0.1  # Assume 10Hz update rate
                self.base_position[1] += base_command.linear.y * 0.1
                return  # Don't try IK again immediately, wait for base to move
        
        # If both solutions found or no base movement needed, publish joint states
        if left_solution is not None and right_solution is not None:
            self.publish_joint_states(left_solution, right_solution)
            # Stop base movement if it was previously moving
            if not base_movement_needed:
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
        else:
            self.get_logger().warn("Failed to find IK solution for one or both arms even after base movement consideration")

    def calculate_base_movement(self, left_target, right_target):
        """Calculate required base movement to bring targets within reach"""
        # Get target positions in XY plane
        left_target_xy = [left_target.position.x, left_target.position.y]
        right_target_xy = [right_target.position.x, right_target.position.y]
        
        # Calculate distances from current base position to targets
        left_distance = math.sqrt(
            (left_target_xy[0] - self.base_position[0] - self.left_arm_base_offset[0])**2 +
            (left_target_xy[1] - self.base_position[1] - self.left_arm_base_offset[1])**2
        )
        
        right_distance = math.sqrt(
            (right_target_xy[0] - self.base_position[0] - self.right_arm_base_offset[0])**2 +
            (right_target_xy[1] - self.base_position[1] - self.right_arm_base_offset[1])**2
        )
        
        # Check if targets are within reach
        left_reachable = left_distance <= self.max_arm_reach_xy
        right_reachable = right_distance <= self.max_arm_reach_xy
        
        if left_reachable and right_reachable:
            return False, Twist()  # No base movement needed
        
        # Calculate desired base position to minimize maximum distance to both targets
        desired_base_pos = self.calculate_optimal_base_position(left_target_xy, right_target_xy)
        
        # Calculate movement vector
        movement_x = desired_base_pos[0] - self.base_position[0]
        movement_y = desired_base_pos[1] - self.base_position[1]
        
        # Apply proportional control with velocity limits
        cmd_vel = Twist()
        cmd_vel.linear.x = max(-self.max_base_velocity, 
                             min(self.max_base_velocity, movement_x * self.base_movement_gain))
        cmd_vel.linear.y = max(-self.max_base_velocity, 
                             min(self.max_base_velocity, movement_y * self.base_movement_gain))
        
        # Check if movement is significant enough
        movement_magnitude = math.sqrt(movement_x**2 + movement_y**2)
        if movement_magnitude < self.base_xy_tolerance:
            return False, Twist()  # Close enough, no movement needed
        
        return True, cmd_vel

    def calculate_optimal_base_position(self, left_target_xy, right_target_xy):
        """Calculate optimal base position to reach both targets"""
        # Simple approach: position base at the midpoint between targets, 
        # adjusted for arm base offsets
        left_adjusted = [
            left_target_xy[0] - self.left_arm_base_offset[0],
            left_target_xy[1] - self.left_arm_base_offset[1]
        ]
        right_adjusted = [
            right_target_xy[0] - self.right_arm_base_offset[0], 
            right_target_xy[1] - self.right_arm_base_offset[1]
        ]
        
        # Calculate midpoint
        optimal_x = (left_adjusted[0] + right_adjusted[0]) / 2.0
        optimal_y = (left_adjusted[1] + right_adjusted[1]) / 2.0
        
        return [optimal_x, optimal_y]

    def check_xy_reachability(self, target_pose, arm_type):
        """Check if target pose is reachable in XY plane from current base position"""
        target_xy = [target_pose.position.x, target_pose.position.y]
        
        if arm_type == 'left':
            arm_base = self.left_arm_base_offset
        else:
            arm_base = self.right_arm_base_offset
        
        # Calculate distance from arm base to target
        distance = math.sqrt(
            (target_xy[0] - self.base_position[0] - arm_base[0])**2 +
            (target_xy[1] - self.base_position[1] - arm_base[1])**2
        )
        
        return distance <= self.max_arm_reach_xy

    def get_current_arm_positions(self):
        """Get current arm joint positions"""
        with self.joint_state_lock:
            if self.current_joint_states is None:
                return None, None
                
            if len(self.current_joint_states.position) < 18:
                return None, None
                
            left_positions = [
                self.current_joint_states.position[10],  # LeftARMJoint1
                self.current_joint_states.position[11],  # LeftARMJoint02
                self.current_joint_states.position[12],  # LeftARMJoint03
                self.current_joint_states.position[13]   # LeftEndJoint
            ]
            
            right_positions = [
                self.current_joint_states.position[14],  # RightARMJoint1
                self.current_joint_states.position[15],  # RihgtARMJoint2
                self.current_joint_states.position[16],  # RightARMJoint3
                self.current_joint_states.position[17]   # RightEndJOINT
            ]
            
            return left_positions, right_positions

    def pose_to_transform_matrix(self, pose):
        """Convert Pose message to 4x4 transformation matrix"""
        # Extract position
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z
        
        # Extract quaternion and convert to rotation matrix
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w
        
        # Normalize quaternion
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm > 0:
            qx /= norm
            qy /= norm
            qz /= norm
            qw /= norm
        
        # Convert quaternion to rotation matrix
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])
        
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        
        return T

    def calculate_transform_matrix(self, translation, rotation_rpy):
        """Calculate 4x4 transformation matrix from translation and RPY rotation"""
        x, y, z = translation
        roll, pitch, yaw = rotation_rpy
        
        # Create rotation matrix from RPY
        rot_matrix = self.euler_to_rotation_matrix(roll, pitch, yaw)
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = [x, y, z]
        
        return transform

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """Convert Euler angles (RPY) to rotation matrix"""
        # Roll (rotation around x-axis)
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(roll), -math.sin(roll)],
                        [0, math.sin(roll), math.cos(roll)]])
        
        # Pitch (rotation around y-axis)
        R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                        [0, 1, 0],
                        [-math.sin(pitch), 0, math.cos(pitch)]])
        
        # Yaw (rotation around z-axis)
        R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                        [math.sin(yaw), math.cos(yaw), 0],
                        [0, 0, 1]])
        
        # Combined rotation matrix (R = R_z * R_y * R_x)
        R = R_z @ R_y @ R_x
        return R

    def calculate_left_arm_fk(self, joint_positions):
        """Calculate forward kinematics for left arm - copied from FK node"""
        if len(joint_positions) < 4:
            return None
            
        q1, q2, q3, q4 = joint_positions[:4]
        
        # Base to LeftARM1 (LeftARMJoint1)
        T01 = self.calculate_transform_matrix([0, -0.195, 1], [1.5708, 0, 0])
        R_q1 = np.array([[math.cos(q1), -math.sin(q1), 0, 0],
                         [math.sin(q1), math.cos(q1), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T01 = T01 @ R_q1
        
        # LeftARM1 to LeftARM02 (LeftARMJoint02)
        T12 = self.calculate_transform_matrix([0.12634, -0.0001172, 0.073], [3.1416, -1.5708, 3.1416])
        R_q2 = np.array([[math.cos(-q2), -math.sin(-q2), 0, 0],
                         [math.sin(-q2), math.cos(-q2), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T12 = T12 @ R_q2
        
        # LeftARM02 to LeftARM03 (LeftARMJoint03)
        T23 = self.calculate_transform_matrix([0.059067, -0.003961, 0.28311], [1.5708, 1.5708, 1.5708])
        R_q3 = np.array([[math.cos(q3), -math.sin(q3), 0, 0],
                         [math.sin(q3), math.cos(q3), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T23 = T23 @ R_q3
        
        # LeftARM03 to LeftEnd (LeftEndJoint)
        T34 = self.calculate_transform_matrix([-0.66222, 0.086205, -0.049114], [-1.5708, 0, 0])
        R_q4 = np.array([[math.cos(q4), -math.sin(q4), 0, 0],
                         [math.sin(q4), math.cos(q4), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T34 = T34 @ R_q4
        
        # Calculate final transformation
        T_final = T01 @ T12 @ T23 @ T34
        
        # Apply final 180 degree reverse rotation about Z-axis
        R_reverse = np.array([[math.cos(math.pi), -math.sin(math.pi), 0, 0],
                             [math.sin(math.pi), math.cos(math.pi), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        T_final = T_final @ R_reverse
        
        return T_final

    def calculate_right_arm_fk(self, joint_positions):
        """Calculate forward kinematics for right arm - copied from FK node"""
        if len(joint_positions) < 4:
            return None
            
        q1, q2, q3, q4 = joint_positions[:4]
        
        # Base to RightARM1 (RightARMJoint1)
        T01 = self.calculate_transform_matrix([0, 0.195, 1], [-1.5708, 0, 0])
        R_q1 = np.array([[math.cos(q1), -math.sin(q1), 0, 0],
                         [math.sin(q1), math.cos(q1), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T01 = T01 @ R_q1
        
        # RightARM1 to RihgtARM2 (RihgtARMJoint2)
        T12 = self.calculate_transform_matrix([0.126339132538937, 0.000117203456298176, 0.0729999999999077], 
                                            [1.5708, 1.5708, -1.5708])
        R_q2 = np.array([[math.cos(-q2), -math.sin(-q2), 0, 0],
                         [math.sin(-q2), math.cos(-q2), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T12 = T12 @ R_q2
        
        # RihgtARM2 to RightARM3 (RightARMJoint3)
        T23 = self.calculate_transform_matrix([0.058694, -0.0077252, 0.28311], [-1.5708, 1.5708, 1.5708])
        R_q3 = np.array([[math.cos(-q3), -math.sin(-q3), 0, 0],
                         [math.sin(-q3), math.cos(-q3), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T23 = T23 @ R_q3
        
        # RightARM3 to RightEnd (RightEndJOINT)
        T34 = self.calculate_transform_matrix([-0.66559, -0.054292, 0.049114], [1.5708, 1.5708, 0])
        R_q4 = np.array([[math.cos(q4), -math.sin(q4), 0, 0],
                         [math.sin(q4), math.cos(q4), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T34 = T34 @ R_q4
        
        # Calculate final transformation
        T_final = T01 @ T12 @ T23 @ T34
        
        # Apply final 90 degree rotation about Z-axis
        R_reverse = np.array([[math.cos(-math.pi/2), -math.sin(-math.pi/2), 0, 0],
                             [math.sin(-math.pi/2), math.cos(-math.pi/2), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        T_final = T_final @ R_reverse
        
        return T_final

    def calculate_pose_error(self, current_transform, target_transform):
        """Calculate position and orientation error between current and target poses"""
        # Position error
        pos_error = np.linalg.norm(current_transform[:3, 3] - target_transform[:3, 3])
        
        # Orientation error (using rotation matrix difference)
        R_current = current_transform[:3, :3]
        R_target = target_transform[:3, :3]
        R_error = R_target @ R_current.T
        
        # Convert rotation matrix error to angle
        trace = np.trace(R_error)
        angle_error = math.acos(min(1.0, max(-1.0, (trace - 1) / 2)))
        
        return pos_error, angle_error

    def ik_cost_function(self, joint_positions, target_transform, arm_type):
        """Cost function for IK optimization"""
        # Calculate forward kinematics
        if arm_type == 'left':
            current_transform = self.calculate_left_arm_fk(joint_positions)
        else:
            current_transform = self.calculate_right_arm_fk(joint_positions)
            
        if current_transform is None:
            return 1e6  # Large penalty for invalid solutions
        
        # Calculate errors
        pos_error, ori_error = self.calculate_pose_error(current_transform, target_transform)
        
        # Combined cost with weights
        position_weight = 1000.0
        orientation_weight = 100.0
        
        cost = position_weight * pos_error**2 + orientation_weight * ori_error**2
        
        # Add penalty for joint limit violations
        limits = self.joint_limits[arm_type]
        for i, (q, (q_min, q_max)) in enumerate(zip(joint_positions, limits)):
            if q < q_min:
                cost += 1000 * (q_min - q)**2
            elif q > q_max:
                cost += 1000 * (q - q_max)**2
        
        return cost

    def solve_ik_arm(self, target_pose, initial_guess, arm_type):
        """Solve IK for a single arm using numerical optimization"""
        target_transform = self.pose_to_transform_matrix(target_pose)
        
        # Set up optimization bounds
        bounds = self.joint_limits[arm_type]
        
        # Try multiple optimization attempts with different initial guesses
        best_solution = None
        best_cost = float('inf')
        
        # First try with current position as initial guess
        attempts = [
            initial_guess,
            [0.0, 0.0, 0.0, 0.0],  # Zero position
            np.random.uniform(-0.5, 0.5, 4).tolist(),  # Small random
            np.random.uniform(-1.0, 1.0, 4).tolist()   # Larger random
        ]
        
        for attempt, init_guess in enumerate(attempts):
            try:
                result = minimize(
                    self.ik_cost_function,
                    init_guess,
                    args=(target_transform, arm_type),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': self.max_iterations,
                        'ftol': 1e-9,
                        'gtol': 1e-9
                    }
                )
                
                if result.success and result.fun < best_cost:
                    # Verify the solution meets tolerance requirements
                    if arm_type == 'left':
                        current_transform = self.calculate_left_arm_fk(result.x)
                    else:
                        current_transform = self.calculate_right_arm_fk(result.x)
                    
                    if current_transform is not None:
                        pos_error, ori_error = self.calculate_pose_error(current_transform, target_transform)
                        
                        if pos_error < self.position_tolerance and ori_error < self.orientation_tolerance:
                            best_solution = result.x
                            best_cost = result.fun
                            self.get_logger().info(
                                f"{arm_type.capitalize()} arm IK solved (attempt {attempt+1}): "
                                f"pos_error={pos_error:.4f}m, ori_error={ori_error:.4f}rad"
                            )
                            break
                        
            except Exception as e:
                self.get_logger().warn(f"IK optimization attempt {attempt+1} failed: {str(e)}")
                continue
        
        if best_solution is not None:
            return best_solution.tolist()
        else:
            self.get_logger().warn(f"Failed to find valid IK solution for {arm_type} arm")
            return None

    def publish_joint_states(self, left_joint_positions, right_joint_positions):
        """Publish complete joint states for the entire robot"""
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.header.frame_id = ''
        
        # Update only the arm joint positions in the current positions array
        for i, pos in enumerate(left_joint_positions):
            self.current_positions[self.left_arm_indices[i]] = pos
        
        for i, pos in enumerate(right_joint_positions):
            self.current_positions[self.right_arm_indices[i]] = pos
        
        # Set complete joint state message
        joint_state.name = self.all_joint_names
        joint_state.position = self.current_positions.copy()
        joint_state.velocity = []  # Empty as per your example
        joint_state.effort = []    # Empty as per your example
        
        self.joint_states_pub.publish(joint_state)
        
        self.get_logger().info(
            f"Published joint states - "
            f"Left arm: [{', '.join([f'{pos:.3f}' for pos in left_joint_positions])}], "
            f"Right arm: [{', '.join([f'{pos:.3f}' for pos in right_joint_positions])}]"
        )


def main(args=None):
    rclpy.init(args=args)
    
    ik_controller = InverseKinematicsController()
    
    try:
        rclpy.spin(ik_controller)
    except KeyboardInterrupt:
        pass
    finally:
        ik_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()