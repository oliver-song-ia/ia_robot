#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import numpy as np
import math

class ForwardKinematicsPublisher(Node):
    def __init__(self):
        super().__init__('fk_publisher')
        
        # Create TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.pose_array_pub = self.create_publisher(
            PoseArray, 
            '/ia_robot/end_effector_poses', 
            10
        )
        
        # Subscriber to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Timer for publishing at regular intervals
        self.timer = self.create_timer(0.1, self.publish_end_effector_poses)  # 10 Hz
        
        # Store latest joint states
        self.latest_joint_states = None
        
        # Define the kinematic chains for both arms
        self.left_arm_joints = [
            'LeftARMJoint1',
            'LeftARMJoint02', 
            'LeftARMJoint03',
            'LeftEndJoint'
        ]
        
        self.right_arm_joints = [
            'RightARMJoint1',
            'RihgtARMJoint2',  # Note: keeping original spelling from URDF
            'RightARMJoint3',
            'RightEndJOINT'   # Note: keeping original spelling from URDF
        ]
        
        self.get_logger().info('Forward Kinematics Publisher initialized')

    def joint_state_callback(self, msg):
        """Store the latest joint states"""
        self.latest_joint_states = msg

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

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion [x, y, z, w]"""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        
        return [qx, qy, qz, qw]

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

    def calculate_left_arm_fk(self, joint_positions):
        """Calculate forward kinematics for left arm"""
        # Extract joint positions (assuming they are in the same order as defined)
        if len(joint_positions) < 4:
            self.get_logger().warn("Not enough joint positions for left arm FK")
            return None
            
        q1, q2, q3, q4 = joint_positions[:4]
        
        # Base to LeftARM1 (LeftARMJoint1)
        # Origin: xyz="0 -0.195 1" rpy="1.5708 0 0"
        T01 = self.calculate_transform_matrix([0, -0.195, 1], [1.5708, 0, 0])
        # Rotation about z-axis by q1 (axis="0 0 1")
        R_q1 = np.array([[math.cos(q1), -math.sin(q1), 0, 0],
                         [math.sin(q1), math.cos(q1), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T01 = T01 @ R_q1
        
        # LeftARM1 to LeftARM02 (LeftARMJoint02)
        # Origin: xyz="0.12634 -0.0001172 0.073" rpy="3.1416 -1.5708 3.1416"
        # Axis: xyz="0 0 -1" (negative z-axis)
        T12 = self.calculate_transform_matrix([0.12634, -0.0001172, 0.073], [3.1416, -1.5708, 3.1416])
        R_q2 = np.array([[math.cos(-q2), -math.sin(-q2), 0, 0],
                         [math.sin(-q2), math.cos(-q2), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T12 = T12 @ R_q2
        
        # LeftARM02 to LeftARM03 (LeftARMJoint03)
        # Origin: xyz="0.059067 -0.003961 0.28311" rpy="1.5708 1.5708 1.5708"
        # Axis: xyz="0 0 1"
        T23 = self.calculate_transform_matrix([0.059067, -0.003961, 0.28311], [1.5708, 1.5708, 1.5708])
        R_q3 = np.array([[math.cos(q3), -math.sin(q3), 0, 0],
                         [math.sin(q3), math.cos(q3), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T23 = T23 @ R_q3
        
        # LeftARM03 to LeftEnd (LeftEndJoint)
        # Origin: xyz="-0.66222 0.086205 -0.049114" rpy="-1.5708 0 0"
        # Axis: xyz="0 0 1"
        T34 = self.calculate_transform_matrix([-0.66222, 0.086205, -0.049114], [-1.5708, 0, 0])
        R_q4 = np.array([[math.cos(q4), -math.sin(q4), 0, 0],
                         [math.sin(q4), math.cos(q4), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T34 = T34 @ R_q4
        
        # Calculate final transformation: Base -> LeftEnd
        T_final = T01 @ T12 @ T23 @ T34
        
        # Apply final 180 degree reverse rotation about Z-axis for end effector orientation
        R_reverse = np.array([[math.cos(math.pi), -math.sin(math.pi), 0, 0],
                             [math.sin(math.pi), math.cos(math.pi), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        T_final = T_final @ R_reverse
        
        return T_final

    def calculate_right_arm_fk(self, joint_positions):
        """Calculate forward kinematics for right arm"""
        if len(joint_positions) < 4:
            self.get_logger().warn("Not enough joint positions for right arm FK")
            return None
            
        q1, q2, q3, q4 = joint_positions[:4]
        
        # Base to RightARM1 (RightARMJoint1)
        # Origin: xyz="0 0.195 1" rpy="-1.5708 0 0"
        # Axis: xyz="0 0 1"
        T01 = self.calculate_transform_matrix([0, 0.195, 1], [-1.5708, 0, 0])
        R_q1 = np.array([[math.cos(q1), -math.sin(q1), 0, 0],
                         [math.sin(q1), math.cos(q1), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T01 = T01 @ R_q1
        
        # RightARM1 to RihgtARM2 (RihgtARMJoint2)
        # Origin: xyz="0.126339132538937 0.000117203456298176 0.0729999999999077" rpy="1.5708 -1.5708 -1.5708"
        # Axis: xyz="0 0 -1" (negative z-axis)
        T12 = self.calculate_transform_matrix([0.126339132538937, 0.000117203456298176, 0.0729999999999077], 
                                            [1.5708, 1.5708, -1.5708])
        R_q2 = np.array([[math.cos(-q2), -math.sin(-q2), 0, 0],
                         [math.sin(-q2), math.cos(-q2), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T12 = T12 @ R_q2
        
        # RihgtARM2 to RightARM3 (RightARMJoint3)
        # Origin: xyz="0.058694 -0.0077252 0.28311" rpy="-1.5708 1.5708 1.5708"
        # Axis: xyz="0 0 -1" (negative z-axis)
        T23 = self.calculate_transform_matrix([0.058694, -0.0077252, 0.28311], [-1.5708, 1.5708, 1.5708])
        R_q3 = np.array([[math.cos(-q3), -math.sin(-q3), 0, 0],
                         [math.sin(-q3), math.cos(-q3), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T23 = T23 @ R_q3
        
        # RightARM3 to RightEnd (RightEndJOINT)
        # Origin: xyz="-0.66559 -0.054292 0.049114" rpy="1.5708 1.5708 0"
        # Axis: xyz="0 0 1"
        T34 = self.calculate_transform_matrix([-0.66559, -0.054292, 0.049114], [1.5708, 1.5708, 0])
        R_q4 = np.array([[math.cos(q4), -math.sin(q4), 0, 0],
                         [math.sin(q4), math.cos(q4), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T34 = T34 @ R_q4
        
        # Calculate final transformation: Base -> RightEnd
        T_final = T01 @ T12 @ T23 @ T34
        
        # Apply final 90 degree rotation about Z-axis for end effector orientation
        R_reverse = np.array([[math.cos(-math.pi/2), -math.sin(-math.pi/2), 0, 0],
                             [math.sin(-math.pi/2), math.cos(-math.pi/2), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        T_final = T_final @ R_reverse
        
        return T_final

    def transform_matrix_to_pose(self, transform_matrix):
        """Convert 4x4 transformation matrix to Pose message"""
        pose = Pose()
        
        # Extract position
        pose.position.x = float(transform_matrix[0, 3])
        pose.position.y = float(transform_matrix[1, 3])
        pose.position.z = float(transform_matrix[2, 3])
        
        # Extract rotation and convert to quaternion
        rotation_matrix = transform_matrix[:3, :3]
        quat = self.rotation_matrix_to_quaternion(rotation_matrix)  # Returns [x, y, z, w]
        
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        
        return pose

    def get_joint_positions(self, joint_names):
        """Extract joint positions for given joint names from latest joint states"""
        if self.latest_joint_states is None:
            return None
            
        positions = []
        for joint_name in joint_names:
            try:
                idx = self.latest_joint_states.name.index(joint_name)
                positions.append(self.latest_joint_states.position[idx])
            except ValueError:
                self.get_logger().warn(f"Joint {joint_name} not found in joint states")
                positions.append(0.0)  # Default to 0 if joint not found
                
        return positions

    def get_arm_joint_positions_optimized(self):
        """Extract arm joint positions using known indices for efficiency"""
        if self.latest_joint_states is None:
            return None, None
            
        # Based on the joint_states format you provided:
        # Index 10: LeftARMJoint1
        # Index 11: LeftARMJoint02  
        # Index 12: LeftARMJoint03
        # Index 13: LeftEndJoint
        # Index 14: RightARMJoint1
        # Index 15: RihgtARMJoint2
        # Index 16: RightARMJoint3
        # Index 17: RightEndJOINT
        
        if len(self.latest_joint_states.position) < 18:
            self.get_logger().warn("Not enough joint positions in joint_states")
            return None, None
            
        left_positions = [
            self.latest_joint_states.position[10],  # LeftARMJoint1
            self.latest_joint_states.position[11],  # LeftARMJoint02
            self.latest_joint_states.position[12],  # LeftARMJoint03
            self.latest_joint_states.position[13]   # LeftEndJoint
        ]
        
        right_positions = [
            self.latest_joint_states.position[14],  # RightARMJoint1
            self.latest_joint_states.position[15],  # RihgtARMJoint2
            self.latest_joint_states.position[16],  # RightARMJoint3
            self.latest_joint_states.position[17]   # RightEndJOINT
        ]
        
        return left_positions, right_positions

    def publish_end_effector_poses(self):
        """Calculate and publish end effector poses"""
        if self.latest_joint_states is None:
            return
            
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'base_link'
        
        # Get joint positions using optimized method
        left_joint_positions, right_joint_positions = self.get_arm_joint_positions_optimized()
        
        if left_joint_positions is None or right_joint_positions is None:
            self.get_logger().warn("Failed to get arm joint positions")
            return
        
        # Calculate left arm FK
        left_transform = self.calculate_left_arm_fk(left_joint_positions)
        if left_transform is not None:
            left_pose = self.transform_matrix_to_pose(left_transform)
            pose_array.poses.append(left_pose)
        
        # Calculate right arm FK
        right_transform = self.calculate_right_arm_fk(right_joint_positions)
        if right_transform is not None:
            right_pose = self.transform_matrix_to_pose(right_transform)
            pose_array.poses.append(right_pose)
        
        # Publish pose array
        if len(pose_array.poses) > 0:
            self.pose_array_pub.publish(pose_array)
            
            # Log poses for debugging (only occasionally to avoid spam)
            if self.get_clock().now().nanoseconds % 1000000000 < 100000000:  # Every ~1 second
                for i, pose in enumerate(pose_array.poses):
                    arm_name = "Left" if i == 0 else "Right"
                    self.get_logger().info(
                        f"{arm_name} End Effector - "
                        f"Pos: [{pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}], "
                        f"Ori: [{pose.orientation.x:.3f}, {pose.orientation.y:.3f}, "
                        f"{pose.orientation.z:.3f}, {pose.orientation.w:.3f}]"
                    )
                    
                # Also log the joint positions for debugging
                self.get_logger().info(
                    f"Left arm joints: {[f'{pos:.3f}' for pos in left_joint_positions]}"
                )
                self.get_logger().info(
                    f"Right arm joints: {[f'{pos:.3f}' for pos in right_joint_positions]}"
                )


def main(args=None):
    rclpy.init(args=args)
    
    fk_publisher = ForwardKinematicsPublisher()
    
    try:
        rclpy.spin(fk_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        fk_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()