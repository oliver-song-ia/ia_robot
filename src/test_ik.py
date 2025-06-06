#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
import numpy as np
import random
import math

class RandomPosePublisher(Node):
    def __init__(self):
        super().__init__('random_pose_publisher')
        
        # Publisher for target poses
        self.pose_publisher = self.create_publisher(
            PoseArray,
            '/target_end_effector_poses',
            10
        )
        
        # Timer to publish poses at regular intervals
        self.timer = self.create_timer(2.0, self.publish_random_poses)  # Publish every 2 seconds
        
        # Center positions for left and right arms
        self.left_center = {
            'position': [-0.819, -0.278, 1.082],
            'orientation': [0.000, 0.000, 1.000, 0.000]  # [x, y, z, w]
        }
        
        self.right_center = {
            'position': [-0.822, 0.258, 1.046],
            'orientation': [0.000, 0.000, 1.000, 0.000]  # [x, y, z, w]
        }
        
        # Random variation ranges
        self.position_variation = 0.1  # ±10cm variation in each axis
        self.orientation_variation = 0.3  # ±0.3 radians variation for orientation
        
        self.get_logger().info('Random Pose Publisher initialized')
        self.get_logger().info(f'Publishing random poses every 2 seconds')
        self.get_logger().info(f'Position variation: ±{self.position_variation}m')
        self.get_logger().info(f'Orientation variation: ±{self.orientation_variation} radians')

    def generate_random_position(self, center_pos):
        """Generate a random position around the center position"""
        return [
            center_pos[0] + random.uniform(-self.position_variation, self.position_variation),
            center_pos[1] + random.uniform(-self.position_variation, self.position_variation),
            center_pos[2] + random.uniform(-self.position_variation, self.position_variation)
        ]

    def generate_random_quaternion(self, center_quat):
        """Generate a random quaternion around the center orientation"""
        # Convert center quaternion to euler angles
        center_euler = self.quaternion_to_euler(center_quat)
        
        # Add random variations to euler angles
        random_euler = [
            center_euler[0] + random.uniform(-self.orientation_variation, self.orientation_variation),
            center_euler[1] + random.uniform(-self.orientation_variation, self.orientation_variation),
            center_euler[2] + random.uniform(-self.orientation_variation, self.orientation_variation)
        ]
        
        # Convert back to quaternion
        return self.euler_to_quaternion(random_euler)

    def quaternion_to_euler(self, quat):
        """Convert quaternion [x, y, z, w] to euler angles [roll, pitch, yaw]"""
        x, y, z, w = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return [roll, pitch, yaw]

    def euler_to_quaternion(self, euler):
        """Convert euler angles [roll, pitch, yaw] to quaternion [x, y, z, w]"""
        roll, pitch, yaw = euler
        
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return [x, y, z, w]

    def create_pose(self, position, orientation):
        """Create a Pose message from position and orientation arrays"""
        pose = Pose()
        
        # Set position
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        
        # Set orientation (quaternion)
        pose.orientation.x = orientation[0]
        pose.orientation.y = orientation[1]
        pose.orientation.z = orientation[2]
        pose.orientation.w = orientation[3]
        
        return pose

    def publish_random_poses(self):
        """Generate and publish random target poses"""
        # Generate random positions
        left_pos = self.generate_random_position(self.left_center['position'])
        right_pos = self.generate_random_position(self.right_center['position'])
        
        # Generate random orientations
        left_ori = self.generate_random_quaternion(self.left_center['orientation'])
        right_ori = self.generate_random_quaternion(self.right_center['orientation'])
        
        # Create pose messages
        left_pose = self.create_pose(left_pos, left_ori)
        right_pose = self.create_pose(right_pos, right_ori)
        
        # Create PoseArray message
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'base_link'  # Adjust frame_id as needed
        pose_array.poses = [left_pose, right_pose]
        
        # Publish the poses
        self.pose_publisher.publish(pose_array)
        
        # Log the published poses
        self.get_logger().info(
            f'Published random poses:\n'
            f'  Left:  Pos: [{left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f}], '
            f'Ori: [{left_ori[0]:.3f}, {left_ori[1]:.3f}, {left_ori[2]:.3f}, {left_ori[3]:.3f}]\n'
            f'  Right: Pos: [{right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f}], '
            f'Ori: [{right_ori[0]:.3f}, {right_ori[1]:.3f}, {right_ori[2]:.3f}, {right_ori[3]:.3f}]'
        )

    def set_variation_ranges(self, position_var, orientation_var):
        """Set the variation ranges for position and orientation"""
        self.position_variation = position_var
        self.orientation_variation = orientation_var
        self.get_logger().info(
            f'Updated variation ranges - Position: ±{position_var}m, Orientation: ±{orientation_var} rad'
        )


def main(args=None):
    rclpy.init(args=args)
    
    # Create the publisher node
    publisher = RandomPosePublisher()
    
    # You can adjust variation ranges if needed
    publisher.set_variation_ranges(0.05, 0.2)  # Smaller variations
    # publisher.set_variation_ranges(0.15, 0.5)  # Larger variations
    
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        publisher.get_logger().info('Shutting down Random Pose Publisher')
    finally:
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()