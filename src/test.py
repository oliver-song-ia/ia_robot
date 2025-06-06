#!/usr/bin/env python3
import rclpy, math, threading, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose          # ⬅ pose message
from std_msgs.msg import Float32MultiArray

class InverseKinematicsController(Node):
    def __init__(self):
        super().__init__('ik_controller')

        # ─── Publishers ──────────────────────────────────────────────────────────
        self.joint_states_pub  = self.create_publisher(JointState, '/joint_states', 10)
        self.joint_command_pub = self.create_publisher(JointState, '/joint_command', 10)
        self.robot_pose_pub    = self.create_publisher(Pose,        '/robot_pose',   10)

        # ─── Target subscriber ───────────────────────────────────────────────────
        self.create_subscription(Float32MultiArray,
                                 '/target_midpoint_width',
                                 self.target_callback,
                                 10)

        # ─── Robot parameters (unchanged) ────────────────────────────────────────
        self.base_movement_gain = 0.3
        self.base_tolerance     = 0.01

        self.L1, self.L2, self.L3 = 0.12634, 0.28311, 0.66222
        self.left_joint2_fixed, self.right_joint2_fixed = math.pi/2, -math.pi/2

        self.all_joint_names = [
            'LeftARMJoint1', 'LeftFrontWheelOrient2_1', 'RightARMJoint1',
            'RightFrontWheelOrient2_1', 'Rightbackorient_1', 'leftbackorient01',
            'LeftARMJoint02', 'LeftFrontWheelOrient1_1', 'RihgtARMJoint2',
            'RightFrontWheelOrient01_1', 'Rightbackwheel_1', 'leftbackwheel01',
            'LeftARMJoint03', 'LeftFrontwheel_1', 'RightARMJoint3',
            'RightFrontWheel', 'LeftEndJoint', 'RightEndJOINT'
        ]

        self.current_positions = [0.0] * len(self.all_joint_names)
        self.zero_vec          = [0.0] * len(self.all_joint_names)
        self.left_arm_idx  = [0, 6, 12, 16]
        self.right_arm_idx = [2, 8, 14, 17]

        self.get_logger().info('IK Controller ready – publishing /joint_command & /robot_pose')

    # ─────────────────────────────────────────────────────────────────────────────
    def target_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 4:
            self.get_logger().warn('Need [x, y, z, width]')
            return
        # --- inside target_callback() -----------------------------------------------
        tx, ty, tz, width = msg.data[0], msg.data[1], msg.data[2], msg.data[3] - 0.5

        base_move, arm_angles = self.solve_ik(tx, ty, tz, width)
        if arm_angles is None:
            self.get_logger().warn('IK failed')
            return

        # Δx, Δy are already relative to the current base position
        pose = Pose()
        pose.position.x = base_move[0]
        pose.position.y = base_move[1]
        print('x movement =', base_move[0])
        print('y movement =', base_move[1])
        pose.position.z = 0.0
        pose.orientation.w = 1.0                # identity quaternion
        pose.orientation.x = pose.orientation.y = pose.orientation.z = 0.0
        self.robot_pose_pub.publish(pose)

        # finally push the arm command
        self.publish_joint_command(arm_angles)
        # ----------------------------------------------------------------------------


    # IK math (unchanged) ---------------------------------------------------------
    def solve_ik(self, x, y, z, width):
        lateral = max(0.0, width/2.0)
        j3_mag  = math.asin(min(1.0, lateral/(self.L2+self.L3))) if width>0 else 0.0
        l_j3 = r_j3 = -j3_mag
        height_target = z - 0.6
        vertical_j3   = self.L2*math.cos(j3_mag)
        h_from_j1     = height_target - self.L1 - vertical_j3
        j1 = math.asin(max(-1.0, min(1.0, h_from_j1/self.L3))) if abs(h_from_j1) < self.L3 else 0.0
        l_j1, r_j1 = -j1, j1
        l_end = -math.pi/2 - l_j1
        r_end =  math.pi/2 - r_j1
        arm_angles = [l_j1, self.left_joint2_fixed, l_j3, l_end,
                      r_j1, self.right_joint2_fixed, r_j3, r_end]
        return (x, y), arm_angles

    # Publish joint command -------------------------------------------------------
    def publish_joint_command(self, arm_angles):
        for i in range(4):
            self.current_positions[self.left_arm_idx[i]]  = arm_angles[i]
            self.current_positions[self.right_arm_idx[i]] = arm_angles[i+4]
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name, js.position = self.all_joint_names, self.current_positions
        js.velocity, js.effort = self.zero_vec, self.zero_vec
        self.joint_command_pub.publish(js)
        self.joint_states_pub.publish(js)

# Boilerplate -------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematicsController()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
