#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import tkinter as tk
from tkinter import ttk
import threading

class JointControllerGUI(Node):
    def __init__(self):
        super().__init__('joint_controller_gui')
        self.pub = self.create_publisher(JointState, '/joint_command', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

        # Joint names
        self.names = [
            'LeftARMJoint1',
            'LeftFrontWheelOrient2_1',
            'RightARMJoint1',
            'RightFrontWheelOrient2_1',
            'Rightbackorient_1',
            'leftbackorient01',
            'LeftARMJoint02',
            'LeftFrontWheelOrient1_1',
            'RihgtARMJoint2',
            'RightFrontWheelOrient01_1',
            'Rightbackwheel_1',
            'leftbackwheel01',
            'LeftARMJoint03',
            'LeftFrontwheel_1',
            'RightARMJoint3',
            'RightFrontWheel',
            'LeftEndJoint',
            'RightEndJOINT',
        ]

        # Initialize joint positions
        self.joint_positions = [0.0] * len(self.names)
        
        # Create GUI
        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Joint Controller")
        self.root.geometry("600x800")
        
        # Create main frame with scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create sliders for each joint
        self.sliders = []
        self.value_labels = []
        
        for i, joint_name in enumerate(self.names):
            # Frame for each joint
            joint_frame = ttk.Frame(scrollable_frame)
            joint_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Joint name label
            name_label = ttk.Label(joint_frame, text=joint_name, width=25)
            name_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # Value label
            value_label = ttk.Label(joint_frame, text="0.00", width=6)
            value_label.pack(side=tk.RIGHT, padx=(10, 0))
            self.value_labels.append(value_label)
            
            # Slider
            slider = tk.Scale(
                joint_frame,
                from_=-3.14159,  # -π radians
                to=3.14159,      # +π radians
                resolution=0.01,
                orient=tk.HORIZONTAL,
                length=300,
                command=lambda val, idx=i: self.update_joint_position(idx, val)
            )
            slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 10))
            slider.set(0.0)
            self.sliders.append(slider)
            
        # Control buttons frame
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Reset all button
        reset_button = ttk.Button(
            button_frame,
            text="Reset All to Zero",
            command=self.reset_all_joints
        )
        reset_button.pack(side=tk.LEFT, padx=5)
        
        # Randomize button
        randomize_button = ttk.Button(
            button_frame,
            text="Randomize",
            command=self.randomize_joints
        )
        randomize_button.pack(side=tk.LEFT, padx=5)
        
        # Bind mouse wheel to canvas for scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
    def update_joint_position(self, joint_index, value):
        """Update joint position when slider changes"""
        self.joint_positions[joint_index] = float(value)
        self.value_labels[joint_index].config(text=f"{float(value):.2f}")
        
    def reset_all_joints(self):
        """Reset all joints to zero position"""
        for i, slider in enumerate(self.sliders):
            slider.set(0.0)
            self.joint_positions[i] = 0.0
            self.value_labels[i].config(text="0.00")
            
    def randomize_joints(self):
        """Set random positions for all joints"""
        import random
        for i, slider in enumerate(self.sliders):
            random_val = random.uniform(-1.0, 1.0)
            slider.set(random_val)
            self.joint_positions[i] = random_val
            self.value_labels[i].config(text=f"{random_val:.2f}")

    def timer_callback(self):
        """Publish joint states at regular intervals"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.name = self.names
        msg.position = self.joint_positions[:]
        msg.velocity = [0.0] * len(self.names)
        msg.effort = [0.0] * len(self.names)
        
        self.pub.publish(msg)

    def run_gui(self):
        """Run the GUI main loop"""
        self.root.mainloop()


def ros_thread(node):
    """Run ROS2 spinning in separate thread"""
    try:
        rclpy.spin(node)
    except Exception:
        pass


def main(args=None):
    rclpy.init(args=args)
    node = JointControllerGUI()
    
    # Start ROS2 spinning in separate thread
    ros_thread_obj = threading.Thread(target=ros_thread, args=(node,))
    ros_thread_obj.daemon = True
    ros_thread_obj.start()
    
    try:
        # Run GUI in main thread
        node.run_gui()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()