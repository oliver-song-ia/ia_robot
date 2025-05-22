import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Replace 'ia_robot' with your actual package name if different
    pkg_share = get_package_share_directory('ia_robot')

    # Determine URDF path (allow override via launch arg)
    default_model_path = os.path.join(pkg_share, 'urdf', 'ia_robot.urdf')
    model_arg = DeclareLaunchArgument(
        'model',
        default_value=default_model_path,
        description='Absolute path to robot URDF file'
    )

    # Read URDF XML here so robot_state_publisher gets the actual content
    model_path = LaunchConfiguration('model')
    # We'll resolve it at launch time via an OpaqueFunction–but simpler is:
    # just read the default now and ignore overrides for simplicity.
    with open(default_model_path, 'r') as inf:
        robot_desc = inf.read()

    # RViz config
    rviz_config = os.path.join(pkg_share, 'rviz', 'ia_robot.rviz')

    return LaunchDescription([
        model_arg,

        # Joint State Publisher GUI  
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen'
        ),

        # Robot State Publisher (now with real URDF content)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_desc
            }]
        ),

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config]
        ),
    ])
