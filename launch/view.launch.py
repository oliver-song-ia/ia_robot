import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Locate this package's share directory
    pkg_share = get_package_share_directory('ia_robot')

    # URDF file path
    default_urdf = os.path.join(pkg_share, 'urdf', 'ia_robot_new.urdf')
    urdf_arg = DeclareLaunchArgument(
        'urdf_file', default_value=default_urdf,
        description='Full path to URDF file'
    )

    # Joint states topic argument
    joint_states_topic_arg = DeclareLaunchArgument(
        'joint_states_topic', 
        default_value='/joint_states',
        description='Topic name for joint states'
    )

    # Read URDF XML
    urdf_path = LaunchConfiguration('urdf_file')
    with open(default_urdf, 'r') as inf:
        robot_description_content = inf.read()

    # RViz config (optional)
    rviz_config = os.path.join(pkg_share, 'rviz', 'view.rviz')

    return LaunchDescription([
        urdf_arg,
        joint_states_topic_arg,

        # Joint State Publisher (subscribes to joint_states topic)
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen',
            parameters=[{
                'source_list': [LaunchConfiguration('joint_states_topic')]
            }]
        ),

        # Robot State Publisher to publish transforms
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description_content}]
        ),

        # RViz2 for visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config]
        ),
    ])