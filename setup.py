from setuptools import setup

package_name = 'ia_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/view.launch.py']),
        ('share/' + package_name + '/launch', ['launch/view__all_joints.launch.py']),
        ('share/' + package_name + '/urdf', ['urdf/ia_robot.urdf']),
        ('share/' + package_name + '/rviz', ['rviz/view.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='IA Robot package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fk_publisher.py = ia_robot.fk_publisher:main',
            'ik_controller.py = ia_robot.ik_controller:main',
            'joint_command_controller.py = ia_robot.joint_command_controller:main',
        ],
    },
)