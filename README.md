## Git clone to ros2 workspace src

## Build workspace
```
colcon build --packages-select ia_robot
source install/setup.bash
```

## Play with joints
```
ros2 launch ia_robot view_all_joints.launch.py
```