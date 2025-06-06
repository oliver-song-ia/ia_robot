cmake_minimum_required(VERSION 3.8)
project(ia_robot)

# enforce pedantic warnings on GCC/Clang
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find ament and any other ROS 2 packages you need
find_package(ament_cmake REQUIRED)
# e.g. find_package(rclcpp REQUIRED)
# find_package(robot_state_publisher REQUIRED)
# find_package(joint_state_publisher_gui REQUIRED)
# find_package(rviz2 REQUIRED)

# install your URDFs, meshes, configs, launch files, RViz configs, etc.
install(
  DIRECTORY urdf
            meshes
            launch
            config
            rviz
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  src/fk.py
  src/ik.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # skip copyright/cpplint checks until you add appropriate headers
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

