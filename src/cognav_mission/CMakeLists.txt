cmake_minimum_required(VERSION 3.8)
project(cognav_mission)

find_package(ament_cmake REQUIRED)
find_package(action_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ExecuteMission.action"
)

install(DIRECTORY mission
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/mission_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
