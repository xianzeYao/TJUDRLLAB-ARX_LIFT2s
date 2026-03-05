cmake_minimum_required(VERSION 3.5)
project(lift2)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch meshes textures urdf
  DESTINATION share/${PROJECT_NAME}
  PATTERN ".DS_Store" EXCLUDE)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
