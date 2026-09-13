cmake_minimum_required(VERSION 3.10)
project(pgo)
set(CMAKE_BUILD_TYPE Release)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖项
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

# 包含目录
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

# 添加可执行文件
add_executable(pgo pgo.cpp)

# 链接库
target_link_libraries(pgo ${CERES_LIBRARIES})

# 添加可执行文件
add_executable(pgo_relative pgo_relative.cpp)

# 链接库
target_link_libraries(pgo_relative ${CERES_LIBRARIES})