cmake_minimum_required(VERSION 3.16)
project(GiantHeuristic CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 优化编译选项
if(MSVC)
    add_compile_options(/O2 /W3)
else()
    add_compile_options(-O2 -Wall -Wextra -Wno-unused-parameter)
endif()

# 头文件目录
include_directories(include)
include_directories(third_party)

# 收集所有源文件
file(GLOB_RECURSE SOURCES "src/*.cpp")

# 可执行文件（跨平台不加 .exe，CMake 自动处理）
add_executable(giant_heuristic ${SOURCES})

# 链接线程库（多线程并行支持）
find_package(Threads REQUIRED)
target_link_libraries(giant_heuristic Threads::Threads)

# 链接数学库（Linux 需要）
if(UNIX AND NOT APPLE)
    target_link_libraries(giant_heuristic m)
endif()

# 输出信息
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Sources: ${SOURCES}")
