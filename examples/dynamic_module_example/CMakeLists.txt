# Example: Dynamic Module Loading with rocSHMEM
# This demonstrates why rocshmemx_hipmodule_init is needed for Triton

cmake_minimum_required(VERSION 3.16)
project(dynamic_module_example LANGUAGES CXX)

# Find rocSHMEM
find_package(rocshmem REQUIRED)
find_package(hip REQUIRED)

# Main executable that loads dynamic modules
add_executable(dynamic_load_example
    dynamic_load_example.cpp
)

target_link_libraries(dynamic_load_example
    PRIVATE
        roc::rocshmem
        hip::host
)

# Custom target to compile the kernel to HSACO
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel.hsaco
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/compile_module.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel.hip
            ${CMAKE_CURRENT_SOURCE_DIR}/compile_module.sh
    COMMENT "Compiling HIP kernel to dynamic module (HSACO)"
)

add_custom_target(compile_kernel ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernel.hsaco
)

add_dependencies(dynamic_load_example compile_kernel)

# Install
install(TARGETS dynamic_load_example
        RUNTIME DESTINATION bin/examples)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/kernel.hsaco
        DESTINATION share/rocshmem/examples)

