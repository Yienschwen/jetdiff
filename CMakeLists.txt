cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} 
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

set(PYBIND11_NEWPYTHON ON)
find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ceres REQUIRED)

pybind11_add_module(cjet ${SKBUILD_PROJECT_NAME}/jet/cjet.cc)
target_include_directories(cjet PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(cjet PRIVATE ceres)

install(TARGETS cjet LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}/jet)