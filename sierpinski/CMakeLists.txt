project(sierpinski)

cmake_minimum_required(VERSION 2.8)

enable_testing()

set(bitness "x86")
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(bitness "x64")
endif()

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
include_directories($ENV{INTELOCLSDKROOT}/include ${PROJECT_SOURCE_DIR}/clu)
link_directories($ENV{INTELOCLSDKROOT}/lib/${bitness})

add_library(clu clu/clu.h clu/clu_runtime.cpp)
add_executable(sierpinski sierpinski.cpp ) 
target_link_libraries(sierpinski clu OpenCL)

add_test(NAME sierpinski_smoke
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
         COMMAND sierpinski)

