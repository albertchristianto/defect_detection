cmake_minimum_required(VERSION 3.0)
project(DefectDetectionCpp)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#boost library
set(Boost_USE_STATIC_LIBS ON)
if(NOT BOOST_DIR)
    set(BOOST_DIR E:/Albert_Christianto/third_party/boost_1_76_0) # either set it here or from the command line  
endif()
if(NOT Boost_INCLUDE_DIR)
    set(Boost_INCLUDE_DIR E:/Albert_Christianto/third_party/boost_1_76_0) # either set it here or from the command line  
endif()
if(NOT Boost_LIBRARY_DIR)
    set(Boost_LIBRARY_DIR E:/Albert_Christianto/third_party/boost_1_76_0/stage/lib) # either set it here or from the command line  
endif()
find_package(Boost 1.76.0 REQUIRED filesystem)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

#json library
set(JSON_Install OFF CACHE INTERNAL "")
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(NLOHMANN_JSON_DIR third_party/json)
set(NLOHMANN_JSON_INCLUDE_DIR third_party/json/include)
add_subdirectory(${NLOHMANN_JSON_DIR})
include_directories(${NLOHMANN_JSON_INCLUDE_DIR})

add_executable(DefectDetectionCppTest src/test.cpp)
target_link_libraries(DefectDetectionCppTest PRIVATE nlohmann_json::nlohmann_json Boost::filesystem)