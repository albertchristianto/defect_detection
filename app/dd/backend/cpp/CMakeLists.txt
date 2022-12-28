project(DefectDetectionCpp)
cmake_minimum_required(VERSION 3.0)

#json library
set(JSON_Install OFF CACHE INTERNAL "")
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(third_party/json)

add_executable(DefectDetectionCppTest test.cpp)
target_link_libraries(DefectDetectionCppTest PRIVATE nlohmann_json::nlohmann_json)