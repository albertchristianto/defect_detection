cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(DefectDetectionCpp)

#boost library
set(Boost_USE_STATIC_LIBS ON)
if(NOT BOOST_DIR)
    set(BOOST_DIR E:/Albert_Christianto/third_party/boost_1_76_0) # either set it here or from the command line  
endif()
if(NOT Boost_INCLUDE_DIR)
    set(Boost_INCLUDE_DIR "${BOOST_DIR}") # either set it here or from the command line  
endif()
if(NOT Boost_LIBRARY_DIR)
    set(Boost_LIBRARY_DIR "${BOOST_DIR}/stage/lib") # either set it here or from the command line  
endif()
find_package(Boost 1.76.0 REQUIRED filesystem)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

#argparse library
set(ARGPARSE_INCLUDE_DIR third_party/argparse/include)
include_directories(${ARGPARSE_INCLUDE_DIR})

#json library
set(JSON_Install OFF CACHE INTERNAL "")
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(NLOHMANN_JSON_DIR third_party/json)
set(NLOHMANN_JSON_INCLUDE_DIR third_party/json/include)
add_subdirectory(${NLOHMANN_JSON_DIR})
include_directories(${NLOHMANN_JSON_INCLUDE_DIR})

#night_fury library
if(NOT night_fury_DIR)
    set(night_fury_DIR E:/Albert_Christianto/Project/night_fury/build/install) # either set it here or from the command line  
endif()
find_package(night_fury 0.0.1 REQUIRED)

# setup CUDA and TensorRT
find_package(CUDA REQUIRED)
# message(STATUS "    libraries: ${CUDA_LIBRARIES}")
# message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
# enable_language(CUDA)  # add this line, then no need to setup cuda path in vs
if(NOT TensorRT_DIR)
    set(TensorRT_DIR E:/Albert_Christianto/third_party/TensorRT-8.5.3.1) # either set it here or from the command line  
endif()
if(NOT TensorRT_INCLUDE_DIRS)
    set(TensorRT_INCLUDE_DIRS "${TensorRT_DIR}/include") # either set it here or from the command line  
endif()
if(NOT TensorRT_LIBRARY_DIRS)
    set(TensorRT_LIBRARY_DIRS "${TensorRT_DIR}/lib") # either set it here or from the command line  
endif()
link_directories(${TensorRT_LIBRARY_DIRS})# Add the TensorRT library directories
set(TensorRT_LIBRARIES "nvinfer" "nvinfer_plugin" "nvonnxparser" "nvparsers")

#setup opencv
if(NOT OpenCV_DIR)
    set(OpenCV_DIR E:/Albert_Christianto/third_party/opencv-python/_skbuild/win-amd64-3.9/cmake-install) # either set it here or from the command line  
endif()
find_package(OpenCV)

add_compile_definitions(DD_BACKEND_EXPORTS)
add_library(DdBackend src/Datum.cpp src/ImgClassifier.cpp src/TrtEngine.cpp)
target_include_directories(DdBackend PUBLIC ${NIGHT_FURY_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(DdBackend PUBLIC nlohmann_json::nlohmann_json Boost::filesystem ${NIGHT_FURY_LIBRARIES} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${OpenCV_LIBS})

add_executable(DefectDetectionCppTest src/test.cpp src/TrtEngine.cpp)
target_include_directories(DefectDetectionCppTest PUBLIC ${NIGHT_FURY_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(DefectDetectionCppTest PUBLIC nlohmann_json::nlohmann_json Boost::filesystem ${NIGHT_FURY_LIBRARIES} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${OpenCV_LIBS})

add_executable(OnnxTrtEngineGenerator src/TrtEngineGenerator.cpp src/TrtEngine.cpp)
target_include_directories(OnnxTrtEngineGenerator PUBLIC ${NIGHT_FURY_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(OnnxTrtEngineGenerator PUBLIC nlohmann_json::nlohmann_json Boost::filesystem ${NIGHT_FURY_LIBRARIES} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${OpenCV_LIBS})