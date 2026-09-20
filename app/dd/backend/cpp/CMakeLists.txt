cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(DefectDetectionCpp)

# === Platform-conditional defaults ===
if(WIN32)
    set(DEFAULT_BOOST_DIR "E:/Albert_Christianto/third_party/boost_1_76_0")
    set(DEFAULT_NIGHT_FURY_DIR "E:/Albert_Christianto/Project/night_fury/build/install")
    set(DEFAULT_TENSORRT_DIR "E:/Albert_Christianto/third_party/TensorRT-8.5.3.1")
    set(DEFAULT_OPENCV_DIR "E:/Albert_Christianto/third_party/opencv-python/_skbuild/win-amd64-3.9/cmake-install")
else()
    # On Linux, rely on system packages or user-provided -D flags
    set(DEFAULT_BOOST_DIR "")
    set(DEFAULT_NIGHT_FURY_DIR "")
    set(DEFAULT_TENSORRT_DIR "")
    set(DEFAULT_OPENCV_DIR "")
endif()

# Boost library
set(Boost_USE_STATIC_LIBS ON)
if(NOT BOOST_DIR AND DEFAULT_BOOST_DIR)
    set(BOOST_DIR ${DEFAULT_BOOST_DIR})
endif()
if(BOOST_DIR)
    if(NOT Boost_INCLUDE_DIR)
        set(Boost_INCLUDE_DIR ${BOOST_DIR})
    endif()
    if(NOT Boost_LIBRARY_DIR)
        set(Boost_LIBRARY_DIR ${BOOST_DIR}/stage/lib)
    endif()
endif()
find_package(Boost 1.76.0 REQUIRED filesystem)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

# Argparse library
set(ARGPARSE_INCLUDE_DIR third_party/argparse/include)
include_directories(${ARGPARSE_INCLUDE_DIR})

# JSON library
set(JSON_Install OFF CACHE INTERNAL "")
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(NLOHMANN_JSON_DIR third_party/json)
set(NLOHMANN_JSON_INCLUDE_DIR third_party/json/include)
add_subdirectory(${NLOHMANN_JSON_DIR})
include_directories(${NLOHMANN_JSON_INCLUDE_DIR})

# night_fury library
if(NOT night_fury_DIR AND DEFAULT_NIGHT_FURY_DIR)
    set(night_fury_DIR ${DEFAULT_NIGHT_FURY_DIR})
endif()
find_package(night_fury 0.0.1 REQUIRED)

# CUDA and TensorRT
find_package(CUDA REQUIRED)
if(NOT TensorRT_DIR AND DEFAULT_TENSORRT_DIR)
    set(TensorRT_DIR ${DEFAULT_TENSORRT_DIR})
endif()
if(TensorRT_DIR)
    if(NOT TensorRT_INCLUDE_DIRS)
        set(TensorRT_INCLUDE_DIRS ${TensorRT_DIR}/include)
    endif()
    if(NOT TensorRT_LIBRARY_DIRS)
        set(TensorRT_LIBRARY_DIRS ${TensorRT_DIR}/lib)
    endif()
    link_directories(${TensorRT_LIBRARY_DIRS})
endif()
set(TensorRT_LIBRARIES "nvinfer" "nvinfer_plugin" "nvonnxparser" "nvparsers")

# OpenCV
if(NOT OpenCV_DIR AND DEFAULT_OPENCV_DIR)
    set(OpenCV_DIR ${DEFAULT_OPENCV_DIR})
endif()
find_package(OpenCV REQUIRED)

# Install dir
set(DD_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(DD_LIBRARIES_DEPENDENCIES nlohmann_json::nlohmann_json Boost::filesystem ${NIGHT_FURY_LIBRARIES} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${OpenCV_LIBS})

add_compile_definitions(DD_BACKEND_EXPORTS)
add_compile_definitions(DD_INFERENCE_EXPORTS)

# On Linux, hide symbols by default so DD_BACKEND_API/DD_INFERENCE_API
# visibility("default") actually controls what's exported
if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
endif()

add_library(DdBackendModule SHARED src/Datum.cpp src/ImgClassifier.cpp src/TrtUtils.cpp)
target_include_directories(DdBackendModule PUBLIC ${NIGHT_FURY_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(DdBackendModule PUBLIC ${DD_LIBRARIES_DEPENDENCIES})

add_library(DdInference SHARED src/ApiImgInfer.cpp src/ResultsSender.cpp src/InferenceManager.cpp src/SystemManager.cpp src/cWrapper.cpp)
target_include_directories(DdInference PUBLIC ${NIGHT_FURY_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(DdInference PUBLIC ${DD_LIBRARIES_DEPENDENCIES} $<TARGET_OBJECTS:DdBackendModule>)

add_executable(ImageClassifierTest src/ImgClassifierTest.cpp)
target_include_directories(ImageClassifierTest PUBLIC ${NIGHT_FURY_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ImageClassifierTest PUBLIC ${DD_LIBRARIES_DEPENDENCIES} $<TARGET_OBJECTS:DdBackendModule>)

add_executable(OnnxTrtEngineGenerator src/TrtEngineGenerator.cpp)
target_include_directories(OnnxTrtEngineGenerator PUBLIC ${NIGHT_FURY_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(OnnxTrtEngineGenerator PUBLIC ${DD_LIBRARIES_DEPENDENCIES} $<TARGET_OBJECTS:DdBackendModule>)

add_executable(DefectDetectionTest src/DdApiTest.cpp)
target_include_directories(DefectDetectionTest PUBLIC ${NIGHT_FURY_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(DefectDetectionTest PUBLIC ${DD_LIBRARIES_DEPENDENCIES} $<TARGET_OBJECTS:DdBackendModule> $<TARGET_OBJECTS:DdInference>)

install(TARGETS DdBackendModule EXPORT DdBackendModule DESTINATION ${DD_INSTALL_DIR})
install(TARGETS DdInference EXPORT DdInference DESTINATION ${DD_INSTALL_DIR})
install(TARGETS ImageClassifierTest EXPORT ImageClassifierTest DESTINATION ${DD_INSTALL_DIR})
install(TARGETS OnnxTrtEngineGenerator EXPORT OnnxTrtEngineGenerator DESTINATION ${DD_INSTALL_DIR})
install(TARGETS DefectDetectionTest EXPORT DefectDetectionTest DESTINATION ${DD_INSTALL_DIR})
install(FILES src/cWrapper.h DESTINATION ${DD_INSTALL_DIR})