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
    set(Boost_INCLUDE_DIR ${BOOST_DIR}) # either set it here or from the command line  
endif()
if(NOT Boost_LIBRARY_DIR)
    set(Boost_LIBRARY_DIR ${BOOST_DIR}/stage/lib) # either set it here or from the command line  
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
    set(TensorRT_INCLUDE_DIRS ${TensorRT_DIR}/include) # either set it here or from the command line  
endif()
if(NOT TensorRT_LIBRARY_DIRS)
    set(TensorRT_LIBRARY_DIRS ${TensorRT_DIR}/lib) # either set it here or from the command line  
endif()
link_directories(${TensorRT_LIBRARY_DIRS})# Add the TensorRT library directories
set(TensorRT_LIBRARIES "nvinfer" "nvinfer_plugin" "nvonnxparser" "nvparsers")

#setup opencv
if(NOT OpenCV_DIR)
    set(OpenCV_DIR E:/Albert_Christianto/third_party/opencv-python/_skbuild/win-amd64-3.9/cmake-install) # either set it here or from the command line  
endif()
find_package(OpenCV)

set(DD_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(DD_LIBRARIES_DEPENDENCIES nlohmann_json::nlohmann_json Boost::filesystem ${NIGHT_FURY_LIBRARIES} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${OpenCV_LIBS})

add_compile_definitions(DD_BACKEND_EXPORTS)
add_compile_definitions(DD_INFERENCE_EXPORTS)

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