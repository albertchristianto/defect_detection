# Defect Detection Inference API
Defect Detection Inference API supports:
1. C / C++ 
2. Python

## Getting Started
Follow these commands to build the library.
```
mkdir build
cd build/
cmake ..
# in Windows
cmake --build . --target install --config Release
# in Linux
make all
make install
```

## How Does It Work?
This backend application is using [night_fury](https://github.com/albertchristianto/night_fury). In order to use the library, there are 4 classes that must inherit classes from the framework. Here is the list of the classes that must be inherit from night_fury library:

1. [*nf::async::BaseDatum*](https://github.com/albertchristianto/night_fury/blob/main/include/nf/async/base_datum.hpp)

    I create a [*dd::Datum*](https://github.com/albertchristianto/defect_detection/blob/main/app/dd/backend/cpp/src/Datum.hpp) which holds all the required variables inside the inference backend system. [*nf::async::BaseDatum*](https://github.com/albertchristianto/night_fury/blob/main/include/nf/async/base_datum.hpp) contains important variables for the [night_fury](https://github.com/albertchristianto/night_fury) threading system.

2. [*nf::I_InferEngine*](https://github.com/albertchristianto/night_fury/blob/main/include/nf/inference_core/i_InferEngine.hpp)

    In order to use the [*nf::async::WInference*](https://github.com/albertchristianto/night_fury/blob/main/include/nf/async/worker/wInference.hpp), I must wrap [image classification function](https://github.com/albertchristianto/defect_detection/blob/main/app/dd/backend/cpp/src/ImgClassifier.hpp) into a class which inherit [*nf::I_InferEngine*](https://github.com/albertchristianto/night_fury/blob/main/include/nf/inference_core/i_InferEngine.hpp). 

3. [*nf::ImgInfer*](https://github.com/albertchristianto/night_fury/blob/main/include/nf/inference_core/imgInfer.hpp)

    I must implement [*dd::ImgInfer::GetResults*](https://github.com/albertchristianto/defect_detection/blob/ed4c404820de8de67ff6ea45c254c30de52ccb79/app/dd/backend/cpp/src/ApiImgInfer.cpp#L10), [*dd::ImgInfer::Produce*](https://github.com/albertchristianto/defect_detection/blob/ed4c404820de8de67ff6ea45c254c30de52ccb79/app/dd/backend/cpp/src/ApiImgInfer.cpp#L30), and [*dd::ImgInfer::CheckImage*](https://github.com/albertchristianto/defect_detection/blob/ed4c404820de8de67ff6ea45c254c30de52ccb79/app/dd/backend/cpp/src/ApiImgInfer.cpp#L25) by inheriting [*nf::ImgInfer*](https://github.com/albertchristianto/night_fury/blob/main/include/nf/inference_core/imgInfer.hpp). [*nf::ImgInfer*](https://github.com/albertchristianto/night_fury/blob/main/include/nf/inference_core/imgInfer.hpp) will handle the data conversion from a [*dd::C_Image*](https://github.com/albertchristianto/defect_detection/blob/ed4c404820de8de67ff6ea45c254c30de52ccb79/app/dd/backend/cpp/src/cWrapper.h#L21) to a [*dd::Datum*](https://github.com/albertchristianto/defect_detection/blob/main/app/dd/backend/cpp/src/Datum.hpp).

4. [*nf::async::WDataSender*](https://github.com/albertchristianto/night_fury/blob/main/include/nf/async/worker/wDataSender.hpp)

    To send the data from the user of the inference backend system, I implement [*dd::ResultsSender::CollectResults*](https://github.com/albertchristianto/defect_detection/blob/ed4c404820de8de67ff6ea45c254c30de52ccb79/app/dd/backend/cpp/src/ResultsSender.cpp#L10) which will convert a [*dd::Datum*](https://github.com/albertchristianto/defect_detection/blob/main/app/dd/backend/cpp/src/Datum.hpp) to a [*dd::C_Results*](https://github.com/albertchristianto/defect_detection/blob/ed4c404820de8de67ff6ea45c254c30de52ccb79/app/dd/backend/cpp/src/cWrapper.h#L31). 

[*dd::InferenceManager*](https://github.com/albertchristianto/defect_detection/blob/ed4c404820de8de67ff6ea45c254c30de52ccb79/app/dd/backend/cpp/src/InferenceManager.hpp#L16) handles thread for the inference worker to provide image classification inference service. By passing a [*dd::C_Image*](https://github.com/albertchristianto/defect_detection/blob/ed4c404820de8de67ff6ea45c254c30de52ccb79/app/dd/backend/cpp/src/cWrapper.h#L21), the inference thread, which is spawn by [*dd::InferenceManager*](https://github.com/albertchristianto/defect_detection/blob/ed4c404820de8de67ff6ea45c254c30de52ccb79/app/dd/backend/cpp/src/InferenceManager.hpp#L16), automatically produces the inference results.
[*dd::SystemManager*](https://github.com/albertchristianto/defect_detection/blob/ed4c404820de8de67ff6ea45c254c30de52ccb79/app/dd/backend/cpp/src/SystemManager.hpp#L9) handles all the necessary modules, such as [inference manager](https://github.com/albertchristianto/defect_detection/blob/ed4c404820de8de67ff6ea45c254c30de52ccb79/app/dd/backend/cpp/src/InferenceManager.hpp#L16), [inference API modules](https://github.com/albertchristianto/defect_detection/blob/ed4c404820de8de67ff6ea45c254c30de52ccb79/app/dd/backend/cpp/src/ApiImgInfer.hpp#L30), [image throughput counter](https://github.com/albertchristianto/defect_detection/blob/ed4c404820de8de67ff6ea45c254c30de52ccb79/app/dd/backend/cpp/src/SystemManager.hpp#L17), and other future features for the system manager. 

[cWrapper.h](https://github.com/albertchristianto/defect_detection/blob/main/app/dd/backend/cpp/src/cWrapper.h) is the C-API which the API for using Defect Detection Inference Backend System. We can also directly load the built dll using dll loader from other languages, such as [python-ctypes](https://github.com/albertchristianto/defect_detection/blob/main/app/dd/backend/cpp/DdInference.py), C#-interop, and many more.

## Contact Me
If you have any questions or suggestions, contact me at albertchristianto1994@gmail.com or create an issue in this repository.