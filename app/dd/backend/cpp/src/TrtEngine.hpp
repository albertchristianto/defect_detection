#pragma once
#include <memory>
#include <NvInfer.h>
#include <boost/filesystem.hpp>
#include <nf/inference_core/i_InferEngine.hpp>
#include <nf/utilities/logger.hpp>
#include <nf/utilities/stringManipulation.hpp>

namespace dd {
    class TrtLogger : public nvinfer1::ILogger {
    public:
        void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
            // log errors, warnings, and other information during the build and inference phases
            // remove this 'if' if you need more logged info
            if ((severity == nvinfer1::ILogger::Severity::kERROR) || (severity == nvinfer1::ILogger::Severity::kINTERNAL_ERROR))
                NF_LOGGER_ERROR(msg);
        }
    };

    struct TrtDestroy {// destroy TensorRT objects if something goes wrong
        template <class T>
        void operator()(T* obj) const {
            if (obj)
                obj->destroy();
        }
    };

    template <class T>
    using TrtUniquePtr = std::unique_ptr<T, TrtDestroy>;

    std::string TrtParseOnnxModel(std::string& module_name, std::string onnx_weight_path, int batch_size, bool force_create);

    /// This interface class is used to control a inference engine.
    /// The user of this framework must inherit their deep learning engine implementtation from this class.
    template<typename SpTDatum>
    class TrtEngine: nf::I_InferEngine<SpTDatum> {
    public:
        /**
         * Class destructor function
         */
        virtual ~TrtEngine() {}
        /**
         * Function to initialize the inference engine. the user of this framework must implement this functions.
         * @return true if the initialization process is successful
         * @return false if the initialization process is failed
         */
        virtual bool Init() = 0;
        /**
         * Function to do the inference. the user of this framework must implement this functions.
         * @param the_datas batch data which to be processed
         */
        virtual void Forward(std::vector<SpTDatum> &the_datas) = 0;
        /**
         * Function to warm up of the engine. the user of this framework must implement this functions.
         * @param n_times the number of how many iterations to try the engine
         * @return the name of the engine
         */
        virtual void WarmUp(int n_times) = 0;
        /**
         * Function to the name of the engine. the user of this framework must implement this functions.
         * @return the name of the engine
         */
        virtual std::string Name() = 0;
        /**
         * Function to check the readiness of the inference engine. the user of this framework must implement this functions.
         * @return true if the inference engine is ready
         * @return false if the inference engine is not ready
         */
        bool IsReady() {
            try
                WarmUp(1);
            catch (std::exception& e) {
                NF_LOGGER_ERROR("{0}: {1}", this->Name(), e.what());
                return false;
            }
            return true;
        }
        size_t GetSize(const nvinfer1::Dims& dims) {// calculate size of tensor from tensorRT
            size_t size = 1;
            for (size_t i = 0; i < dims.nbDims; ++i)
                size *= dims.d[i];
            return size;
        }
        std::vector<unsigned char> ReadFile(std::string& file_path) {
            std::ifstream instream(file_path, std::ios::in | std::ios::binary);
            std::vector<unsigned char> data((std::istreambuf_iterator<unsigned char>(instream)), std::istreambuf_iterator<unsigned char>());
            return data;
        }
    };
}