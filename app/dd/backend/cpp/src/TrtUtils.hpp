#ifndef TRT_ENGINE_HPP
#define TRT_ENGINE_HPP
#include <string>
#include <memory>
#include <NvInfer.h>
#include <boost/filesystem.hpp>
#include <nf/utilities/logger.hpp>

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
    std::string TrtParseOnnxModel(std::string module_name, std::string onnx_weight_path, int batch_size, bool force_create);
    size_t TrtGetSize(const nvinfer1::Dims& dims);
    std::vector<unsigned char> TrtReadEngineFile(std::string& file_path);
}
#endif