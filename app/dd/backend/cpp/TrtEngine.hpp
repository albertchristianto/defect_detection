#include <memory>
#include <NvInfer.h>
#include <boost/filesystem.hpp>
#include <nf/inference_core/i_InferEngine.hpp>
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

    /// This interface class is used to control a inference engine.
    /// The user of this framework must inherit their deep learning engine implementtation from this class.
    template<typename SpTDatum>
    class TrtEngine: nf::I_InferEngine<SpTDatum> {
    public:
        /**
         * Class destructor function
         */
        virtual ~ImageClassifier() {}
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
        std::string ParseOnnxModel(std::string module_name, std::string onnx_weight_path, int batch_size, bool force_create) {
            if (!boost::filesystem::exists(boost::filesystem::path(onnx_weight_path)))
                throw std::runtime_error(std::string(module_name + ": Could not find the onnx weight path!!"));//throw an error

            TrtLogger mTrtLogger;
            TRTUniquePtr<nvinfer1::ICudaEngine> mEngine{ nullptr };

            TRTUniquePtr<nvinfer1::IBuilder> builder{ nvinfer1::createInferBuilder(mTrtLogger) };
            //get the tensorRT path
            boost::filesystem::path weight_path(onnx_weight_path);
            std::string the_tmp_name = aio::StringSplit(onnx_weight_path, '/').back();
            std::string weight_name = aio::StringSplit(the_tmp_name, '.')[0];
            std::string temp_path;

            if (!boost::filesystem::exists(boost::filesystem::path("./cache")))
                boost::filesystem::create_directory(boost::filesystem::path("./cache"));

            if (builder->platformHasFastFp16())
                temp_path = "cache/" + weight_name + "_batch" + std::to_string(batch_size) + "_fp16.engine";
            else
                temp_path = "cache/" + weight_name + "_batch" + std::to_string(batch_size) + "_fp32.engine";

            AIO_LOGGER_TRACE("{0}: Checking File Path {1}", module_name, temp_path);
            if (boost::filesystem::exists(boost::filesystem::path(temp_path)) && !force_create) {
                //Logging::Info("the weight path: " + mWeight);
                return temp_path;
            }
            AIO_LOGGER_INFO("{0}: Building tensorRT engine!", module_name);

            const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
            TRTUniquePtr<nvinfer1::INetworkDefinition> network{ builder->createNetworkV2(explicitBatch) };
            TRTUniquePtr<nvonnxparser::IParser> parser{ nvonnxparser::createParser(*network, mTrtLogger) };
            TRTUniquePtr<nvinfer1::IBuilderConfig> config{ builder->createBuilderConfig() };
            // parse ONNX
            if (!parser->parseFromFile(onnx_weight_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR)))
                throw std::runtime_error(std::string(module_name + ": Could not parse the Onnx model"));

            // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
            config->setMaxWorkspaceSize(1ULL << 30);
            //if (builder->platformHasFastInt8())
            //{
            //	AIO_LOGGER_INFO("{0}: Using INT8", module_name);
            //	config->setFlag(nvinfer1::BuilderFlag::kINT8);
            //}
            //else 
            if (builder->platformHasFastFp16())
            {// use FP16 mode if possible
                AIO_LOGGER_INFO("{0}: Using FP16", module_name);
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
            }
            else
                AIO_LOGGER_INFO("{0}: Using FP32", module_name);

            // we have only one image in batch
            //Logging::Info("the max batch: " + std::to_string(mBatchSize));
            //AIO_LOGGER_INFO("Maximum Batch Size: {0}", batch_size);
            builder->setMaxBatchSize(batch_size);

            auto input = network->getInput(0);
            auto input_shape = input->getDimensions();

            //std::cout << input_shape.d[0] << " " << input_shape.d[1] << " " << input_shape.d[2] << " " << input_shape.d[3] << " " << std::endl;

            //input_shape.d[0] = 1;
            input_shape.d[0] = batch_size;

            nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_shape);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_shape);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_shape);

            config->addOptimizationProfile(profile);
            profile = nullptr;

            // generate TensorRT engine optimized for the target platform
            mEngine.reset(builder->buildEngineWithConfig(*network, *config));

            TRTUniquePtr<nvinfer1::IHostMemory> modelStream{ nullptr };
            modelStream.reset(mEngine->serialize());

            AIO_LOGGER_TRACE("{0}: Saving to {1}", module_name, temp_path);
            std::ofstream p(temp_path, std::ios::binary);
            if (!p)
                throw std::runtime_error(std::string(module_name + ": Failed to save the tensorRT engine!!"));//throw an error

            p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
            mEngine.reset();
            //AIO_LOGGER_INFO("done building");
            return temp_path;
        }
    };
}