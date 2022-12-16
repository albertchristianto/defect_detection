
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) override {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "n";
        }
    }
} gLogger;

// destroy TensorRT objects if something goes wrong
struct TRTDestroy
{
    template< class T >
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};
 
template< class T >
using TRTUniquePtr = std::unique_ptr< T, TRTDestroy >;

void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                    TRTUniquePtr< nvinfer1::IExecutionContext >& context)
{
    TRTUniquePtr< nvinfer1::IBuilder > builder{nvinfer1::createInferBuilder(gLogger)};
    TRTUniquePtr< nvinfer1::INetworkDefinition > network{builder->createNetwork()};
    TRTUniquePtr< nvonnxparser::IParser > parser{nvonnxparser::createParser(*network, gLogger)};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast< int >(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }
    TRTUniquePtr< nvinfer1::IBuilderConfig > config{builder->createBuilderConfig()};
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // we have only one image in batch
    builder->setMaxBatchSize(1);
    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
}