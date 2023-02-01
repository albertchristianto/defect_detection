#include "TrtEngine.hpp"
#include <NvOnnxParser.h>

namespace dd {
    std::string TrtParseOnnxModel(std::string& module_name, std::string onnx_weight_path, int batch_size, bool force_create) {
        if (!boost::filesystem::exists(boost::filesystem::path(onnx_weight_path)))
            throw std::runtime_error(std::string(module_name + ": Could not find the onnx weight path!!"));//throw an error
        TrtLogger the_logger;
        TrtUniquePtr<nvinfer1::IBuilder> builder{ nvinfer1::createInferBuilder(the_logger) };
        std::string ret_val;
        {//define the TensorRT engine path
            if (!boost::filesystem::exists(boost::filesystem::path("./cache")))
                boost::filesystem::create_directory(boost::filesystem::path("./cache"));
            boost::filesystem::path weight_path(onnx_weight_path);
            std::string the_tmp_name = nf::String::Split(onnx_weight_path, '/').back();
            std::string weight_name = nf::String::Split(the_tmp_name, '.')[0];
            if (builder->platformHasFastFp16())
                ret_val = "cache/" + weight_name + "_batch" + std::to_string(batch_size) + "_fp16.engine";
            else
                ret_val = "cache/" + weight_name + "_batch" + std::to_string(batch_size) + "_fp32.engine";
        }
        if (boost::filesystem::exists(boost::filesystem::path(ret_val)) && !force_create) {
            NF_LOGGER_TRACE("{0}: TensorRT engine path {1}", module_name, ret_val);
            return ret_val;
        }
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        TrtUniquePtr<nvinfer1::INetworkDefinition> network{ builder->createNetworkV2(explicitBatch) };
        TrtUniquePtr<nvonnxparser::IParser> parser{ nvonnxparser::createParser(*network, the_logger) };
        TrtUniquePtr<nvinfer1::IBuilderConfig> config{ builder->createBuilderConfig() };
        if (!parser->parseFromFile(onnx_weight_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR)))//parse ONNX
            throw std::runtime_error(std::string(module_name + ": Could not parse the Onnx model"));
        config->setMaxWorkspaceSize(1ULL << 30);//allow TensorRT to use up to 1GB of GPU memory for tactic selection.
        if (builder->platformHasFastFp16()) {
            NF_LOGGER_TRACE("{0}: Using FP16", module_name);
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        else
            NF_LOGGER_TRACE("{0}: Using FP32", module_name);
        builder->setMaxBatchSize(batch_size);
        auto input = network->getInput(0);
        auto input_shape = input->getDimensions();
        input_shape.d[0] = batch_size;
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();//check NVidia TensorRT documentation, could be the source of memory leak
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_shape);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_shape);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_shape);
        config->addOptimizationProfile(profile);
        TrtUniquePtr<nvinfer1::ICudaEngine> the_engine{ nullptr };
        the_engine.reset(builder->buildEngineWithConfig(*network, *config));
        TrtUniquePtr<nvinfer1::IHostMemory> modelStream{ nullptr };
        modelStream.reset(the_engine->serialize());
        NF_LOGGER_TRACE("{0}: Saving to {1}", module_name, ret_val);
        std::ofstream p(ret_val, std::ios::binary);
        if (!p)
            throw std::runtime_error(std::string(module_name + ": Failed to save the tensorRT engine!!"));
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        the_engine.reset();
        return ret_val;
    }
}