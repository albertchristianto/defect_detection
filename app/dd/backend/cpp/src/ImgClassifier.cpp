#include "ImgClassifier.hpp"
#include <string>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>
#include <nf/utilities/logger.hpp>
#include <cuda_runtime_api.h>

namespace dd {
    /// This interface class is used to control a inference engine.
    /// The user of this framework must inherit their deep learning engine implementtation from this class.
    template<typename SpTDatum>
    ImageClassifier<SpTDatum>::ImageClassifier(std::string path_to_json, int batch_size, int gpu_id):
        m_Engine{ nullptr }, m_InputBufferCpu{ nullptr }, m_Context{ nullptr }
    {
        LoadConfig(path_to_json);
        m_BatchSize = batch_size;
        m_GpuId = gpu_id;
        if (m_GpuId < 0)
            throw std::runtime_error(std::string(this->Name() + ": This system is using TensorRT backend!! Must use GPU to start the inference engine!!"));//throw an error
        cudaSetDevice(m_GpuId);
        int ret = cudaStreamCreate(&m_CudaStream);
        if (ret != 0) {
            std::string log_str = this->Name() + ": Failed to create CUDA memory stream (Direct Memory Access). Error code: " + std::to_string(ret);
            throw std::runtime_error(log_str);//throw an error
        }
        m_WeightsPath = TrtParseOnnxModel(this->Name(), m_WeightsPath, m_BatchSize, false);
        if (!boost::filesystem::exists(boost::filesystem::path(m_WeightsPath)))
            throw std::runtime_error(std::string(this->Name() + ":  Couldn't find the YOLO TensorRT engine file!!"));//throw an error
    }
    template<typename SpTDatum>
    ImageClassifier<SpTDatum>::~ImageClassifier() {
        for (void* buf : m_BuffersGpu)
            cudaFree(buf);
        cudaStreamDestroy(m_CudaStream);
        if (m_InputBufferCpu != nullptr) {
            delete[] m_InputBufferCpu;
            m_InputBufferCpu = nullptr;
        }
    }
    template<typename SpTDatum>
    bool ImageClassifier<SpTDatum>::IsReady() {
        try {
            this->WarmUp(1);
        }
        catch (std::exception& e) {
            NF_LOGGER_ERROR("{0}: {1}", this->Name(), e.what());
            return false;
        }
        return true;
    }
    template<typename SpTDatum>
    bool ImageClassifier<SpTDatum>::Init() {// initialize TensorRT engine and parse ONNX model --------------------------------------------------------------------
        try {
            // NF_LOGGER_TRACE("{0}: Initialize", this->Name());
            cudaSetDevice(m_GpuId);
            std::vector<unsigned char> weight_buffer = TrtReadEngineFile(m_WeightsPath);

            m_Runtime.reset(nvinfer1::createInferRuntime(m_Logger));
            m_Engine.reset(m_Runtime->deserializeCudaEngine(weight_buffer.data(), weight_buffer.size()));
            m_Context.reset(m_Engine->createExecutionContext());

            m_BuffersGpu.resize(m_Engine->getNbBindings()); // buffers for input and output data
            m_ArraySizes.resize(m_Engine->getNbBindings());
            for (size_t i = 0; i < m_Engine->getNbBindings(); ++i) {
                m_ArraySizes[i] = TrtGetSize(m_Engine->getBindingDimensions(i));
                // m_ArraySizes[i] = m_ArraySizes[i] * m_BatchSize;
                cudaMalloc(&m_BuffersGpu[i], m_ArraySizes[i] * sizeof(float));
                if (m_Engine->bindingIsInput(i)) {
                    m_InputDim = m_Engine->getBindingDimensions(i);
                    //NF_LOGGER_TRACE("{0}: {1}x{2}x{3}x{4}", this->Name(), m_InputDim.d[0], m_InputDim.d[1], m_InputDim.d[2], m_InputDim.d[3]);
                    continue;
                }
                m_OutputDim = m_Engine->getBindingDimensions(i);
                //NF_LOGGER_TRACE("{0}: {1}x{2}x{3}x{4}", this->Name(), m_OutputDim.d[0], m_OutputDim.d[1], m_OutputDim.d[2], m_OutputDim.d[3]);
            }
            m_InputBufferCpu = new float[m_ArraySizes[0]];
        }
        catch (std::exception& e) {
            NF_LOGGER_ERROR("{0}: {1}", this->Name(), e.what());
            return false;
        }
        // NF_LOGGER_TRACE("{0}: Initialize success!", this->Name());
        return true;
    }
    template<typename SpTDatum>
    void ImageClassifier<SpTDatum>::WarmUp(int n_times) {
        cudaSetDevice(m_GpuId);//set the active GPU device
        cv::Mat pr_img(m_NetDimension, CV_32FC3, cv::Scalar(0, 0, 0));
        // for (int i = 0; i < m_NetDimension.height * m_NetDimension.width; i++) {
        //     m_InputBufferCpu[i] = float(pr_img.at<cv::Vec3b>(i)[2]) / 255.0;
        //     m_InputBufferCpu[i + m_NetDimension.height * m_NetDimension.width] = float(pr_img.at<cv::Vec3b>(i)[1]) / 255.0;
        //     m_InputBufferCpu[i + 2 * m_NetDimension.height * m_NetDimension.width] = float(pr_img.at<cv::Vec3b>(i)[0]) / 255.0;
        // }
        cudaMemcpyAsync(m_BuffersGpu[0], m_InputBufferCpu, m_ArraySizes[0] * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream);
        for (size_t i = 0; i < n_times; ++i)// inference
            m_Context->enqueue(m_BatchSize, m_BuffersGpu.data(), m_CudaStream, nullptr);
    }
    /**
     * Function to do the inference. the user of this framework must implement this functions.
     * @param the_datas batch data which to be processed
     */
    template<typename SpTDatum>
    void ImageClassifier<SpTDatum>::Forward(std::vector<SpTDatum> &the_datas) {
        //NF_LOGGER_TRACE("{0}: {1}x{2}x{3}x{4} {5}", this->Name(), m_OutputDim.d[0], m_OutputDim.d[1], m_OutputDim.d[2], m_OutputDim.d[3], m_ArraySizes[1]);
        for (int b = 0; b < the_datas.size(); b++) {
            cv::Mat pr_img;
            cv::resize(the_datas[b]->cvInputData, pr_img, m_NetDimension);
            for (int j = 0; j < m_NetDimension.height * m_NetDimension.width; j++) {
                m_InputBufferCpu[j] = ((pr_img.at<cv::Vec3b>(j)[2] * 0.00392156f) - m_Means[0]) / m_Stds[0];
                m_InputBufferCpu[j + m_NetDimension.height * m_NetDimension.width] = ((pr_img.at<cv::Vec3b>(j)[1] * 0.00392156f) - m_Means[1]) / m_Stds[1];
                m_InputBufferCpu[j + 2 * m_NetDimension.height * m_NetDimension.width] = ((pr_img.at<cv::Vec3b>(j)[0] * 0.00392156f) - m_Means[2]) / m_Stds[2];
            }
            cudaMemcpyAsync(m_BuffersGpu[0], m_InputBufferCpu, m_ArraySizes[0] * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream);
            m_Context->enqueue(m_BatchSize, m_BuffersGpu.data(), m_CudaStream, nullptr);

            std::vector<float> cpu_output(m_ArraySizes[1]);
            cudaMemcpyAsync(cpu_output.data(), m_BuffersGpu[1], m_ArraySizes[1] * sizeof(float), cudaMemcpyDeviceToHost, m_CudaStream);
            cudaStreamSynchronize(m_CudaStream);
            int max_idx = 0;
            for (int c = 1; c < m_OutputDim.d[1]; c++) {
                //int idx_now = b * m_BatchSize;
                //NF_LOGGER_TRACE("{0}: {1}, {2}", this->Name(), cpu_output[max_idx], cpu_output[c]);
                if (cpu_output[max_idx] < cpu_output[c])
                    max_idx = c;
            }
            for (int c = 0; c < m_OutputDim.d[1]; c++) {
                //int idx_now = b * m_BatchSize;
                NF_LOGGER_TRACE("{0}: {1}", this->Name(), cpu_output[c]);
            }
            the_datas[b]->className = m_ClassesName[max_idx];
            // NF_LOGGER_TRACE("{0}: {1}x{2}x{3}x{4}", this->Name(), m_OutputDim.d[0], m_OutputDim.d[1], m_OutputDim.d[2], m_OutputDim.d[3]);
        }
    }
    template<typename SpTDatum>
    std::string ImageClassifier<SpTDatum>::Name() {
        return "TRT_Image_Classifier";
    }
    template<typename SpTDatum>
    void ImageClassifier<SpTDatum>::LoadConfig(const std::string& path) {
        if (!boost::filesystem::exists(boost::filesystem::path(path)))
            throw std::runtime_error(std::string(this->Name() + ": Could not find the json config file!!!"));//throw an error
        std::ifstream file(path);
        nlohmann::json data = nlohmann::json::parse(file);
        m_WeightsPath = data["weights_path"].get<std::string>();
        m_NetDimension.width = data["input_size"].get<int>();
        m_NetDimension.height = data["input_size"].get<int>();
        m_Means = data["means"].get<std::vector<float>>();
        //NF_LOGGER_TRACE("{0}: {1}x{2}x{3}", this->Name(), m_Means[0], m_Means[1], m_Means[2]);
        m_Stds = data["stds"].get<std::vector<float>>();
        //NF_LOGGER_TRACE("{0}: {1}x{2}x{3}", this->Name(), m_Stds[0], m_Stds[1], m_Stds[2]);
        m_ClassesName = data["class_name"].get<std::vector<std::string>>();
    }
    COMPILE_TEMPLATE_DATUM(ImageClassifier);
    DEFINE_TEMPLATE_DATUM(ImageClassifier);
}