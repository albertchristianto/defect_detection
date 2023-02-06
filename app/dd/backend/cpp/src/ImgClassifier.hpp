#ifndef IMAGE_CLASSIFIER_HPP
#define IMAGE_CLASSIFIER_HPP

#include "TrtEngine.hpp"

namespace dd {
    /// This interface class is used to control a inference engine.
    /// The user of this framework must inherit their deep learning engine implementtation from this class.
    template<typename SpTDatum>
    class ImageClassifier: TrtEngine<SpTDatum> {
    public:
        ImageClassifier(std::string path_to_json, int batch_size, int gpu_id);
        ~ImageClassifier();
        bool Init();
        /**
         * Function to do the inference. the user of this framework must implement this functions.
         * @param the_datas batch data which to be processed
         */
        void Forward(std::vector<SpTDatum> &the_datas);
        void WarmUp(int n_times);
        std::string Name();
    private:
        void LoadConfig(const std::string& path);
        //TensorRT engine is loaded using these variable++++
        TrtLogger m_Logger;
        TrtUniquePtr<nvinfer1::ICudaEngine> m_Engine;
        TrtUniquePtr<nvinfer1::IExecutionContext> m_Context;
        TrtUniquePtr<nvinfer1::IRuntime> m_Runtime;
        int m_GpuId;
        int m_BatchSize;
        //++++++++++++++++++++++++++++++++++++++++++++++++++
        //pointer for inference purposes--------------------
        float* m_InputBufferCpu;
        std::vector<void*> m_BuffersGpu;
        cudaStream_t m_CudaStream;
        std::vector<float> m_Means;
        std::vector<float> m_Stds;
        std::string m_WeightsPath;
        std::vector<std::string>  m_ClassesName;
        //--------------------------------------------------
        //variable for pre-processing and post-processing===
        cv::Size m_NetDimension;
        std::vector<unsigned int> m_ArraySizes;
        nvinfer1::Dims m_InputDim;//we expect only one input
        nvinfer1::Dims m_OutputDim;//and one output
        //==================================================
    };
}
#endif