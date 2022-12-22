#pragma once
#include "macros.hpp"

#include <opencv2/opencv.hpp>
#include <aio/async/engine.hpp>

#include <FDASFR_aio_utils/modelParam.hpp>
#include <FDASFR_aio_utils/trtUtils.hpp>
#include <FDASFR_aio_utils/datum.hpp>
#include <matcher_aio/matcher.hpp>

namespace FDASFR_AIO
{
	template<typename SpTDatum>
	class FRtrtNet : public aio::async::IEngine<SpTDatum>
	{
	public:
		FRtrtNet(const ModelParam& param);

		~FRtrtNet();

		bool Init();//start function

		void Forward(std::vector<SpTDatum>& the_datas);//net forward function

		bool IsReady();//checker function

		float GetScoreThreshold();

		void SetScoreThreshold(float the_thresh);

		std::string Name();

		std::string Version();//version checker function

		unsigned char CheckType();

	private:
		//cv::dnn::Net model;
		/*std::vector<int64_t> inputDims;
		std::vector<int64_t> outputDims;*/

		std::vector<unsigned char> mAesParam;
		std::string mModelVersion;
		std::string mFRPath;
		int mGpuId;
		bool enable_fp16;
		/*bool enable_trt_cache;
		std::string trt_cache_path;*/

		//std::vector<cv::Mat> input_data;
		int mBatchSize;
		TrtLogger mTrtLogger;
		TRTUniquePtr<nvinfer1::ICudaEngine> mEngine;
		TRTUniquePtr<nvinfer1::IExecutionContext> mContext;
		TRTUniquePtr<nvinfer1::IRuntime> mRuntime;
		std::vector<nvinfer1::Dims> mInputDims; // we expect only one input
		std::vector<nvinfer1::Dims> mOutputDims; // and one output
		std::vector<unsigned int> mArraySizes;
		cudaStream_t mCudaStream;
		float* mInputBufferCpu;
		std::vector<void*> mBuffers; // buffers for input and output data
		cv::Size mNetDimension;

		cv::Mat alignFace(cv::Mat img, std::vector<float> landmark);
		//std::vector<float> prepareImage(std::vector<cv::Mat>& image, std::vector<std::vector<float>>& vec_lmk);
		//void ReshapeandNormalize(float* out, cv::Mat& feature, const int& MAT_SIZE, const int& outSize);
		void LoadConfig(const std::string& path);
		void WarmUp(int n_times);

		DELETE_COPY(FRtrtNet);
	};
}