#include "arcface_trt.hpp"
#include "utils.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

#include <boost/filesystem.hpp>

#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <aio/utils.hpp>
#include <aio/logger.hpp>
#include <encryption/encryption.hpp>

#include <FDASFR_aio_utils/datum.hpp>

namespace FDASFR_AIO {
    template<typename SpTDatum>
    FRtrtNet<SpTDatum>::FRtrtNet(const ModelParam& param) :
        mEngine{ nullptr },
        mInputBufferCpu{nullptr},
        mContext{ nullptr }
    {
        mBatchSize = param.batch_size;

        mGpuId = param.gpu_id;
        mAesParam = param.aes_param;
        LoadConfig(param.path_to_config);
        mNetDimension = { 112, 112 };
        std::string tmp;
        if (mGpuId < 0)
            throw std::runtime_error(std::string(this->Name() + ": This system is using TensorRT backend!! Must use GPU to start the inference engine!!"));//throw an error
        cudaSetDevice(mGpuId);
        int ret = cudaStreamCreate(&mCudaStream);
        if (ret != 0) {
            std::string log_str = this->Name() + ": Failed to create CUDA memory stream (Direct Memory Access). Error code: " + std::to_string(ret);
            throw std::runtime_error(log_str);//throw an error
        }

        if (mAesParam.size() > 0)
        {
            try {
                tmp = CheckCreateTensorRtEngine(this->Name(), mFRPath, mBatchSize, false, mAesParam);
            }
            catch (std::exception& e) {
                std::string log_str = this->Name() + ": " + std::string(e.what());
                throw std::runtime_error(log_str);//throw an error
            }
            catch (...) {
                std::string log_str = this->Name() + ": Unexpected error while loading encrypted model";
                throw std::runtime_error(log_str);//throw an error
            }
        }
        else
            tmp = CheckCreateTensorRtEngine(this->Name(), mFRPath, mBatchSize, false);
        mFRPath = tmp;
        if (!boost::filesystem::exists(boost::filesystem::path(mFRPath)))
            throw std::runtime_error(std::string(this->Name() + ":  Couldn't find the YOLO TensorRT engine file!!"));//throw an error

    }

    template<typename SpTDatum>
    FRtrtNet<SpTDatum>::~FRtrtNet() {
        for (void* buf : mBuffers)
            cudaFree(buf);
        cudaStreamDestroy(mCudaStream);
        if (mInputBufferCpu != nullptr) {
            delete[] mInputBufferCpu;
            mInputBufferCpu = nullptr;
        }
    }

    template<typename SpTDatum>
    bool FRtrtNet<SpTDatum>::Init()
    {// initialize TensorRT engine and parse ONNX model --------------------------------------------------------------------
        try {
            cudaSetDevice(mGpuId);
            std::vector<unsigned char> weight_buffer;
            if (mAesParam.size() > 0)
            {
                try {
                    weight_buffer = Encryption::AES::Decrypt(mFRPath, &mAesParam[0], 16, &mAesParam[16]);
                }
                catch (std::exception& e) {
                    AIO_LOGGER_ERROR("{0}: {1}", this->Name(), e.what());
                    return false;
                }
                catch (...) {
                    AIO_LOGGER_ERROR("{0}: Unexpected error while loading encrypted model", this->Name());
                    return false;
                }
            }
            else
                weight_buffer = Encryption::ReadFile(mFRPath);

            mRuntime.reset(nvinfer1::createInferRuntime(mTrtLogger));
            mEngine.reset(mRuntime->deserializeCudaEngine(weight_buffer.data(), weight_buffer.size()));
            mContext.reset(mEngine->createExecutionContext());

            mBuffers.resize(mEngine->getNbBindings()); // buffers for input and output data
            mArraySizes.resize(mEngine->getNbBindings());
            for (size_t i = 0; i < mEngine->getNbBindings(); ++i)
            {
                mArraySizes[i] = TrtGetSize(mEngine->getBindingDimensions(i));
                auto binding_size = mArraySizes[i] * mBatchSize * sizeof(float);
                /*std::cout <<"i: " << i << " " << mArraySizes[i] << std::endl;
                std::cout << mEngine->getBindingDimensions(i).d[0] << " " << mEngine->getBindingDimensions(i).d[1] << " " << mEngine->getBindingDimensions(i).d[2] << " " << mEngine->getBindingDimensions(i).d[3] << " " << std::endl;*/

                mArraySizes[i] = binding_size;
                cudaMalloc(&mBuffers[i], binding_size);
                if (mEngine->bindingIsInput(i))
                    mInputDims.emplace_back(mEngine->getBindingDimensions(i));
                else
                    mOutputDims.emplace_back(mEngine->getBindingDimensions(i));
            }
            mInputBufferCpu = new float[mArraySizes[0]];
            AIO_LOGGER_INFO("DONE @arcface_ort.Start()");
        }
        catch (std::exception& e)
        {
            AIO_LOGGER_ERROR("{0}: {1}", this->Name(), e.what());
            return false;
        }

        return true;
    }

    template<typename SpTDatum>
    cv::Mat FRtrtNet<SpTDatum>::alignFace(cv::Mat img, std::vector<float> landmark)
    {
        float v2[5][2] =
        {
            {landmark[0], landmark[1]},
            {landmark[2], landmark[3]},
            {landmark[4], landmark[5]},
            {landmark[6], landmark[7]},
            {landmark[8], landmark[9]}
        };

        float norm_face[5][2] =
        {
            { 30.2946f + 8.0, 51.6963f },
            { 65.5318f + 8.0, 51.5014f },
            { 48.0252f + 8.0, 71.7366f },
            { 33.5493f + 8.0, 92.3655f },
            { 62.7299f + 8.0, 92.2041f }
        };

        cv::Mat dst(5, 2, CV_32FC1, v2);
        cv::Mat src(5, 2, CV_32FC1, norm_face);
        cv::Mat m = similarTransform(dst, src);
        cv::Mat aligned(112, 112, CV_32FC3);
        cv::Size size(112, 112);

        cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));
        cv::warpAffine(img, aligned, transfer, size, 1, 0, 0);

        return aligned;
    }

    //template<typename SpTDatum>
    //std::vector<float> FRtrtNet<SpTDatum>::prepareImage(std::vector<cv::Mat>& vec_img, std::vector<std::vector<float>>& vec_lmk) {
    //    std::vector<float> result(mBatchSize * 112 * 112 * 3);
    //    float* data = result.data();
    //    int index = 0;
    //    //for (const cv::Mat& src_img : vec_img)
    //    for (int i = 0; i < vec_img.size(); i++)
    //    {
    //        cv::Mat pr_img = alignFace(vec_img[i], vec_lmk[i]);
    //        pr_img = (pr_img - 127.5) / 127.5;
    //        //HWC TO CHW
    //        /*std::vector<cv::Mat> split_img(3);
    //        cv::split(pr_img, split_img);*/

    //        int channelLength = 112 * 112;
    //        //HWC TO CHW
    //        std::vector<cv::Mat> split_img = {
    //                cv::Mat(112, 112, CV_32FC1, data + channelLength * (index + 2)),
    //                cv::Mat(112, 112, CV_32FC1, data + channelLength * (index + 1)),
    //                cv::Mat(112, 112, CV_32FC1, data + channelLength * index)
    //        };
    //        index += 3;
    //        cv::split(pr_img, split_img);
    //    }
    //    return result;
    //}

    template<typename SpTDatum>
    void FRtrtNet<SpTDatum>::Forward(std::vector<SpTDatum>& the_data)
    {
        //int index = 0;
        //int batch_id = 0;
        //std::vector<cv::Mat> vec_Mat(mBatchSize);
        ////std::vector<std::vector<float>> vec_lmk(mBatchSize);
        //std::vector<cv::Mat> vec_emb;

        for (int b = 0; b < the_data.size(); b++)
        {
            if (the_data[b]->results->empty()) continue;
            //std::chrono::system_clock::time_point a = std::chrono::system_clock::now();
            for (int i = 0; i < the_data[b]->results->size(); i++)
            {
                //std::cout << "b " << b << " i " << i << std::endl;

                the_data[b]->results->at(i).queueNumber = i;

                //index++;

                //if (the_data[b]->cvInputData.data)
                //{
                //    vec_Mat[batch_id] = the_data[b]->cvInputData.clone();
                //    vec_lmk[batch_id] = the_data[b]->results->at(i).landmark;
                //    batch_id++;
                //}
                //if (batch_id == mBatchSize or index == the_data.size())
                //{
                //    std::vector<float>curInput = prepareImage(vec_Mat, vec_lmk);
                //    batch_id = 0;
                //    if (!curInput.data()) {
                //        std::cout << "prepare images ERROR!" << std::endl;
                //        continue;
                //    }

                //    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
                //    cudaMemcpyAsync(mBuffers[0], curInput.data(), mArraySizes[0], cudaMemcpyHostToDevice, mCudaStream);

                //    // do inference
                //    mContext->execute(mBatchSize, mBuffers.data());

                //    int outSize = int(mArraySizes[1] / sizeof(float) / mBatchSize);
                //    auto* out = new float[outSize * mBatchSize];

                //    cudaMemcpyAsync(out, mBuffers[1], mArraySizes[1], cudaMemcpyDeviceToHost, mCudaStream);
                //    cudaStreamSynchronize(mCudaStream);

                //    /*int rowSize = index % mBatchSize == 0 ? mBatchSize : index % mBatchSize;
                //    cv::Mat feature(rowSize, outSize, CV_32FC1);
                //    ReshapeandNormalize(out, feature, rowSize, outSize);*/

                //    for (int j = 0; j < (int)vec_Mat.size(); j++)
                //    {
                //        cv::Mat emb_sliced = cv::Mat(1, outSize, CV_32FC1, out + j * outSize);
                //        cv::normalize(emb_sliced, emb_sliced);

                //        vec_emb.push_back(emb_sliced.clone());
                //    }

                //    delete[] out;
                //    vec_Mat = std::vector<cv::Mat>(mBatchSize);
                //    vec_lmk = std::vector<std::vector<float>>(mBatchSize);
                //}

                //index++;
                //if (the_data[b]->cvInputData.data)
                //{
                //    //vec_Mat[batch_id] = the_data[b]->cvInputData.clone();
                //    cv::Mat pr_img = alignFace(the_data[b]->cvInputData.clone(), the_data[b]->results->at(i).landmark);
                //    vec_Mat.push_back(pr_img.clone());
                //    batch_id++;
                //}
                //if (batch_id == mBatchSize or index == the_data.size() * the_data[b]->results->size())
                //{
                //    std::cout << batch_id << std::endl;
                //    for (int bat = 0; bat < mBatchSize; bat++) {
                //        for (int j = 0; j < 112 * 112; j++) {
                //            mInputBufferCpu[bat * 3 * 112 * 112 + j] = (vec_Mat[bat].at<cv::Vec3b>(j)[2] - 127.5f) / 127.5f;
                //            mInputBufferCpu[bat * 3 * 112 * 112 + j + 112 * 112] = (vec_Mat[bat].at<cv::Vec3b>(j)[1] - 127.5f) / 127.5f;
                //            mInputBufferCpu[bat * 3 * 112 * 112 + j + 2 * 112 * 112] = (vec_Mat[bat].at<cv::Vec3b>(j)[0] - 127.5f) / 127.5f;
                //        }
                //    }
                //    //std::cout << "2" << std::endl;
                //    cudaMemcpyAsync((float*)mBuffers[0], mInputBufferCpu, mArraySizes[0], cudaMemcpyHostToDevice, mCudaStream);

                //    mContext->enqueue(mBatchSize, mBuffers.data(), mCudaStream, nullptr);

                //    int outSize = int(mArraySizes[1] / sizeof(float) / mBatchSize);
                //    //std::cout << outSize << std::endl;
                //    std::vector<float> cpu_output(outSize * mBatchSize);
                //    cudaMemcpyAsync(cpu_output.data(), (float*)mBuffers[1], mArraySizes[1], cudaMemcpyDeviceToHost, mCudaStream);
                //    cudaStreamSynchronize(mCudaStream);

                //    for (int j = 0; j < (int)vec_Mat.size(); j++)
                //    {
                //        cv::Mat emb_sliced = cv::Mat(1, 512, CV_32FC1, cpu_output.data() + j * 512);
                //        if (!emb_sliced.empty())
                //        {
                //            cv::normalize(emb_sliced, emb_sliced);
                //            vec_emb.push_back(emb_sliced.clone());
                //        }
                //    }
                //    vec_Mat = std::vector<cv::Mat>(mBatchSize);
                //    batch_id = 0;
                //    std::cout << "here" << std::endl;
                //}

                std::vector<float> landmark = the_data[b]->results->at(i).landmark;
                cv::Mat frame = the_data[b]->cvInputData.clone();
                //cv::Rect r = the_data[b]->results->at(i).bbox;
                /*for (const auto& l : landmark)
                    std::cout << l << std::endl;
                std::cout << std::endl;*/

                cv::Mat pr_img = alignFace(frame, landmark);
                /*cv::imshow("pr_img", pr_img);
                cv::waitKey(0);*/

                for (int j = 0; j < mNetDimension.height * mNetDimension.width; j++) {
                    mInputBufferCpu[j] = (pr_img.at<cv::Vec3b>(j)[2] - 127.5f) / 127.5f;
                    mInputBufferCpu[j + mNetDimension.height * mNetDimension.width] = (pr_img.at<cv::Vec3b>(j)[1] - 127.5f) / 127.5f;
                    mInputBufferCpu[j + 2 * mNetDimension.height * mNetDimension.width] = (pr_img.at<cv::Vec3b>(j)[0] - 127.5f) / 127.5f;
                }

                cudaMemcpyAsync((float*)mBuffers[0], mInputBufferCpu, mArraySizes[0], cudaMemcpyHostToDevice, mCudaStream);

                mContext->enqueue(mBatchSize, mBuffers.data(), mCudaStream, nullptr);

                int outSize = int(mArraySizes[1] / sizeof(float) / mBatchSize);
                std::vector<float> cpu_output(outSize*mBatchSize);
                cudaMemcpyAsync(cpu_output.data(), (float*)mBuffers[1], mArraySizes[1], cudaMemcpyDeviceToHost, mCudaStream);
                cudaStreamSynchronize(mCudaStream);

                cv::Mat emb_sliced = cv::Mat(1, 512, CV_32FC1, cpu_output.data());
                cv::normalize(emb_sliced, emb_sliced);
                cv::Mat out_norm = emb_sliced.clone();

                the_data[b]->results->at(i).faceEmbedding = out_norm.clone();
            }
            the_data[b]->finished = true;

            /*std::chrono::system_clock::time_point c = std::chrono::system_clock::now();
            the_data[b]->duration += std::chrono::duration_cast<std::chrono::milliseconds>(c - a).count();*/
            //AIO_LOGGER_INFO("Total Duration @ ARCFACE: {0}", the_data[b]->duration);
        }
        //AIO_LOGGER_TRACE("DONE @arcface_ort.Forward-End()");

        //int cnt = 0;
        ///*std::cout << "data size: " << the_data.size() << std::endl;
        //std::cout << "result size: " << the_data[0]->results->size() << std::endl;*/
        //for (int b = 0; b < the_data.size(); b++)
        //{
        //    for (int i = 0; i < the_data[b]->results->size(); i++)
        //    {
        //        the_data[b]->results->at(i).faceEmbedding = vec_emb[cnt].clone();
        //        cnt += 1;
        //    }
        //    the_data[b]->finished = true;
        //}
        ////std::cout << cnt << std::endl;
        //vec_emb = std::vector<cv::Mat>();
    }

    /*template<typename SpTDatum>
    void FRtrtNet<SpTDatum>::ReshapeandNormalize(float* out, cv::Mat& feature, const int& MAT_SIZE, const int& outSize) {
        for (int i = 0; i < MAT_SIZE; i++)
        {
            cv::Mat onefeature(1, outSize, CV_32FC1, out + i * outSize);
            cv::normalize(onefeature, onefeature);
            onefeature.copyTo(feature.row(i));
        }
    }*/

    template<typename SpTDatum>
    bool FRtrtNet<SpTDatum>::IsReady()
    {
        try
        {
            AIO_LOGGER_INFO("Starting @arcface_ort.IsReady()");
            WarmUp(1);
        }
        catch (std::exception& e) {
            AIO_LOGGER_ERROR("{0}: {1}", this->Name(), e.what());
            return false;
        }
        return true;
    }

    template<typename SpTDatum>
    float FRtrtNet<SpTDatum>::GetScoreThreshold() {
        AIO_LOGGER_WARN("{0}: Does not have a score threshold mechanism", this->Name());
        return 0.0f;
    }

    template<typename SpTDatum>
    void FRtrtNet<SpTDatum>::SetScoreThreshold(float the_thresh) {
        AIO_LOGGER_WARN("{0}: Does not have a score threshold mechanism", this->Name());
    }

    template<typename SpTDatum>
    std::string FRtrtNet<SpTDatum>::Name()
    {
        return "ArcFace_glintr100_Net";
    }
    template<typename SpTDatum>
    std::string FRtrtNet<SpTDatum>::Version()
    {
        std::string version_str = "Model: " + this->Name() + "\n";
        version_str += "Version: " + mModelVersion + "-Onnxruntime_TensorRT\n";
        return version_str;
    }

    template<typename SpTDatum>
    unsigned char FRtrtNet<SpTDatum>::CheckType()
    {
        return 2;
    }

    template<typename SpTDatum>
    void FRtrtNet<SpTDatum>::LoadConfig(const std::string& path)
    {
        std::ifstream file(path);
        std::vector<std::string> data;
        std::string line, str;
        while (std::getline(file, line))
        {//collect the data
            str = aio::StringSplit(line, '=')[1];
            data.push_back(str);
        }
        mModelVersion = data[0];
        mFRPath = data[1];
    }

    template<typename SpTDatum>
    void FRtrtNet<SpTDatum>::WarmUp(int n_times)
    {
        AIO_LOGGER_INFO("Starting @arcface_ort.WarmUp()");
        //cudaSetDevice(mGpuId);//set the active GPU device
        //cv::Mat pr_img(mNetDimension, CV_32FC3, cv::Scalar(0, 0, 0));
        //for (int i = 0; i < mNetDimension.height * mNetDimension.width; i++) {
        //    mInputBufferCpu[i] = float(pr_img.at<cv::Vec3b>(i)[2]) / 255.0;
        //    mInputBufferCpu[i + mNetDimension.height * mNetDimension.width] = float(pr_img.at<cv::Vec3b>(i)[1]) / 255.0;
        //    mInputBufferCpu[i + 2 * mNetDimension.height * mNetDimension.width] = float(pr_img.at<cv::Vec3b>(i)[0]) / 255.0;
        //}
        //cudaMemcpyAsync((float*)mBuffers[0], mInputBufferCpu, mArraySizes[0] * sizeof(float), cudaMemcpyHostToDevice, mCudaStream);
        //for (size_t i = 0; i < n_times; ++i)// inference
        //    mContext->enqueue(mBatchSize, mBuffers.data(), mCudaStream, nullptr);
    }

    COMPILE_TEMPLATE_DATUM(FRtrtNet);
    DEFINE_TEMPLATE_DATUM(FRtrtNet);
}
