#include <nf/inference_core/i_InferEngine.hpp>

namespace dd {
    /// This interface class is used to control a inference engine.
    /// The user of this framework must inherit their deep learning engine implementtation from this class.
    template<typename SpTDatum>
    class ImageClassifier: nf::I_InferEngine<SpTDatum> {
    public:
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
        ImageClassifier<SpTDatum>::~ImageClassifier() {
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
        /**
         * Function to do the inference. the user of this framework must implement this functions.
         * @param the_datas batch data which to be processed
         */
        void Forward(std::vector<SpTDatum> &the_datas) {
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
    };
}