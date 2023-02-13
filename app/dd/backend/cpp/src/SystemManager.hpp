#pragma once
#include "macros.hpp"
#include <FDASFR_aio/fdasfr.hpp>
#include <FDASFR_aio_channel/imgInfer.hpp>
#include "fdasfr_c_wrapper.h"
#include <dbConsumer_aio/database.hpp>
#include <dbConsumer_aio/consumer.hpp>

namespace dd {
    class InferenceManager {
    public:
        Manager();
        ~Manager();
        int Init(SystemParam& param);
        void Restart();
        void Stop();
        bool IsReady();
        double CheckFPS(int what_to_check);

        int AddApiFuncPtr(int (*hits_func_ptr)(C_Results&));
        void DeleteApi(int api_id);
        int SendImage(int api_id, C_Image& image, C_FilterParameters& c_fParam);

    private:

        unsigned long long mWorker_id;//counter for the worker object
        unsigned long long mSubThread_id;//counter for the subthread object
        unsigned long long mThread_id;//counter for the thread object
        bool mDrawInfo;
        bool mDrawBbox;
        std::deque<int> mAvailApiIds;//available id for an api
        std::mutex mMtx;
        //std::mutex mMtxR;
        bool mValidLicense;

        std::shared_ptr<database> mDatabase;
        std::vector<unsigned char> mHaspData;
        std::shared_ptr<Inference::FDASFR> mFdasfrEngine;
        std::vector<std::shared_ptr<ImgInfer>> mImgInfers;
        std::vector<std::shared_ptr<ImgInferParam>> mImgInferDatas;//api information
        std::vector<BASE_THREAD_SP> mSenderThreads;

        aio::FpsCounter mSendImageCounter;

#ifdef USE_PROTECTION
        void CheckLicense();
#endif
    };
}