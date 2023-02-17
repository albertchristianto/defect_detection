#ifndef SYSTEM_MANAGER_HPP
#define SYSTEM_MANAGER_HPP

#include "cWrapper.h"
#include "InferenceManager.hpp"
#include "ApiImgInfer.hpp"

namespace dd {
    class SystemManager {
    public:
        SystemManager();
        ~SystemManager();
        int Init();
        //void Restart();
        void Stop();
        bool IsReady();
        double CheckFPS(int what_to_check);

        int AddApiFuncPtr(int (*send_results)(C_Results&));
        void DeleteApi(int api_id);
        int SendImage(int api_id, C_Image& image);
        std::string Name() { return "DD_System_Manager"; }

    private:
        bool LoadSystemConfig();
        unsigned long long m_Worker_id;//counter for the worker object
        unsigned long long m_SubThread_id;//counter for the subthread object
        unsigned long long m_Thread_id;//counter for the thread object

        int m_MaxAPIs;
        std::deque<int> m_AvailApiIds;//available id for an api
        std::mutex m_Mtx;

        std::shared_ptr<InferenceManager> m_InferEngine;
        std::vector<std::shared_ptr<ImgInfer>> m_ImgInfers;
        std::vector<API_IMG_INFER_DATA_SP> m_ImgInferDatas;//api information

        FPS_COUNTER m_SendImageCounter;
    };
}
#endif