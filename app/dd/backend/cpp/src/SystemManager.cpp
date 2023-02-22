#include "SystemManager.hpp"

#include <nlohmann/json.hpp>

#include <nf/utilities/logger.hpp>
#include <nf/async/queue/standardQueue.hpp>
#include <nf/async/sub_thread/standAlone.hpp>
#include "InferenceManager.hpp"
#include "ApiImgInfer.hpp"
#include "ResultsSender.hpp"

#define LOG_LEVEL NF_LOGGER_LEVEL_TRACE
#define SHOW_IN_CMD_PROMPT true //set it to false if you want to 

namespace dd {
    SystemManager::SystemManager() {
        if (!(NF_LOGGER_INITIALIZE()))
            NF_LOGGER_INIT(SHOW_IN_CMD_PROMPT, LOG_LEVEL);// Initialize the logger system. IT IS A MUST TO ADD THIS LINE OF CODE!!!!
        NF_LOGGER_INFO("The defect detection system is started!");
    }

    SystemManager::~SystemManager() {
        this->Stop();
        NF_LOGGER_INFO("The defect detection system is closed!");
        if (NF_LOGGER_INITIALIZE())
            NF_LOGGER_CLOSE();// Close the logger system. IT IS A MUST TO ADD THIS LINE OF CODE!!!!
    }
    int SystemManager::Init() {
        if (!LoadSystemConfig())
            return -4;
        m_InferEngine = std::make_shared<dd::InferenceManager>();

        for (int i = 0; i < m_MaxAPIs; ++i) {//create available channel api
            m_AvailApiIds.push_back(i);//set the available id for api

            API_IMG_INFER_DATA_SP img_infer_data;
            m_ImgInferDatas.push_back(img_infer_data);
            std::shared_ptr<ImgInfer> img_infer;
            m_ImgInfers.push_back(img_infer);
        }

        return m_InferEngine->Init();
    }

    // void SystemManager::Restart() {
    //     const std::lock_guard<std::mutex> lock{ m_Mtx };
    //     NF_LOGGER_TRACE("{0}: Attempting to restart the engine!", Name());

    //     for (int i = 0; i < m_ImgInferDatas.size(); ++i) {
    //         if (m_ImgInfers[i] != nullptr)
    //             m_ImgInfers[i]->Pause();
    //     }
    //     m_InferEngine = std::make_shared<dd::InferenceManager>();
    //     m_InferEngine->Init();
    //     NF_LOGGER_TRACE("{0}: Successfully executing restart function!", Name());
    // }

    void SystemManager::Stop() {
        for (int i = 0; i < m_ImgInferDatas.size(); ++i)
            DeleteApi(i);
        if (m_InferEngine != nullptr)
            m_InferEngine->Stop();// stop the inference service
        NF_LOGGER_TRACE("{0}: Successfully executing stop function!", Name());
    }

    bool SystemManager::IsReady() {
        if (m_InferEngine != nullptr)
            return m_InferEngine->IsReady();
        return false;
    }

    double SystemManager::CheckFPS(int what_to_check) {
        if (what_to_check == 0)
            return m_SendImageCounter.Get();
        else if (what_to_check == 1)
            return m_InferEngine->GetFps();
        else
            return 0;
    }

    int SystemManager::AddApiFuncPtr(int (*send_results)(C_Results&)) {
        NF_LOGGER_TRACE("{0}: opening an API!", Name());
        int id = -1;
        bool res = true;
        {//critical section of the system
            const std::lock_guard<std::mutex> lock{ m_Mtx };
            if (!m_AvailApiIds.empty())
            {//if we still have available id for the API
                id = m_AvailApiIds.front();
                m_AvailApiIds.pop_front();
            }
        }

        if (id < 0) {
            NF_LOGGER_WARN("{0}: Can't open a new api! Maximum api is {0}", Name(), m_MaxAPIs);
            return -3;
        }

        try { //create the image inference information
            m_ImgInferDatas[id] = std::make_shared<API_IMG_INFER_DATA>();
            m_ImgInferDatas[id]->ApiId = id;
            m_ImgInferDatas[id]->WorkerId = m_Worker_id;
            m_Worker_id++;
            m_ImgInferDatas[id]->InputQueues = m_InferEngine->GetInputQueues();
            m_ImgInferDatas[id]->WaitQueue = std::make_shared<nf::async::StandardQueue<BASE_DATUM_SP>>(100);

            m_ImgInfers[id] = std::make_shared<ImgInfer>(m_ImgInferDatas[id]);
        }
        catch (std::exception& e) {
            res = false;
            NF_LOGGER_ERROR("{0}: AddApi no. {1}", Name(), e.what());
        }
        if (!res) {
            DeleteApi(id);
            NF_LOGGER_ERROR("{0}: Failed to open a new api! API id: {1}", Name(), id);
            return -1;
        }

        NF_LOGGER_INFO("{0}: Successfully opened 1 api! API id: {1}", Name(), id);
        if (id > -1) {
            m_ImgInferDatas[id]->SenderThread = std::make_shared<BASE_THREAD>(m_Thread_id);
            {//create the input reader
                BASE_WORKER_SP send_worker = std::make_shared<ResultsSender>(m_ImgInferDatas[id]->WorkerId, send_results, m_ImgInfers[id]);
                BASE_SUBTHREAD_SP send_subthread = std::make_shared<nf::async::StandAlone<BASE_DATUM_SP, BASE_WORKER_SP>>
                    (m_SubThread_id, std::vector<BASE_WORKER_SP>{ send_worker });
                m_ImgInferDatas[id]->SenderThread->Add(send_subthread);
                m_ImgInferDatas[id]->SenderThread->Init();//<-- Init the video reader thread
                m_SubThread_id++;
                m_Thread_id++;
            }
            m_ImgInferDatas[id]->SenderThread->Start();
            while (!m_ImgInferDatas[id]->SenderThread->IsRunning())
                std::this_thread::yield();
            NF_LOGGER_INFO("{0}: Successfully starting thread for executing callback function, API Id: {1}", Name() , id);
        }
        return id;
    }

    void SystemManager::DeleteApi(int api_id) {
        if (api_id < 0 && api_id >= m_MaxAPIs)
            return;
        if (m_ImgInferDatas[api_id] != nullptr) {
            NF_LOGGER_INFO("{0}: Deleting API id: {1}", Name(), api_id);
            m_ImgInfers[api_id]->Stop();
            m_ImgInferDatas[api_id]->SenderThread.reset();//stop the Sender thread
            m_ImgInfers[api_id].reset();
            m_ImgInferDatas[api_id].reset();
            size_t tmp;
            {//critical section of the system
                const std::lock_guard<std::mutex> lock{ m_Mtx };
                m_AvailApiIds.push_back(api_id);
                tmp = m_AvailApiIds.size();
            }
            NF_LOGGER_TRACE("{0}: Available ID: {1}", Name(), tmp);
        }
    }

    int SystemManager::SendImage(int api_id, C_Image& image) {
        try {
            int the_queue_id = m_InferEngine->GetQueueIdx();
            m_ImgInfers[api_id]->SendImage(image, image.TimeStamp, the_queue_id);

            m_SendImageCounter.Increment();
            m_SendImageCounter.CheckAndUpdate();
        }
        catch (std::exception& e) {
            NF_LOGGER_TRACE("{0}: API Id {1} -> {2}", Name(), api_id, e.what());
            return -3;
        }
        return 1;
    }
    bool SystemManager::LoadSystemConfig() {
        NF_LOGGER_TRACE("{0}: Load the system settings!", this->Name());
        std::string path = "cfgs/AI_System.cfg";
        if (!boost::filesystem::exists(boost::filesystem::path(path))) {
            NF_LOGGER_ERROR("{0}: Could not find the config file!!! the config file path is {1}", this->Name(), path);//throw an error
            return false;
        }
        std::ifstream file(path);
        nlohmann::json data = nlohmann::json::parse(file);
        m_MaxAPIs = data["max_api_spawn"].get<int>();
        NF_LOGGER_INFO("{0}: Succesfully load the system settings!", this->Name());

        return true;
    }
}