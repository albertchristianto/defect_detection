#include "InferenceManager.hpp"

#include <nlohmann/json.hpp>
#include <nf/utilities/logger.hpp>

#include <nf/async/queue/ringQueue.hpp>
#include <nf/async/worker/wInference.hpp>
#include <nf/async/sub_thread/appSinkInfer.hpp>

namespace dd {
    int InferenceManager::Init() {
        NF_LOGGER_TRACE("Init {0}!", this->Name());
        unsigned long long thread_id = 0;
        unsigned long long subthread_id = 0;
        unsigned long long worker_id = 0;

        if (!LoadEngineSettings()) //1. load the configuration path setting
            return -6;

        try {//2. create the system
            m_ImgClassifier_QM = std::make_shared<QNM>(m_DefaultNumberOfImgClassifier);
            for (int i = 0;i < m_DefaultNumberOfImgClassifier; ++i) {
                m_Queues.push_back(std::make_shared<nf::async::RingQueue<BASE_DATUM_SP>>(m_DefaultEachBufferSize[i]));
                m_Threads.push_back(std::make_shared<nf::async::Thread<BASE_DATUM_SP, BASE_WORKER_SP, BASE_SUBTHREAD_SP>>(thread_id));
                BASE_ENGINE_SP the_engine = std::make_shared<dd::ImageClassifier<BASE_DATUM_SP>>(m_Configs[0], m_DefaultEachGpuId[i]);
                FPS_COUNTER_SP the_fps = std::make_shared<FPS_COUNTER>();
                BASE_WORKER_SP the_worker = std::make_shared<nf::async::WInference<BASE_DATUM, BASE_DATUM_SP>>(worker_id, the_fps, the_engine);
                BASE_SUBTHREAD_SP the_subthread = std::make_shared<nf::async::AppSinkInfer<BASE_DATUM_SP, BASE_WORKER_SP, BASE_QUEUE_SP>>(subthread_id,
                        m_DefaultEachBatchSize[i], std::vector<BASE_WORKER_SP>{ the_worker }, m_Queues[i]);
                the_subthread->EnableStandBy();
                m_Threads[thread_id]->Add(the_subthread);
                //update the id
                worker_id++;
                subthread_id++;
                thread_id++;

                m_Fps.push_back(the_fps);
                m_ImgClassifier.push_back(the_engine);
            }
        }
        catch (std::exception& e) {
            NF_LOGGER_ERROR("Failed to create {0}!", this->Name());
            NF_LOGGER_ERROR(e.what());
            return -1;
        }
        {//3. Init the system thread
            bool res = true;
            for (auto& the_thread : m_Threads)
                res &= the_thread->Init();
            if (!res) {
                NF_LOGGER_ERROR("Failed to initialize {0}!", this->Name());
                return -2;
            }
        }
        try {//4. Start the system thread
            for (auto& the_thread : m_Threads)
                the_thread->Start();
        }
        catch (std::exception& e) {
            NF_LOGGER_ERROR("Failed to start {0}!", this->Name());
            NF_LOGGER_ERROR(e.what());
            return -3;
        }

        NF_LOGGER_INFO("Successfully starting up {0} and ready to be used!", this->Name());
        return 1;
    }

    void InferenceManager::Stop() { //stop the service
        for (auto& the_thread : m_Threads)
            the_thread.reset();
        for (auto& t : m_ImgClassifier)
            t.reset();
    }

    bool InferenceManager::IsReady() {
        if (m_Threads.size() == 0)
            return false;

        bool res = true;
        for (auto& the_thread : m_Threads)
            res &= the_thread->IsRunning();

        return res;
    }

    double InferenceManager::GetFps() {//get the Fps value
        double tmp = 0;
        for (int j = 0; j < m_Fps.size(); ++j)
            tmp += m_Fps[j]->Get();
        return tmp;
    }
    std::vector<BASE_QUEUE_SP> InferenceManager::GetInputQueues() {
        return m_Queues;
    }
    int InferenceManager::GetQueueIdx() {
        return int(m_ImgClassifier_QM->GetTagNumber());
    }
    bool InferenceManager::LoadEngineSettings() {
        NF_LOGGER_TRACE("{0}: Load the inference engine settings!", this->Name());
        std::string path = "cfgs/AI_System.cfg";
        if (!boost::filesystem::exists(boost::filesystem::path(path))) {
            NF_LOGGER_ERROR("{0}: Could not find the config file!!! the config file path is {1}", this->Name(), path);//throw an error
            return false;
        }
        std::ifstream file(path);
        nlohmann::json data = nlohmann::json::parse(file);
        m_Configs.push_back(data["img_classifier_cfg_path"].get<std::string>());
        m_DefaultNumberOfImgClassifier = data["spawn_engine"].get<int>();
        m_DefaultEachGpuId = data["gpu_id_spawn_engine"].get<std::vector<int>>();
        if (m_DefaultEachGpuId.size() != m_DefaultNumberOfImgClassifier) {
            NF_LOGGER_ERROR("{0}: Number of spawn engine and spesific engine gpu id does not match!", this->Name());
            return false;
        }
        m_DefaultEachBatchSize = data["batch_size_spawn_engine"].get<std::vector<int>>();
        if (m_DefaultEachBatchSize.size() != m_DefaultNumberOfImgClassifier) {
            NF_LOGGER_ERROR("{0}: Number of spawn engine and spesific engine batch size does not match!", this->Name());
            return false;
        }
        m_DefaultEachBufferSize = data["buffer_size"].get<std::vector<int>>();
        NF_LOGGER_INFO("{0}: Succesfully load the inference engine settings!", this->Name());

        return true;
    }
}