#ifndef INFERENCE_MANAGER_HPP
#define INFERENCE_MANAGER_HPP

#include <vector>
#include <memory>
#include <atomic>

#include <nf/async/thread.hpp>
#include <nf/utilities/throughputCounter.hpp>
#include <nf/inference_core/queueNumberMachine.hpp>

#include "ImgClassifier.hpp"
#include "Datum.hpp"

namespace dd {
    class InferenceManager {
    public:
        ~InferenceManager() {}
        int Init();
        void Stop();
        bool IsReady();
        double GetFps();
        BASE_QUEUE_SP GetInputQueue();
        std::string Name() { return "DD_Inference_Manager"; }

    private:
        int m_DefaultNumberOfImgClassifier;
        std::vector<int> m_DefaultEachGpuId;
        std::vector<int> m_DefaultEachBufferSize;
        std::vector<int> m_DefaultEachBatchSize;
        std::vector<std::string> m_Configs;
        std::vector<BASE_QUEUE_SP> m_Queues;
        std::vector<BASE_THREAD_SP> m_Threads;
        std::vector<FPS_COUNTER_SP> m_Fps;

        std::vector<BASE_ENGINE_SP> m_ImgClassifier;
        QNM_SP m_ImgClassifier_QM;

        bool LoadEngineSettings();
    };
}
#endif