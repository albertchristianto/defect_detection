#ifndef DATUM_HPP
#define DATUM_HPP
#include "macros.hpp"
#include <memory>
#include <chrono>
#include <atomic>
#include <string>
#include <nf/inference_core/i_InferEngine.hpp>
#include <nf/async/queue/i_Queue.hpp>
#include <nf/async/worker/i_Worker.hpp>
#include <nf/async/sub_thread/i_SubThread.hpp>
#include <nf/async/thread.hpp>
#include <opencv2/opencv.hpp>

namespace dd {
    struct DD_BACKEND_API Datum {
        //This is the general datum
        //////////////////////////////////////////
        /// PLEASE DON'T USE NAKED POINTERS    ///
        /// PLEASE USE STD OR BASIC DATA TYPES ///
        //////////////////////////////////////////

        unsigned long long timeStamp;//data tag
        std::chrono::time_point<std::chrono::steady_clock> spawnTime;//for compute system's latency
        cv::Mat cvInputData;//frame data => make sure that the data isn't modified by the system
        std::string className;//image classification results
        float confScore;
        //faster data access
        std::atomic<bool> finished;//variable to give a finished signal

        Datum();
        ~Datum();
        Datum(const Datum& datum);
        Datum& operator=(const Datum& datum);
        Datum(Datum&& datum);
        Datum& operator=(Datum&& datum);
        Datum clone() const;
    };

    #define BASE_DATUM dd::Datum
    #define BASE_DATUM_SP std::shared_ptr<BASE_DATUM>

    #define DEFINE_TEMPLATE_DATUM(templateName) template class templateName<BASE_DATUM_SP>
    #define COMPILE_TEMPLATE_DATUM(templateName) extern template class templateName<BASE_DATUM_SP>

    #define DEFINE_TEMPLATE_DATUM2(templateName) template class templateName<BASE_DATUM, BASE_DATUM_SP>
    #define COMPILE_TEMPLATE_DATUM2(templateName) extern template class templateName<BASE_DATUM, BASE_DATUM_SP>

    #define BASE_ENGINE nf::I_InferEngine<BASE_DATUM_SP>
    #define BASE_ENGINE_SP std::shared_ptr<BASE_ENGINE>
    #define BASE_QUEUE nf::async::I_Queue<BASE_DATUM_SP>
    #define BASE_QUEUE_SP std::shared_ptr<BASE_QUEUE>
    #define BASE_WORKER nf::async::I_Worker<BASE_DATUM_SP>
    #define BASE_WORKER_SP std::shared_ptr<BASE_WORKER>
    #define BASE_SUBTHREAD nf::async::I_SubThread<BASE_DATUM_SP, BASE_WORKER_SP>
    #define BASE_SUBTHREAD_SP std::shared_ptr<BASE_SUBTHREAD>
    #define BASE_THREAD nf::async::Thread<BASE_DATUM_SP, BASE_WORKER_SP, BASE_SUBTHREAD_SP>
    #define BASE_THREAD_SP std::shared_ptr<BASE_THREAD>
}

#endif