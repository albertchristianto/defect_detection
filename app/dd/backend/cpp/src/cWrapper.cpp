#include "cWrapper.h"
#include "SystemManager.hpp"
#include <nf/utilities/logger.hpp>

namespace dd {
    static std::unique_ptr<SystemManager> the_system;
    //std::shared_ptr<FDASFR_AIO::database> db;

    int Start() {
        the_system = std::make_unique<SystemManager>();//create the object
        return the_system->Init();
    }

    void Stop() {
        if (the_system != nullptr)
            the_system.reset();
    }

    int Initialize() {
        return Start();
    }

    int Restart() {
        Stop();
        return Start();
    }

    int Terminate() {
        Stop();
        return 1;
    }

    int IsReady() {
        if (the_system != nullptr) {
            try {
                return int(the_system->IsReady());
            }
            catch (std::exception& e) {
                NF_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }
    double CheckFPS(int what_to_check) {
        if (the_system != nullptr) {
            try {
                return the_system->CheckFPS(what_to_check);
            }
            catch (std::exception& e) {
                NF_LOGGER_ERROR(e.what());
            }
        }
        return 0.0;
    }

    int AddApiFuncPtr(int (*send_results)(C_Results&)) {
        if (the_system != nullptr) {
            try {
                return the_system->AddApiFuncPtr(send_results);
            }
            catch (std::exception& e) {
                NF_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }

    int DeleteApi(int api_id) {
        if (the_system != nullptr) {
            try {
                the_system->DeleteApi(api_id);
                return 1;
            }
            catch (std::exception& e) {
                NF_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }

    int SendImage(int api_id, C_Image& image) {
        if (the_system != nullptr) {
            try {
                return the_system->SendImage(api_id, image);
            }
            catch (std::exception& e) {
                NF_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }

    void Info(const char* str) {
        NF_LOGGER_INFO("DD backend log: {0}", str);
    }

    void Error(const char* str) {
        NF_LOGGER_ERROR("DD backend log: {0}", str);
    }

    void Trace(const char* str) {
        NF_LOGGER_TRACE("DD backend log: {0}", str);
    }
}
