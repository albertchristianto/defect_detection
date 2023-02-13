#include "fdasfr_c_wrapper.h"
#include "manager.hpp"
#include <aio/logger.hpp>

namespace FDASFR_AIO 
{
    static std::unique_ptr<Manager> fdasfr_system;
    //std::shared_ptr<FDASFR_AIO::database> db;

    int Start(SystemParam& config) 
    {
        //db = std::make_shared<FDASFR_AIO::database>("database/face.db");
        fdasfr_system = std::make_unique<FDASFR_AIO::Manager>();//create the object
        return fdasfr_system->Init(config);//start the process of the fdasfr system manager
    }

    void Stop() 
    {
        if (fdasfr_system != nullptr)
            fdasfr_system.reset();
    }

    int Initialize(SystemParam& config) {
        return Start(config);
    }

    int Restart(SystemParam& config)
    {
        Stop();
        return Start(config);
    }

    int Terminate() {
        Stop();
        return 1;
    }

    int IsReady() {
        if (fdasfr_system != nullptr) {
            try {
                return int(fdasfr_system->IsReady());
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }

    int AddApi()
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->AddApi();
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }

    int AddApiFuncPtr(int(*hits_func_ptr)(C_Results&), int(*nots_func_ptr)(C_Results&))
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->AddApiFuncPtr(hits_func_ptr, nots_func_ptr);
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }

    int DeleteApi(int api_id)
    {
        if (fdasfr_system != nullptr) {
            try {
                fdasfr_system->DeleteApi(api_id);
                return 1;
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }

    int SendImage(int api_id, C_Image& image, C_FilterParameters& filter)
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->SendImage(api_id, image, filter);
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }

    int GetResults(int api_id, C_Results& result)
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->GetResults(api_id, result);
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }

    int GetResultsWithImage(int api_id, C_Image& image, C_Results& results)
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->GetResults(api_id, image, results);
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }

    int SearchImage(int api_id, C_Image& input, C_FilterParameters& filter, 
        C_DBResults& dbresults)
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->SearchImage(api_id, input, filter, dbresults);
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }

    int DeleteImg(unsigned char* img_ptr) {
        try {
            if (img_ptr != nullptr)
                delete[] img_ptr;
            return 1;
        }
        catch (std::exception& e) {
            AIO_LOGGER_ERROR(e.what());
        }
        return -1;
    }

    int EnrollPath(int api_id, const char* name, const char* uuid, const char* imgPath, 
        C_FilterParameters& filter, int& id)
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->Enroll(api_id, uuid, name, imgPath, filter, id);
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }
    int EnrollImage(int api_id, const char* name, const char* uuid, C_Image& image, 
        C_FilterParameters& filter, int& id)
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->Enroll(api_id, uuid, name, image, filter, id);
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }
    int DeEnroll(int api_id, int id)
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->DeEnroll(api_id, id);
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }
    int UpdateUser(int api_id, int id, const char* name, const char* uuid, C_Image& image,
        const char* timestamp, C_FilterParameters& filter)
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->UpdateUser(api_id, id, name, uuid, image, timestamp, filter);
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }
    int RemoveHit(int id)
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->RemoveHit(id);
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }
    int RemoveNot(int id)
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->RemoveNot(id);
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }
    int DeEnrollSome(int api_id, C_IntArray ids)
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->DeEnroll_Some(api_id, ids);
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }
    int RemoveSomeHits(C_IntArray ids)
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->RemoveSomeHits(ids);
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }
    int RemoveSomeNots(C_IntArray ids)
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->RemoveSomeNots(ids);
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }
    int DeEnrollAll(int api_id)
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->DeEnroll_All(api_id);
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }
    int RemoveAllHits()
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->RemoveAllHits();
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }
    int RemoveAllNots()
    {
        if (fdasfr_system != nullptr) {
            try {
                return fdasfr_system->RemoveAllNots();
            }
            catch (std::exception& e) {
                AIO_LOGGER_ERROR(e.what());
            }
        }
        return -1;
    }

    C_SystemInfo GetInfo()
    {
        return fdasfr_system->GetInfo();
    }

#ifdef USE_LOGGER_API
    void Info(const char* str) {
        AIO_LOGGER_INFO("FDASFR log: {0}", str);
    }

    void Error(const char* str) {
        AIO_LOGGER_ERROR("FDASFR log: {0}", str);
    }

    void Trace(const char* str) {
        AIO_LOGGER_TRACE("FDASFR log: {0}", str);
    }
#endif
}
