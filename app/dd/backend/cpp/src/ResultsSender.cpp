//our headers
#include "ResultsSender.hpp"

namespace dd {
    ResultsSender::ResultsSender(unsigned long long worker_id, int (*func_ptr)(C_Results&), std::shared_ptr<ImgInfer>& img_infer_api):
        nf::async::WDataSender<C_Results, BASE_DATUM, BASE_DATUM_SP>{ worker_id, func_ptr },
        m_ImgInfer{ img_infer_api }
    {}
    bool ResultsSender::CollectResults(BASE_DATUM_SP& the_datum, C_Results& results) {
        if (!m_ImgInfer->GetResults(the_datum))
            return false;
        results.TimeStamp = the_datum->timeStamp;
#if _WIN32
        strcpy_s(results.ClassName, the_datum->className.c_str());
#elif __linux__
        strcpy(results.ClassName, the_datum->className.c_str());
#endif
        return true;
    }
}