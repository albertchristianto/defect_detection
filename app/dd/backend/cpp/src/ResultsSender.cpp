//our headers
#include <nf/utilities/logger.hpp>
#include <nf/async/worker/wDataSender.hpp>

#include "cWrapper.h"
#include "ApiImgInfer.hpp"

namespace dd {
    bool CollectResults(BASE_DATUM_SP& the_datum, C_Results& results) {
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