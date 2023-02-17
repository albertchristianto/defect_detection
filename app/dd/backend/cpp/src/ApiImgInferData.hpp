
#ifndef API_IMG_INFER_DATA_HPP
#define API_IMG_INFER_DATA_HPP

#include <mutex>
#include <queue>
#include <memory>

#include "Datum.hpp"

namespace dd {
    struct ApiImgInferData {
        unsigned long long WorkerId;//datasender worker id
        int ApiId;
        std::vector<BASE_QUEUE_SP> InputQueues;
        BASE_QUEUE_SP WaitQueue; // queue to wait the results
        BASE_THREAD_SP SenderThread;
    };
    #define API_IMG_INFER_DATA dd::ApiImgInferData
    #define API_IMG_INFER_DATA_SP std::shared_ptr<API_IMG_INFER_DATA>
}
#endif