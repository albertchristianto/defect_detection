#ifndef RESULTS_SENDER_HPP
#define RESULTS_SENDER_HPP
//our headers
#include <nf/utilities/logger.hpp>
#include <nf/async/worker/wDataSender.hpp>

#include "cWrapper.h"
#include "ApiImgInfer.hpp"

namespace dd {
    class ResultsSender : public nf::async::WDataSender<C_Results, BASE_DATUM, BASE_DATUM_SP>
    {
    public:
        explicit ResultsSender(unsigned long long worker_id, int (*func_ptr)(C_Results&), std::shared_ptr<ImgInfer>& img_infer_api);
        ~ResultsSender() {}

    private:
        bool CollectResults(BASE_DATUM_SP& the_datum, C_Results& results);
        std::shared_ptr<ImgInfer> m_ImgInfer;

        DELETE_COPY(ResultsSender);
    };
}
#endif