#include "ApiImgInfer.hpp"

#include <nf/utilities/logger.hpp>

namespace dd {
    ImgInfer::ImgInfer(API_IMG_INFER_DATA_SP& the_param):
        nf::ImgInfer<C_Image, BASE_DATUM_SP, BASE_QUEUE_SP>{ the_param->WorkerId, the_param->InputQueues, the_param->WaitQueue },
        m_Info{ the_param }
    {}
    bool ImgInfer::GetResults(BASE_DATUM_SP& results) {
        if (m_Info->WaitQueue->Empty())
            return false;

        BASE_DATUM_SP the_datum;
        if (!m_Info->WaitQueue->Peek(the_datum))
            return false;

        if (the_datum->Finished) {//if the data is finished processed
            m_Info->WaitQueue->Pop();//pop the data
            results = the_datum;
            return true;
        }
        return false;
    }
    bool ImgInfer::CheckImage(C_Image& image) {
        if (image.Height <= 0 || image.Width <= 0 || (image.Depth <= 0 && image.Depth > 3))
            return false;
        return true;
    }
    BASE_DATUM_SP ImgInfer::Produce(C_Image& image, unsigned long long& tag) {
        BASE_DATUM_SP the_datum = std::make_shared<BASE_DATUM>();
        the_datum->timeStamp = tag;
        the_datum->spawnTime = std::chrono::steady_clock::now();
        the_datum->cvInputData = DataBufferToMat(image.Ptr, image.Height, image.Width, image.Depth);
        m_Info->WaitQueue->Send(the_datum);//send to the wait queue
        return the_datum;
    }
}