#ifndef API_IMG_INFER_HPP
#define API_IMG_INFER_HPP
#include <atomic>

#include <nf/inference_core/imgInfer.hpp>
#include <opencv2/opencv.hpp>

#include "cWrapper.h"
#include "ApiImgInferData.hpp"
#include "Datum.hpp"

namespace dd {
    template <typename T>
    T* MatToDataBuffer(cv::Mat& m) {
        int rows = m.rows;
        int cols = m.cols;
        int chs = m.channels();
        T* buffer = new T[rows * cols * chs];
        memcpy(buffer, m.data, rows * cols * chs * sizeof(T));// Create buffer from Mat
        return buffer;
    }

    template <typename T>
    cv::Mat DataBufferToMat(T* data, int& rows, int& cols, int& chs) {
        cv::Mat mat(rows, cols, CV_MAKETYPE(cv::DataType<T>::type, chs));// Create Mat from buffer 
        memcpy(mat.data, data, rows * cols * chs * sizeof(T));
        return mat;
    }

    class ImgInfer: public nf::ImgInfer<C_Image, BASE_DATUM_SP, BASE_QUEUE_SP> {
    public:
        ImgInfer(API_IMG_INFER_DATA_SP& the_param);
        ~ImgInfer() {}

        bool GetResults(BASE_DATUM_SP& results);//user must implement this function
        bool CheckImage(C_Image& image);
        BASE_DATUM_SP Produce(C_Image& image, unsigned long long& tag);
        
    private:
        API_IMG_INFER_DATA_SP m_Info;
    };
}
#endif