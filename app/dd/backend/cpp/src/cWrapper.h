/*! C-style wrapper for the dll. */
#ifndef C_WRAPPER_H
#define C_WRAPPER_H

#ifdef _WIN32
    #ifdef DD_INFERENCE_EXPORTS
        #define DD_INFERENCE_API __declspec(dllexport)
    #else
        #define DD_INFERENCE_API __declspec(dllimport)
    #endif
#elif __linux__ 
    #define DD_INFERENCE_API
#endif

#define C_MAX_OBJECTS 100

namespace dd {
    /**
      A struct for a C-style Image
    */
    struct C_Image {
        unsigned long long TimeStamp;
        unsigned char* Ptr; /**< unsigned char pointer to the image */
        int Height; /**< image height */
        int Width; /**< image width */
        int Depth; /**< image depth or channel */
    };
    /**
      A struct for all the inference results
    */
    struct C_Results {
        unsigned long long TimeStamp;
        char ClassName[10]; /**< the class name of the inference result */
    };
    /*!
    Initialize the Manager;
    ouput shows error, please check c# project for the meaning
    */
    extern "C" DD_INFERENCE_API int Initialize();
    /*!
    Not Ready yet, please close the system and then re-open manually for now
    */
    extern "C" DD_INFERENCE_API int Restart();
    /*!
    Terminate the manager;
    ouput shows error, please check c# project for the meaning
    */
    extern "C" DD_INFERENCE_API int Terminate();
    /*!
    Check if the manager is ready;
    ouput shows error, please check c# project for the meaning
    */
    extern "C" DD_INFERENCE_API int IsReady();
    extern "C" DD_INFERENCE_API double CheckFPS(int what_to_check);
    /*!
    Open a pipeline for the callback system
    ouput shows error, please check c# project for the meaning
    @param[in]  send_results  Callback function for the hits; the callback function must return 0;
    */
    extern "C" DD_INFERENCE_API int AddApiFuncPtr(int (*send_results)(C_Results&));
    /*!
    Delete a pipeline
    ouput shows error, please check c# project for the meaning
    @param[in]  api_id  which pipeline to be deleted
    */
    extern "C" DD_INFERENCE_API int DeleteApi(int api_id);
    /*!
    Send an image to a pipeline
    ouput shows error, please check c# project for the meaning
    @param[in]  api_id  which pipeline the function is applied on
    @param[in]  image   input image
    */
    extern "C" DD_INFERENCE_API int SendImage(int api_id, C_Image& image);
//Logging API
    extern "C" DD_INFERENCE_API void Info(const char* str);
    extern "C" DD_INFERENCE_API void Error(const char* str);
    extern "C" DD_INFERENCE_API void Trace(const char* str);

}

#endif