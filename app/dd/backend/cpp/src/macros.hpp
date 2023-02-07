#ifndef DD_MACROS_HPP
#define DD_MACROS_HPP

#ifdef _WIN32
    #ifdef DD_BACKEND_EXPORTS
        #define DD_BACKEND_API __declspec(dllexport)
    #else
        #define DD_BACKEND_API __declspec(dllimport)
    #endif
#elif __linux__ 
    #define DD_BACKEND_API
#endif

#endif