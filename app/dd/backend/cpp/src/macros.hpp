#ifndef DD_MACROS
#define DD_MACROS

#ifdef _WIN32
    #ifdef DD_BACKEND_EXPORTS
        #define DD_BACKEND __declspec(dllexport)
    #else
        #define DD_BACKEND __declspec(dllimport)
    #endif
#elif __linux__ 
    #define DD_BACKEND
#endif

#endif