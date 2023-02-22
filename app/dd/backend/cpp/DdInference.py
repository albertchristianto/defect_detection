#!python3
from ctypes import *
import os

class C_Image(Structure):
    _fields_ = [("TimeStamp", c_ulonglong),
                ("Ptr", POINTER(c_ubyte)),
                ("Height", c_int),
                ("Width", c_int),
                ("Depth", c_int)]

class C_Results(Structure):
    _fields_ = [("TimeStamp", c_ulonglong),
                ("ClassName", c_char * 10)]

SEND_RESULTS = CFUNCTYPE(c_int, C_Results)

def GetTheAPI():
    abs_path_dll = os.path.join(os.path.abspath(os.getcwd()), "build")
    os.environ['PATH'] = abs_path_dll + os.pathsep + os.environ['PATH']
    if os.name == "nt":
        lib = CDLL("./build/DdInference.dll", winmode=0)
    else:
        lib = CDLL("./build//DdInference.so", winmode=0)

    #load all the functions
    lib.Initialize.argtypes = []
    lib.Initialize.restype = c_int
    lib.Restart.argtypes = []
    lib.Restart.restype = c_int
    lib.Terminate.argtypes = []
    lib.Terminate.restype = c_int
    lib.IsReady.argtypes = []
    lib.IsReady.restype = c_int
    lib.CheckFPS.argtypes = [c_int]
    lib.CheckFPS.restype = c_double
    lib.AddApiFuncPtr.argtypes = [SEND_RESULTS]
    lib.AddApiFuncPtr.restype = c_int
    lib.DeleteApi.argtypes = [c_int]
    lib.DeleteApi.restype = c_int
    lib.SendImage.argtypes = [c_int, C_Image]
    lib.SendImage.restype = c_int
    lib.Info.argtypes = [POINTER(c_char)]
    lib.Info.restype = None
    lib.Error.argtypes = [POINTER(c_char)]
    lib.Error.restype = None
    lib.Trace.argtypes = [POINTER(c_char)]
    lib.Trace.restype = None
    return lib
if __name__ == "__main__":
    LIB = GetTheAPI()
    LIB.Initialize()
    LIB.Trace(b"Test")
    LIB.Restart()