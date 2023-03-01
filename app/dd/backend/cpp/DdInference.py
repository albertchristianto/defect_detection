#!python3
from ctypes import *
import os
import cv2
import numpy as np
import time
import ctypes

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
    abs_path_dll = os.path.join(os.path.dirname(__file__), "lib")
    os.environ['PATH'] = abs_path_dll + os.pathsep + os.environ['PATH']
    if os.name == "nt":
        lib = CDLL(os.path.join(abs_path_dll, "DdInference.dll"), winmode=0)
    else:
        lib = CDLL(os.path.join(abs_path_dll, "DdInference.so"))

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

finished = False

class CallbackWrapper:
    def __init__(self):
        # Define the C-compatible callback function
        self._c_callback = SEND_RESULTS(self.py_accept_results)

    def py_accept_results(self, results):
        # Convert the results pointer to a Python object
        # py_results = C_Results.from_address(ctypes.addressof(results.contents))

        # Print the results
        print(f"Results: {results.TimeStamp}, {results.ClassName}")
        global finished
        finished = True
        # Return 0 to indicate success
        return 0

    def get_callback(self):
        # Return the C-compatible callback function
        return self._c_callback

if __name__ == "__main__":
    LIB = GetTheAPI()
    LIB.Initialize()
    wrapper = CallbackWrapper()
    f_ptr = wrapper.get_callback()
    api_id = LIB.AddApiFuncPtr(f_ptr)
    img = cv2.imread('samples/mt_defect.jpg')
    the_image = C_Image()
    the_image.TimeStamp = 0
    the_image.Ptr = img.ctypes.data_as(POINTER(c_ubyte))
    the_image.Height, the_image.Width, the_image.Depth = img.shape
    if (LIB.SendImage(api_id, the_image) != 1):
        LIB.Terminate()
    while(not finished):
        time.sleep(0.1)
    LIB.Terminate()