#!python3
from ctypes import *
import os
import cv2
import numpy as np
import time
from loguru import logger
import sys

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

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(__file__)
 
    return os.path.join(base_path, relative_path)

def GetTheAPI():
    logger.trace("Load the API!")
    abs_path_dll = resource_path("lib")
    os.environ['PATH'] = abs_path_dll + os.pathsep + os.environ['PATH']
    lib_path = ""
    if os.name == "nt":
        lib_path = os.path.join(abs_path_dll, "DdInference.dll")
    else:
        lib_path = os.path.join(abs_path_dll, "DdInference.so")
    lib = CDLL(lib_path, winmode=0)
    if not os.path.exists(lib_path):
        return None
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

class DdInferenceWrapper:
    def __init__(self):
        self.lib = GetTheAPI()
        if self.lib is not None:
            self.lib.Initialize()
        self.input_image = C_Image()
        self.time_stamp = 0
        # Define the C-compatible callback function
        self._c_callback = SEND_RESULTS(self.py_accept_results)
        if self.lib is not None:
            self.api_id = self.lib.AddApiFuncPtr(self._c_callback)
        self.finished = True
        self.output = ""

    def __del__(self):
        if self.lib is not None:
            self.lib.Terminate()

    def py_accept_results(self, results):
        # Convert the results pointer to a Python object
        # py_results = C_Results.from_address(ctypes.addressof(results.contents))
        self.output = results.ClassName.decode("utf-8") 
        self.finished = True
        # Return 0 to indicate success
        return 0

    def get_callback(self):
        # Return the C-compatible callback function
        return self._c_callback

    def forward(self, img):
        if self.lib is None:
            logger.warning('Failed to load the API!')
            return ""
        if not self.finished:
            return ""
        self.finished = False
        self.input_image.TimeStamp = self.time_stamp
        self.input_image.Ptr = img.ctypes.data_as(POINTER(c_ubyte))
        self.input_image.Height, self.input_image.Width, self.input_image.Depth = img.shape
        if (self.lib.SendImage(self.api_id, self.input_image) != 1):
            return ""
        while(not self.finished):
            time.sleep(0.1)
        self.time_stamp += 1
        tmp = self.output
        self.output = ""
        return tmp
    
    def print_info(self, the_str):
        if self.lib is not None:
            self.lib.Info(str.encode(the_str))

if __name__ == "__main__":
    wrapper = DdInferenceWrapper()
    img = cv2.imread('samples/mt_defect.jpg')
    img = np.array(img)
    wrapper.print_info(f'the image is {wrapper.forward(img)}')
