# Use an unpacked official ONNX Runtime C/C++ distribution.
find_path(ONNXRuntime_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ONNXRuntime_ROOT}" ENV ONNXRuntime_ROOT PATH_SUFFIXES include)
find_library(ONNXRuntime_LIBRARY NAMES onnxruntime
  HINTS "${ONNXRuntime_ROOT}" ENV ONNXRuntime_ROOT PATH_SUFFIXES lib)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime REQUIRED_VARS ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)
if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::ONNXRuntime)
  add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
  set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}")
endif()
if(WIN32)
  find_file(ONNXRuntime_DLL onnxruntime.dll
    HINTS "${ONNXRuntime_ROOT}" ENV ONNXRuntime_ROOT PATH_SUFFIXES lib bin)
endif()
