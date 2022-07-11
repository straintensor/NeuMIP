
cmake_minimum_required(VERSION 3.0)


project(cppacc LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE TRUE)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
enable_language(CUDA)

# torch.__version__
# torch '1.7.0+cu101'
#>>> torch.__file__
#/usr/local/lib/python3.7/dist-packages/torch/__init__.py

# Torch_DIR=/usr/local/lib/python3.7/dist-packages/torch/share/cmake/Torch
#  /usr/lib/x86_64-linux-gnu/libcudnn.so.8
# /usr/include/x86_64-linux-gnu/cudnn_v8.h
#  /usr/include/
# torch.backends.cudnn.version()
#Out[3]: 7603
#

add_subdirectory(gpuhelper)

set(PY_VERSION 3)
find_package(PythonLibs ${PY_VERSION} REQUIRED)
find_package(Boost REQUIRED COMPONENTS system python3 numpy3)





set(DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../deps/")

function(set_common NAME)

    set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 17)
    target_link_libraries(${NAME}   ${PYTHON_LIBRARY}  )
    target_link_libraries(${NAME}    ${Boost_LIBRARIES}  )
    target_link_libraries(${NAME}   Boost::numpy3)
    target_include_directories(${NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gpuhelper)
    target_include_directories(${NAME}  PRIVATE  ${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
    target_include_directories(${NAME} PRIVATE ${DEPS_DIR}/cereal-1.3.0/include/)
    #set_target_properties(${NAME}  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    #set_target_properties(${NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(${NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(${NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --generate-line-info
            --use_fast_math
            --relocatable-device-code=true
            -O3
            -arch=sm_75
            >)

endfunction()



CUDA_ADD_LIBRARY(active_model STATIC  active_model.cpp active_module_utils.cu NeuralMaterial.h)
set_common(active_model)




#ADD_LIBRARY(active_model_cpu STATIC  active_model.cpp active_module_utils.cu )
#set_common(active_model_cpu)

add_library(storage_model SHARED storage_model.cpp)
set_common(storage_model)
set_target_properties(storage_model PROPERTIES PREFIX "")

#target_link_libraries(storage_model active_model_cpu)




CUDA_ADD_LIBRARY(cppacc SHARED  main.cu TorchActiveModel.cu  )
set_common(cppacc)
set_target_properties(cppacc PROPERTIES PREFIX "")




target_include_directories(cppacc PUBLIC "${TORCH_INCLUDE_DIRS}")
target_link_libraries(cppacc   "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY})
target_link_libraries(cppacc GPUhelper)
target_link_libraries(cppacc active_model)
target_link_libraries(cppacc storage_model)







