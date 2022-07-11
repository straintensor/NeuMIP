set (CMAKE_CXX_STANDARD 11)

set(headers
  GPUhelper.h
  GPUhelper.cpp
  )


#CUDA_COMPILE_PTX(ptx_files ${headers})
find_package(CUDA REQUIRED)

include_directories(${CUDA_TOOLKIT_INCLUDE})


add_library(GPUhelper STATIC ${headers} )

target_link_libraries(GPUhelper

    ${CUDA_CUDART_LIBRARY}
)

set_property(TARGET GPUhelper PROPERTY POSITION_INDEPENDENT_CODE ON)
