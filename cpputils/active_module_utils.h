
#ifndef CPPACC_ACTIVE_MODULE_UTILS_H
#define CPPACC_ACTIVE_MODULE_UTILS_H


#ifndef __NVCC__
#define __host__
#define __device__

#endif

#ifndef __NVCC__
#include <math.h>


template <typename T> inline
T max(T a, T b){
    return std::max(a,b);
}


template <typename T> inline
T min(T a, T b){
    return std::min(a,b);
}


template <typename T> inline
T sqrt(T a){
    return std::sqrt(a);
}


template <typename T> inline
T exp(T a){
    return std::exp(a);
}

#else

#include "cuda_runtime.h"


template <typename T>
inline
__host__ __device__
T max(T a, T b){
    return fmax(a,b);
}


template <typename T>
inline
__host__ __device__
T min(T a, T b){
    return fmin(a,b);
}


template <typename T>
inline
__host__ __device__
        T sqrt(T a){
return fsqrt(a);
}


template <typename T>
inline
__host__ __device__
T exp(T a){
return fexp(a);
}


#endif


class utils {
public:
    static __host__ __device__
    float clamp(float val, float minv, float maxv) {
        return max(min(val, maxv), minv);
    }


    static __host__ __device__ int pos_remainder(int a, int b) {
        return ((a%b)+b)%b;

    }

};






#endif //CPPACC_ACTIVE_MODULE_UTILS_H
