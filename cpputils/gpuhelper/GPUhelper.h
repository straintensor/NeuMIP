/**
 * Copyright (c) 2017, Alexandr Kuznetsov
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 
 * * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef __GPUHELPER_H__
#define __GPUHELPER_H__

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <iostream>


template <typename T> class CM : public T {
};



class gpuException : public std::exception {

};


class gpuSizeMismatchException : public gpuException {
	virtual const char* what() const throw()
	{
		return "Size mismatch";
	}
};

class gpuFunctionException : public gpuException {
	virtual const char* what() const throw()
	{
		return "Function exited with non zero code";
	}
};

void gpuCheckExitCode_(int exit_code);

#undef gpuCheckExitCode
#define gpuCheckExitCode(x) gpuCheckExitCode_((int) (x))
//#define gpuCheckExitCode(x)

class gpuMemObj {
protected:
	size_t bytesize;
	void * gpu_obj = nullptr;

	void free_gpu_obj() {
		if (gpu_obj) {
			gpuCheckExitCode(cudaFree(gpu_obj));
			gpu_obj = nullptr;
		}
	}


	void operator=(gpuMemObj const &x) = delete;
	gpuMemObj(gpuMemObj const &) = delete;
public:

	//gpuMemObj(gpuMemObj&&) = default;
	gpuMemObj &operator=(gpuMemObj&& other)
	{
		free_gpu_obj();
		gpu_obj = std::move(other.gpu_obj);

		bytesize = std::move(other.bytesize);
		other.gpu_obj = nullptr;
	}

	gpuMemObj(gpuMemObj&& other)
	{
		free_gpu_obj();
		gpu_obj = std::move(other.gpu_obj);

		bytesize = std::move(other.bytesize);
		other.gpu_obj = nullptr;
	}

	gpuMemObj(size_t size) : bytesize(size), gpu_obj(nullptr)
	{
		if (bytesize > 0) {
			gpuCheckExitCode(cudaMalloc(&gpu_obj, bytesize));
		}
	}

	void setToZero() {
		gpuCheckExitCode(cudaMemset((char*)gpu_obj, 0, bytesize));
	}


	~gpuMemObj() {
		free_gpu_obj();
	}
};





template <typename T> class gpuObjectArr : public gpuMemObj {
private:
	size_t num_elem;
	size_t elem_size;

	void check_size(size_t num) const {
		if (num > num_elem) {
			throw gpuSizeMismatchException();
		}
	}
public:



    gpuObjectArr(std::vector<T> vec): gpuObjectArr(vec.size(), &vec[0]) {

	}

	gpuObjectArr(size_t num, const T *obj) : gpuMemObj(num * sizeof(T)), num_elem(num) {
		elem_size = sizeof(T);

		if (obj) {
			writeToGPU(obj, num);
		}
	}

	gpuObjectArr(size_t num = 0) : gpuMemObj(num * sizeof(T)), num_elem(num) {
		elem_size = sizeof(T);
	}

	void writeToGPU(const T *host_obj, size_t num) {
		check_size(num);

		std::cout << "cudaMemcpy " << num * elem_size << std::endl;
		gpuCheckExitCode(cudaMemcpy(gpu_obj, host_obj, num * elem_size, cudaMemcpyHostToDevice));
	}

	void readFromGPU(T *host_obj, size_t num, size_t offset = 0) const {
		check_size(num + offset);
		gpuCheckExitCode(cudaMemcpy(host_obj, (char*)gpu_obj + elem_size*offset, num * elem_size, cudaMemcpyDeviceToHost));
	}

	std::vector<T> readFromGPU() const {
		std::vector<T> ret(getNumElem());
		readFromGPU(ret.data(), getNumElem());
		return ret;
	}


	T * get_cuda_ptr() {
		return static_cast<T*>(gpu_obj);
	}


	const T * get_cuda_ptr() const {
		return (T*)gpu_obj;
	}


	size_t getNumElem() const {
		return num_elem;
	}

};




typedef gpuObjectArr<float> gpuObjectArrF;

std::ostream &operator<<(std::ostream &os, gpuObjectArrF const &m);


#endif /* __GPUHELPER_H__ */