#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <numeric>
#include <chrono>
#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.cpp>
#else
#include <CL/cl.hpp>
#endif


int main(int argc, char const *argv[])
{
    try{
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform nvidia = platforms[0];

        std::vector<cl::Device> gpus;
        nvidia.getDevices(CL_DEVICE_TYPE_GPU, &gpus);
        cl::Device my_gpu = gpus[0];
        std::cout << my_gpu.getInfo<CL_DEVICE_NAME>() << std::endl;

        cl::Context context({my_gpu});
        cl::Program::Sources src;
        std::ifstream sourceFile("sum.cl");
        std::string sourceCode( std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
        src.push_back(std::make_pair(sourceCode.c_str(),sourceCode.length()));
        
        cl::Program program(context, src);
        program.build({my_gpu});

        cl::CommandQueue q(context, my_gpu);

        cl::Kernel sum = cl::Kernel(program, "sum");
        int k = 0;
        std::cin >> k;
        int len = 1024 * 1024;
        int A[len];
        int res[1] = {0};
        for(int i = 0; i < len; ++i){
            A[i] = k;
        }
        
        cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(cl_int)*len);
        cl::Buffer buffer_res(context, CL_MEM_READ_WRITE, sizeof(int));

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        q.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(cl_int)*len, A);
        q.enqueueWriteBuffer(buffer_res, CL_TRUE, 0, sizeof(int), res);

        sum.setArg(0, buffer_A);
        sum.setArg(1, buffer_res);
        q.enqueueNDRangeKernel(sum, cl::NullRange, cl::NDRange(len), cl::NDRange(64));
        int n[1];
        q.enqueueReadBuffer(buffer_res, CL_TRUE, 0, sizeof(int), n);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << n[0] << std::endl;
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;


        int cpu_res = 0;
        begin = std::chrono::steady_clock::now();
        for(int i = 0; i < len; ++i)
            cpu_res += A[i];
        end = std::chrono::steady_clock::now();
        std::cout << cpu_res <<  std::endl;
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;

        cpu_res = 0;
        begin = std::chrono::steady_clock::now();
        cpu_res = std::accumulate(A, A + len, 0);
        end = std::chrono::steady_clock::now();
        std::cout << cpu_res << std::endl;
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
    }
    catch(cl::Error err){
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        return( EXIT_FAILURE );
    }

    
}

