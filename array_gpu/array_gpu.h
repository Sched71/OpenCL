#include <stdlib.h>
#include <iostream>
#include <fstream>
#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.cpp>
#else
#include <CL/cl.hpp>
#endif
struct array_gpu{
    private:
    int local_size = 32;
    int global_size = 1024;
    cl::Platform platform;
    cl::Device my_gpu;
    cl::Context context;
    cl::Program::Sources source;
    cl::Program program;
    public:
    array_gpu();
    int sum_gpu(int* A, size_t len);
    int count_gpu(int* A, size_t len, int k);
};
array_gpu::array_gpu(){
    try{
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        platform = platforms[0];

        std::vector<cl::Device> gpus;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &gpus);
        my_gpu = gpus[0];

        cl::Context cont({my_gpu});
        context = cont;

        cl::Program::Sources src;
        std::ifstream sourceFile("array_gpu.cl");
        std::string sourceCode( std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
        src.push_back(std::make_pair(sourceCode.c_str(),sourceCode.length()));
        source = src;
        
        cl::Program prog(context, src);
        program = prog;
        program.build({my_gpu});
    }
    catch(cl::Error err){
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
    }
}

int array_gpu::sum_gpu(int* A, size_t len){
    cl::CommandQueue q(context, my_gpu);

    cl::Kernel sum = cl::Kernel(program, "sum");
    int res[1] = {0};

    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * len);
    cl::Buffer buffer_res(context, CL_MEM_READ_WRITE, sizeof(int));
    q.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * len, A);
    q.enqueueWriteBuffer(buffer_res, CL_TRUE, 0, sizeof(int), res);
    
    sum.setArg(0, buffer_A);
    sum.setArg(1, buffer_res);
    sum.setArg(2, (int)len);
    q.enqueueNDRangeKernel(sum, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
    int n[1];
    q.enqueueReadBuffer(buffer_res, CL_TRUE, 0, sizeof(int), n);
    q.finish();
    return n[0];
}

int array_gpu::count_gpu(int* A, size_t len, int k){
    cl::CommandQueue q(context, my_gpu);

    cl::Kernel count = cl::Kernel(program, "count");
    int res[1] = {0};

    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * len);
    cl::Buffer buffer_res(context, CL_MEM_READ_WRITE, sizeof(int));
    q.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * len, A);
    q.enqueueWriteBuffer(buffer_res, CL_TRUE, 0, sizeof(int), res);
    
    count.setArg(0, buffer_A);
    count.setArg(1, buffer_res);
    count.setArg(2, (int)len);
    count.setArg(3, k);
    q.enqueueNDRangeKernel(count, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
    int n[1];
    q.enqueueReadBuffer(buffer_res, CL_TRUE, 0, sizeof(int), n);
    q.finish();
    return n[0];
}
