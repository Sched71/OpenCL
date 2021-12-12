#include <stdlib.h>
#include <iostream>
#include <fstream>
#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.cpp>
#else
#include <CL/cl.hpp>
#endif


const char * kernel = "";

int main(int argc, char const *argv[])
{
    try{
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        std::vector<cl::Device> gpus;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &gpus);
        cl::Device gpu = gpus[0];

        cl::Context context(gpu);

        cl::CommandQueue q(context);

        std::ifstream sourceFile("kernels.cl");
        std::string sourceCode( std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));

        cl::Program program(context, source);
        program.build(gpus);


    }
    catch(cl::Error err){
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        return( EXIT_FAILURE );
    }

    
}
