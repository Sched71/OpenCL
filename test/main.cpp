#include <iostream>
#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl2.hpp>
#endif

int main() {
    // get all platforms (drivers), e.g. NVIDIA
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size()==0) {
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    std::vector<cl::Device> gpus;
    int gpu_index;
    for (int i = 0; i < all_platforms.size(); i++){
        all_platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &gpus);
        if (gpus.size() > 0){
            gpu_index = i;
            break;
        }
        else{
            std::cout << "No GPUs found";
            exit(1);
        }
    }
    cl::Platform default_platform = all_platforms[gpu_index];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    // use device[1] because that's a GPU; device[0] is the CPU
    cl::Device default_device = gpus[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";

    cl::Context context({default_device});

    cl::CommandQueue queue(context, default_device);
}