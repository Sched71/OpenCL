#include <stdlib.h>
#include "array_gpu.h"
#include <chrono>
#include <string>
#include <numeric>
#include <algorithm>

int sum(int* A, int len);
int count(int* A,int len, int k);

int main(int argc, char const *argv[])
{
    /*int res = 0;
    res = a.count_gpu(A, len, k);
    std::cout << res << std::endl;
    res = count(A, len, k);
    std::cout << res << std::endl;*/
    std::string s = "";
    int len = 1024*1024*256;
    array_gpu a = array_gpu();

    while(len != 1024 * 1024 * 512){
        std::srand(std::time(nullptr));
        int* A = new int[len];
        for(int i = 0; i < len; ++i){
            A[i] = std::rand() % 5;
        }
        int k = std::rand() % 5;
        


        int x = 0;
        for (int i = 0; i < 100; ++i){
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            //a.sum_gpu(A, len);
            a.count_gpu(A, len, k);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            x += std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count();
        }
        int a1 = x / 100;
        x = 0;
        for (int i = 0; i < 100; ++i){
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            //sum(A, len);
            count(A, len, k);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            x += std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count();
        }
        int a2 = x / 100;
        x = 0;
        for (int i = 0; i < 100; ++i){
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            //std::accumulate(A, A + len, 0);
            std::count(A, A + len, k);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            x += std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count();
        }
        int a3 = x / 100;
        s.append(",(" + std::to_string(a1) + "," + std::to_string(a2) + "," + std::to_string(a3) + ")");
        delete[] A;
        len *= 2;
    }
    std:: cout << s << std::endl;
    return 0;
}

int sum(int* A, int len){
    int res = 0;
    for(int i = 0; i < len; ++i){
        res += A[i];
    }
    return res;
}

int count(int* A,int len, int k){
    int res = 0;
    for(int i = 0; i < len; ++i){
        if (A[i] == k)
            res += 1;
    }
    return res;
}
