#define WG_SIZE 32

__kernel void sum(__global int* A, __global int* res, int n){
    int localID = get_local_id(0);
    int globalID = get_global_id(0); 
    if(globalID >= n)
        return;
    __local int larr[WG_SIZE];
    larr[localID] = A[globalID];

    for (int i = WG_SIZE; i > 1; i /= 2){
        if(localID * 2 < i){
            larr[localID] += larr[localID + i/2];
        }
    }
    if(localID == 0)
        atomic_add(res, larr[0]);
}

__kernel void count(__global int* A, __global int* res, int n, int k){
    int globalID = get_global_id(0);
    int localID = get_local_id(0);
    if (globalID >= n)
        return;
    __local int local_couner;
    __local int local_A[WG_SIZE];
    local_A[localID] = A[globalID];
    if (localID == 0) 
        local_couner = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_A[localID] == k)
        atomic_inc(&local_couner);

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (localID == 0)
        atomic_add(res, local_couner);
}
