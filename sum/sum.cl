__kernel void sum(__global int* A, __global int* res){
    int localID = get_local_id(0);
    int globalID = get_global_id(0); 
    __local int larr[64];
    larr[localID] = A[globalID];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 64; i > 1; i /= 2){
        if(localID * 2 < i){
            larr[localID] += larr[localID + i/2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if(localID == 0)
        atomic_add(res, larr[0]);
        
    
}
