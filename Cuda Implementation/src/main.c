#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

extern void launchCube(float * h_in, int n, float * h);

int main(){
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    float h_in[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; i++){
        h_in[i] = i;    
    }

    float h_out[ARRAY_SIZE];
    launchCube(h_in, ARRAY_SIZE, h_out);
    
    for (int i =0; i < ARRAY_SIZE; i++) {
        printf("%f", h_out[i]);
        printf(((i % 4) != 3) ? "\t" : "\n");
    }


    return 0;
}