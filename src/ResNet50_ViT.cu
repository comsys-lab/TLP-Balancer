#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MERGE 5
#define SM_COUNT 8
#define WARP_SIZE 32
#define MAX_WARP_PER_SM 48
#define TILE_SIZE 16
#define INPUT_SIZE 224*224*3
#define OUTPUT_SIZE 1000
#define MERGE_MAX 5

#include "layer_fused_matmul.cu"
#include "vgg_kernels.cu"
#include "warp_allocation.cu"

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
        } \
    } while (0)

void read_parameter(const char *pFileName,float *layer_parameters)
{
	FILE *fp = fopen(pFileName, "rb");
	int count = 0;
	double temp_num;
	while(fscanf(fp, "%lf", &temp_num) == 1){
		layer_parameters[count] = temp_num;
		count++;
	}
	fclose(fp);
}

extern "C" {
void Res_ViT_host2gpu(
    float*** r_input_addr, float*** r_weight_addr, float*** r_output_addr,
    float*** v_input_addr, float*** v_weight_addr, float*** v_output_addr,
    float*** Q_addr,       float*** K_addr,        float*** V_addr,     float*** QK_addr,   float*** QKV_addr
)
{
    int r_MKN[50][3] = {
        64, 147, 12544, 
        64, 64, 3136, 
        64, 576, 3136, 
        256, 64, 3136, 
        64, 256, 3136, 
        64, 576, 3136, 
        256, 64, 3136, 
        64, 256, 3136, 
        64, 576, 3136, 
        256, 64, 3136, 
        128, 256, 784, 
        128, 1152, 784, 
        512, 128, 784, 
        128, 512, 784, 
        128, 1152, 784, 
        512, 128, 784, 
        128, 512, 784, 
        128, 1152, 784, 
        512, 128, 784, 
        128, 512, 784, 
        128, 1152, 784, 
        512, 128, 784, 
        256, 512, 196, 
        256, 2304, 196, 
        1024, 256, 196, 
        256, 1024, 196, 
        256, 2304, 196, 
        1024, 256, 196, 
        256, 1024, 196, 
        256, 2304, 196, 
        1024, 256, 196, 
        256, 1024, 196, 
        256, 2304, 196, 
        1024, 256, 196, 
        256, 1024, 196, 
        256, 2304, 196, 
        1024, 256, 196, 
        256, 1024, 196, 
        256, 2304, 196, 
        1024, 256, 196, 
        512, 1024, 49, 
        512, 4608, 49, 
        2048, 512, 49, 
        512, 2048, 49, 
        512, 4608, 49, 
        2048, 512, 49, 
        512, 2048, 49, 
        512, 4608, 49, 
        2048, 512, 49,
        1, 2048, 1000
    };

    int v_MKN[50][3] = {
        196, 768, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768
    };

    init_oracle_warp_count();

    float** r_input = (float**) malloc(50 * sizeof(float*));
    float** r_weight = (float**) malloc(50 * sizeof(float*));
    float** r_output = (float**) malloc(50 * sizeof(float*));

    float** v_input = (float**) malloc(50 * sizeof(float*));
    float** v_weight = (float**) malloc(50 * sizeof(float*));
    float** v_output = (float**) malloc(50 * sizeof(float*));
    float** Q = (float**) malloc(50 * sizeof(float*));
    float** K = (float**) malloc(50 * sizeof(float*));
    float** V = (float**) malloc(50 * sizeof(float*));
    float** QK = (float**) malloc(50 * sizeof(float*));
    float** QKV = (float**) malloc(50 * sizeof(float*));

    *r_input_addr = r_input;
    *r_weight_addr = r_weight;
    *r_output_addr = r_output;

    *v_input_addr = v_input;
    *v_weight_addr = v_weight;
    *v_output_addr = v_output;
    *Q_addr = Q;
    *K_addr = K;
    *V_addr = V;
    *QK_addr = QK;
    *QKV_addr = QKV;

    for (int i = 0; i < 50; i++) {
        /* ResNet */
        int r_input_size = r_MKN[i][0] * r_MKN[i][1] * MERGE;
        int r_weight_size = r_MKN[i][1] * r_MKN[i][2];
        int r_output_size = r_MKN[i][0] * r_MKN[i][2] * MERGE;

        // CPU malloc
        float* r_input_CPU = (float*) malloc(r_input_size * sizeof(float));
        float* r_weight_CPU = (float*) malloc(r_weight_size * sizeof(float));
        float* r_output_CPU = (float*) malloc(r_output_size * sizeof(float));

        // CPU init
        for (int j = 0; j < r_input_size; j++)
            r_input_CPU[j] = ((float)rand() / RAND_MAX) * 2 - 1;
        for (int j = 0; j < r_weight_size; j++)
            r_weight_CPU[j] = ((float)rand() / RAND_MAX) * 2 - 1;
        for (int j = 0; j < r_output_size; j++)
            r_output_CPU[j] = ((float)rand() / RAND_MAX) * 2 - 1;

        // GPU malloc
        cudaMalloc((void**) &r_input[i], r_input_size * sizeof(float));
        cudaMalloc((void**) &r_weight[i], r_weight_size * sizeof(float));
        cudaMalloc((void**) &r_output[i], r_output_size * sizeof(float));

        // GPU memcpy
        cudaMemcpy(r_input[i], r_input_CPU, r_input_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(r_weight[i], r_weight_CPU, r_weight_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(r_output[i], r_output_CPU, r_output_size * sizeof(float), cudaMemcpyHostToDevice);

        // CPU free
        free(r_input_CPU);
        free(r_weight_CPU);
        free(r_output_CPU);

        /* ViT */
        int v_input_size = v_MKN[i][0] * v_MKN[i][1] * MERGE;
        int v_weight_size = v_MKN[i][1] * v_MKN[i][2];
        int v_output_size = v_MKN[i][0] * v_MKN[i][2] * MERGE;

        // CPU malloc
        float* v_input_CPU = (float*) malloc(v_input_size * sizeof(float));
        float* v_weight_CPU = (float*) malloc(v_weight_size * sizeof(float));
        float* v_output_CPU = (float*) malloc(v_output_size * sizeof(float));
        float* Q_CPU = (float*) malloc(12*197*64 * sizeof(float) * 5);
        float* K_CPU = (float*) malloc(12*64*197 * sizeof(float) * 5);
        float* V_CPU = (float*) malloc(12*197*64 * sizeof(float) * 5);
        float* QK_CPU = (float*) malloc(12*197*197 * sizeof(float) * 5);
        float* QKV_CPU = (float*) malloc(12*197*64 * sizeof(float) * 5);
        
        // CPU init
        for (int j = 0; j < v_input_size; j++)
            v_input_CPU[j] = ((float)rand() / RAND_MAX) * 2 - 1;
        for (int j = 0; j < v_weight_size; j++)
            v_weight_CPU[j] = ((float)rand() / RAND_MAX) * 2 - 1;
        for (int j = 0; j < v_output_size; j++)
            v_output_CPU[j] = ((float)rand() / RAND_MAX) * 2 - 1;
        for (int j = 0; j < 12*197*64 * 5; j++)
            Q_CPU[j] = ((float)rand() / RAND_MAX) * 2 - 1;
        for (int j = 0; j < 12*64*197 * 5; j++)
            K_CPU[j] = ((float)rand() / RAND_MAX) * 2 - 1; 
        for (int j = 0; j < 12*197*64 * 5; j++)
            V_CPU[j] = ((float)rand() / RAND_MAX) * 2 - 1;
        for (int j = 0; j < 12*197*197 * 5; j++)
            QK_CPU[j] = ((float)rand() / RAND_MAX) * 2 - 1;
        for (int j = 0; j < 12*197*64 * 5; j++)
            QKV_CPU[j] = ((float)rand() / RAND_MAX) * 2 - 1;       

        // GPU malloc
        cudaMalloc((void**) &v_input[i], v_input_size * sizeof(float));
        cudaMalloc((void**) &v_weight[i], v_weight_size * sizeof(float));
        cudaMalloc((void**) &v_output[i], v_output_size * sizeof(float));
        cudaMalloc((void**) &Q[i], 12*197*64*5 * sizeof(float));
        cudaMalloc((void**) &K[i], 12*64*197*5 * sizeof(float));
        cudaMalloc((void**) &V[i], 12*197*64*5 * sizeof(float));
        cudaMalloc((void**) &QK[i], 12*197*197*5 * sizeof(float));
        cudaMalloc((void**) &QKV[i], 12*197*64*5 * sizeof(float));

        // GPU memcpy
        cudaMemcpy(v_input[i], v_input_CPU, v_input_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(v_weight[i], v_weight_CPU, v_weight_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(v_output[i], v_output_CPU, v_output_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Q[i], Q_CPU, 12*197*64*5 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(K[i], K_CPU, 12*64*197*5 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(V[i], V_CPU, 12*197*64*5 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(QK[i], QK_CPU, 12*197*197*5 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(QKV[i], QKV_CPU, 12*197*64*5 * sizeof(float), cudaMemcpyHostToDevice);

        // CPU free
        free(v_input_CPU);
        free(v_weight_CPU);
        free(v_output_CPU);
        free(Q_CPU);
        free(K_CPU);
        free(V_CPU);
        free(QK_CPU);
        free(QKV_CPU);
    }
}

void Res_ViT_inference(
    float** r_input, float** r_weight, float** r_output,
    float** v_input, float** v_weight, float** v_output,
    float** Q, float** K, float** V, float** QK, float** QKV,
    int r_model_num, int v_model_num, int mode)
{
    int r_MKN[50][3] = {
        64, 147, 12544, 
        64, 64, 3136, 
        64, 576, 3136, 
        256, 64, 3136, 
        64, 256, 3136, 
        64, 576, 3136, 
        256, 64, 3136, 
        64, 256, 3136, 
        64, 576, 3136, 
        256, 64, 3136, 
        128, 256, 784, 
        128, 1152, 784, 
        512, 128, 784, 
        128, 512, 784, 
        128, 1152, 784, 
        512, 128, 784, 
        128, 512, 784, 
        128, 1152, 784, 
        512, 128, 784, 
        128, 512, 784, 
        128, 1152, 784, 
        512, 128, 784, 
        256, 512, 196, 
        256, 2304, 196, 
        1024, 256, 196, 
        256, 1024, 196, 
        256, 2304, 196, 
        1024, 256, 196, 
        256, 1024, 196, 
        256, 2304, 196, 
        1024, 256, 196, 
        256, 1024, 196, 
        256, 2304, 196, 
        1024, 256, 196, 
        256, 1024, 196, 
        256, 2304, 196, 
        1024, 256, 196, 
        256, 1024, 196, 
        256, 2304, 196, 
        1024, 256, 196, 
        512, 1024, 49, 
        512, 4608, 49, 
        2048, 512, 49, 
        512, 2048, 49, 
        512, 4608, 49, 
        2048, 512, 49, 
        512, 2048, 49, 
        512, 4608, 49, 
        2048, 512, 49,
        1, 2048, 1000
    };

    int v_MKN[50][3] = {
        196, 768, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768, 
        197, 768, 2304, 
        197, 768, 768, 
        197, 768, 3072, 
        197, 3072, 768
    };

    FILE* fp = fopen("warp_allocation_results.txt", "a");

    float execution_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int r_warpPerSM = 24, v_warpPerSM = 24;

    // Kernel
    dim3 dimGrid_QK(197/32 + 1, 197/2 + 1, 1);
    dim3 dimBlock_QK(32, 2, 12);

    dim3 dimGrid_QKV(64/32, 197/2 + 1, 1);
    dim3 dimBlock_QKV(32, 2, 12);

    for (int layer = 0; layer < 50; layer++) {
        int r_numOps = 0, v_numOps = 0;

        r_warpPerSM = 24;
        v_warpPerSM = MAX_WARP_PER_SM - r_warpPerSM;

        // allocate warp
        warp_allocation(r_MKN[layer], r_model_num,
                        v_MKN[layer], v_model_num,
                        48, layer+1, &r_warpPerSM, &v_warpPerSM,
                        mode);

        r_numOps = ceil((float)(r_MKN[layer][0]*r_MKN[layer][2]) / (r_warpPerSM * SM_COUNT * WARP_SIZE));
        v_numOps = ceil((float)(v_MKN[layer][0]*v_MKN[layer][2]) / (v_warpPerSM * SM_COUNT * WARP_SIZE));

        cudaEventRecord(start, 0);

        matmul_fusion <<<SM_COUNT*2, WARP_SIZE * MAX_WARP_PER_SM / 2>>>(
            r_input[layer], r_weight[layer], NULL, r_output[layer],
            v_input[layer], v_weight[layer], NULL, v_output[layer],
            r_MKN[layer][0], r_MKN[layer][1], r_MKN[layer][2],
            v_MKN[layer][0], v_MKN[layer][1], v_MKN[layer][2],
            r_model_num, v_model_num,
            r_warpPerSM, v_warpPerSM,
            r_numOps, v_numOps
        );

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&execution_time, start, stop);

        fprintf(fp, "%d:\t%d\t%d\t%d\t%d\t%f\n", layer+1, r_model_num, v_model_num, r_warpPerSM, v_warpPerSM, execution_time);

        matmul_QKV <<< dimGrid_QK, dimBlock_QK >>> (Q[layer], K[layer], QK[layer], v_model_num, true);
        matmul_QKV <<< dimGrid_QKV, dimBlock_QKV >>> (QK[layer], V[layer], QKV[layer], v_model_num, false);
    }
    cudaDeviceSynchronize();

    fclose(fp);
}

void Vgg_host2gpu(
	float** Vgg_Layer1_Neurons,		float** Vgg_Layer2_Neurons,		float** Vgg_Layer3_Neurons,		float** Vgg_Layer4_Neurons,
	float** Vgg_Layer5_Neurons,		float** Vgg_Layer6_Neurons,		float** Vgg_Layer7_Neurons,		float** Vgg_Layer8_Neurons,
	float** Vgg_Layer9_Neurons,		float** Vgg_Layer10_Neurons,	float** Vgg_Layer11_Neurons,	float** Vgg_Layer12_Neurons,
	float** Vgg_Layer13_Neurons,	float** Vgg_Layer14_Neurons,	float** Vgg_Layer15_Neurons,	float** Vgg_Layer16_Neurons,
	
	float** Vgg_Layer1_bias,		float** Vgg_Layer2_bias,		float** Vgg_Layer3_bias,		float** Vgg_Layer4_bias,
	float** Vgg_Layer5_bias,		float** Vgg_Layer6_bias,		float** Vgg_Layer7_bias,		float** Vgg_Layer8_bias,
	float** Vgg_Layer9_bias,		float** Vgg_Layer10_bias,		float** Vgg_Layer11_bias,		float** Vgg_Layer12_bias,
	float** Vgg_Layer13_bias,		float** Vgg_Layer14_bias,		float** Vgg_Layer15_bias,		float** Vgg_Layer16_bias,
	
	float** Vgg_Layer1_Weights,		float** Vgg_Layer2_Weights,		float** Vgg_Layer3_Weights,		float** Vgg_Layer4_Weights,
	float** Vgg_Layer5_Weights,		float** Vgg_Layer6_Weights,		float** Vgg_Layer7_Weights,		float** Vgg_Layer8_Weights,
	float** Vgg_Layer9_Weights,		float** Vgg_Layer10_Weights,	float** Vgg_Layer11_Weights,	float** Vgg_Layer12_Weights,
	float** Vgg_Layer13_Weights,	float** Vgg_Layer14_Weights,	float** Vgg_Layer15_Weights,	float** Vgg_Layer16_Weights,
	
	float** Vgg_Layer2_pool,		float** Vgg_Layer4_pool,		float** Vgg_Layer7_pool,		float** Vgg_Layer10_pool,
	float** Vgg_Layer13_pool,		float** Vgg_Result_Neurons)
{
	///////////////////////
	/*** Vgg Parameter ***/
	///////////////////////
	int bias[16] = { 
		64 * sizeof(float),	  // 1
		64 * sizeof(float),   // 2
		128 * sizeof(float),  // 3
		128 * sizeof(float),  // 4
		256 * sizeof(float),  // 5
		256 * sizeof(float),  // 6
		256 * sizeof(float),  // 7
		512 * sizeof(float),  // 8
		512 * sizeof(float),  // 9
		512 * sizeof(float),  // 10
		512 * sizeof(float),  // 11
		512 * sizeof(float),  // 12
		512 * sizeof(float),  // 13
		4096 * sizeof(float), // 14
		4096 * sizeof(float), // 15
		1000 * sizeof(float)  // 16
	}; 

	int weight[16] = {
		  64 *    3 *  3 *   3 * sizeof(float), // 1
		  64 *    3 *  3 *  64 * sizeof(float), // 2
		 128 *    3 *  3 *  64 * sizeof(float), // 3
		 128 *    3 *  3 * 128 * sizeof(float), // 4
		 256 *    3 *  3 * 128 * sizeof(float), // 5
		 256 *    3 *  3 * 256 * sizeof(float), // 6
		 256 *    3 *  3 * 256 * sizeof(float),	// 7
		 512 *    3 *  3 * 256 * sizeof(float),	// 8
		 512 *    3 *  3 * 512 * sizeof(float), // 9
		 512 *    3 *  3 * 512 * sizeof(float), // 10
		 512 *    3 *  3 * 512 * sizeof(float), // 11
		 512 *    3 *  3 * 512 * sizeof(float), // 12
		 512 *    3 *  3 * 512 * sizeof(float), // 13
		4096 *  512 *  7 *   7 * sizeof(float), // 14
		4096 * 4096 * sizeof(float),			// 15
		1000 * 4096 * sizeof(float)				// 16
	};

	// 1. CPU malloc
	float* Vgg_Layer1_Neurons_CPU;
	float* Vgg_Layer_bias_CPU[16];
	float* Vgg_Layer_Weights_CPU[16];

	Vgg_Layer1_Neurons_CPU = (float*) malloc(INPUT_SIZE * sizeof(float) * MERGE_MAX);
	for (int i = 0; i < 16; i++) {
		Vgg_Layer_bias_CPU[i] = (float*) malloc(bias[i]);
		Vgg_Layer_Weights_CPU[i] = (float*) malloc(weight[i]);
	}

	// 2. CPU initialize
	read_parameter("data_vgg16/input_cat.txt", Vgg_Layer1_Neurons_CPU);

	read_parameter("data_vgg16/bias1.txt", Vgg_Layer_bias_CPU[0]);
	read_parameter("data_vgg16/bias2.txt", Vgg_Layer_bias_CPU[1]);
	read_parameter("data_vgg16/bias3.txt", Vgg_Layer_bias_CPU[2]);
	read_parameter("data_vgg16/bias4.txt", Vgg_Layer_bias_CPU[3]);
	read_parameter("data_vgg16/bias5.txt", Vgg_Layer_bias_CPU[4]);
	read_parameter("data_vgg16/bias6.txt", Vgg_Layer_bias_CPU[5]);
	read_parameter("data_vgg16/bias7.txt", Vgg_Layer_bias_CPU[6]);
	read_parameter("data_vgg16/bias8.txt", Vgg_Layer_bias_CPU[7]);
    read_parameter("data_vgg16/bias9.txt", Vgg_Layer_bias_CPU[8]);
	read_parameter("data_vgg16/bias10.txt", Vgg_Layer_bias_CPU[9]);
	read_parameter("data_vgg16/bias11.txt", Vgg_Layer_bias_CPU[10]);
	read_parameter("data_vgg16/bias12.txt", Vgg_Layer_bias_CPU[11]);
	read_parameter("data_vgg16/bias13.txt", Vgg_Layer_bias_CPU[12]);
	read_parameter("data_vgg16/bias14.txt", Vgg_Layer_bias_CPU[13]);
	read_parameter("data_vgg16/bias15.txt", Vgg_Layer_bias_CPU[14]);
	read_parameter("data_vgg16/bias16.txt", Vgg_Layer_bias_CPU[15]);

	read_parameter("data_vgg16/conv1.txt", Vgg_Layer_Weights_CPU[0]);
	read_parameter("data_vgg16/conv2.txt", Vgg_Layer_Weights_CPU[1]);
	read_parameter("data_vgg16/conv3.txt", Vgg_Layer_Weights_CPU[2]);
	read_parameter("data_vgg16/conv4.txt", Vgg_Layer_Weights_CPU[3]);
	read_parameter("data_vgg16/conv5.txt", Vgg_Layer_Weights_CPU[4]);
	read_parameter("data_vgg16/conv6.txt", Vgg_Layer_Weights_CPU[5]);
	read_parameter("data_vgg16/conv7.txt", Vgg_Layer_Weights_CPU[6]);
	read_parameter("data_vgg16/conv8.txt", Vgg_Layer_Weights_CPU[7]);
 	read_parameter("data_vgg16/conv9.txt", Vgg_Layer_Weights_CPU[8]);
	read_parameter("data_vgg16/conv10.txt", Vgg_Layer_Weights_CPU[9]);
	read_parameter("data_vgg16/conv11.txt", Vgg_Layer_Weights_CPU[10]);
	read_parameter("data_vgg16/conv12.txt", Vgg_Layer_Weights_CPU[11]);
	read_parameter("data_vgg16/conv13.txt", Vgg_Layer_Weights_CPU[12]);
	read_parameter("data_vgg16/fc14.txt", Vgg_Layer_Weights_CPU[13]);
	read_parameter("data_vgg16/fc15.txt", Vgg_Layer_Weights_CPU[14]);
	read_parameter("data_vgg16/fc16.txt", Vgg_Layer_Weights_CPU[15]);

	// 3. GPU cudaMalloc
	float** Vgg_Layer_bias[16] = {
		Vgg_Layer1_bias,  Vgg_Layer2_bias,  Vgg_Layer3_bias,  Vgg_Layer4_bias, 
		Vgg_Layer5_bias,  Vgg_Layer6_bias,  Vgg_Layer7_bias,  Vgg_Layer8_bias,
		Vgg_Layer9_bias,  Vgg_Layer10_bias, Vgg_Layer11_bias, Vgg_Layer12_bias,
		Vgg_Layer13_bias, Vgg_Layer14_bias, Vgg_Layer15_bias, Vgg_Layer16_bias
	};

	float** Vgg_Layer_Weights[16] = {
		Vgg_Layer1_Weights,  Vgg_Layer2_Weights,  Vgg_Layer3_Weights,  Vgg_Layer4_Weights, 
		Vgg_Layer5_Weights,  Vgg_Layer6_Weights,  Vgg_Layer7_Weights,  Vgg_Layer8_Weights, 
		Vgg_Layer9_Weights,  Vgg_Layer10_Weights, Vgg_Layer11_Weights, Vgg_Layer12_Weights, 
		Vgg_Layer13_Weights, Vgg_Layer14_Weights, Vgg_Layer15_Weights, Vgg_Layer16_Weights
	};

	cudaMalloc((void**) Vgg_Layer1_Neurons, INPUT_SIZE * sizeof(float) * MERGE_MAX);
	for (int i = 0; i < 16; i++) {
		cudaMalloc((void**) Vgg_Layer_bias[i], bias[i]);
		cudaMalloc((void**) Vgg_Layer_Weights[i], weight[i]);
	}

	// 4. GPU cudaMemcpy
	for (int i = 0; i < MERGE_MAX; i++) {
		cudaMemcpy(*Vgg_Layer1_Neurons + i * INPUT_SIZE, Vgg_Layer1_Neurons_CPU, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	}
	for (int i = 0; i < 16; i++) {
		cudaMemcpy(*Vgg_Layer_bias[i], Vgg_Layer_bias_CPU[i], bias[i], cudaMemcpyHostToDevice);
		cudaMemcpy(*Vgg_Layer_Weights[i], Vgg_Layer_Weights_CPU[i], weight[i], cudaMemcpyHostToDevice);
	}

	// 5. CPU free
	free(Vgg_Layer1_Neurons_CPU);
	for (int i = 0; i < 16; i++) {
		free(Vgg_Layer_bias_CPU[i]);
		free(Vgg_Layer_Weights_CPU[i]);
	}

	// 추가 cudaMalloc
	cudaMalloc((void**) Vgg_Layer2_Neurons, (64*224*224) * sizeof(float) * MERGE_MAX); //64*224*224
    cudaMalloc((void**) Vgg_Layer2_pool, (64*224*224) * sizeof(float) * MERGE_MAX); //64*224*224
    cudaMalloc((void**) Vgg_Layer3_Neurons, (64*112*112) * sizeof(float) * MERGE_MAX); //64*112*112
    cudaMalloc((void**) Vgg_Layer4_Neurons, (128*112*112) * sizeof(float) * MERGE_MAX); //128*112*112
    cudaMalloc((void**) Vgg_Layer4_pool, (128*112*112) * sizeof(float) * MERGE_MAX); //128*112*112
	cudaMalloc((void**) Vgg_Layer5_Neurons, (128*56*56) * sizeof(float) * MERGE_MAX); //128*56*56
   	cudaMalloc((void**) Vgg_Layer6_Neurons, (256*56*56) * sizeof(float) * MERGE_MAX); //256*56*56
    cudaMalloc((void**) Vgg_Layer7_Neurons, (256*56*56) * sizeof(float) * MERGE_MAX); //256*56*56
    cudaMalloc((void**) Vgg_Layer7_pool, (256*56*56) * sizeof(float) * MERGE_MAX); //256*56*56
    cudaMalloc((void**) Vgg_Layer8_Neurons, (256*28*28) * sizeof(float) * MERGE_MAX); //256*28*28
    cudaMalloc((void**) Vgg_Layer9_Neurons, (512*28*28) * sizeof(float) * MERGE_MAX); //512*28*28
    cudaMalloc((void**) Vgg_Layer10_Neurons, (512*28*28) * sizeof(float) * MERGE_MAX); //512*28*28
    cudaMalloc((void**) Vgg_Layer10_pool, (512*28*28) * sizeof(float) * MERGE_MAX); //512*28*28
    cudaMalloc((void**) Vgg_Layer11_Neurons, (512*14*14) * sizeof(float) * MERGE_MAX); //512*14*14
    cudaMalloc((void**) Vgg_Layer12_Neurons, (512*14*14) * sizeof(float) * MERGE_MAX); //512*14*14 
    cudaMalloc((void**) Vgg_Layer13_Neurons, (512*14*14) * sizeof(float) * MERGE_MAX); //512*14*14
    cudaMalloc((void**) Vgg_Layer13_pool, (512*14*14) * sizeof(float) * MERGE_MAX); //512*14*14
    cudaMalloc((void**) Vgg_Layer14_Neurons, (512*7*7) * sizeof(float) * MERGE_MAX); //512*7*7
	cudaMalloc((void**) Vgg_Layer15_Neurons, 4096 * sizeof(float) * MERGE_MAX); //4096
	cudaMalloc((void**) Vgg_Layer16_Neurons, 4096 * sizeof(float) * MERGE_MAX); //4096
	cudaMalloc((void**) Vgg_Result_Neurons, 1000 * sizeof(float) * MERGE_MAX); //1000
}

void Vgg_inference(
    float *Vgg_Layer1_Neurons,  float *Vgg_Layer2_Neurons,  float *Vgg_Layer3_Neurons,  float *Vgg_Layer4_Neurons,
    float *Vgg_Layer5_Neurons,  float *Vgg_Layer6_Neurons,  float *Vgg_Layer7_Neurons,  float *Vgg_Layer8_Neurons,
    float *Vgg_Layer9_Neurons,  float *Vgg_Layer10_Neurons, float *Vgg_Layer11_Neurons, float *Vgg_Layer12_Neurons,
    float *Vgg_Layer13_Neurons, float *Vgg_Layer14_Neurons, float *Vgg_Layer15_Neurons, float *Vgg_Layer16_Neurons,
    float *Vgg_Layer1_bias,     float *Vgg_Layer2_bias,     float *Vgg_Layer3_bias,     float *Vgg_Layer4_bias,
    float *Vgg_Layer5_bias,     float *Vgg_Layer6_bias,     float *Vgg_Layer7_bias,     float *Vgg_Layer8_bias,
    float *Vgg_Layer9_bias,     float *Vgg_Layer10_bias,    float *Vgg_Layer11_bias,    float *Vgg_Layer12_bias,
    float *Vgg_Layer13_bias,    float *Vgg_Layer14_bias,    float *Vgg_Layer15_bias,    float *Vgg_Layer16_bias,
    float *Vgg_Layer1_Weights,  float *Vgg_Layer2_Weights,  float *Vgg_Layer3_Weights,  float *Vgg_Layer4_Weights,
    float *Vgg_Layer5_Weights,  float *Vgg_Layer6_Weights,  float *Vgg_Layer7_Weights,  float *Vgg_Layer8_Weights,
    float *Vgg_Layer9_Weights,  float *Vgg_Layer10_Weights, float *Vgg_Layer11_Weights, float *Vgg_Layer12_Weights,
    float *Vgg_Layer13_Weights, float *Vgg_Layer14_Weights, float *Vgg_Layer15_Weights, float *Vgg_Layer16_Weights,
    float *Vgg_Layer2_pool,     float *Vgg_Layer4_pool,     float *Vgg_Layer7_pool,     float *Vgg_Layer10_pool,
    float *Vgg_Layer13_pool,    float *Vgg_Result_Neurons,
    int vgg_num)
{
	/* 1st Layer */
    dim3 Block_Thread(7,7);
    dim3 Block1_Block(64,32,32);
	vgg_first_conv<<<Block1_Block,Block_Thread>>>(Vgg_Layer1_bias,Vgg_Layer1_Neurons,Vgg_Layer1_Weights,Vgg_Layer2_Neurons,vgg_num,224,224,1,1,3,3,true,true);

    /* 2nd Layer */
	vgg_conv<<<Block1_Block,Block_Thread>>>(Vgg_Layer2_bias,Vgg_Layer2_Neurons,Vgg_Layer2_Weights,Vgg_Layer2_pool,
                                            vgg_num,
                                            224,224,1,1,
                                            3,64,
                                            true,true);

	dim3 Block1_Pool_Block(64,16,16);
    vgg_maxpool<<<Block1_Pool_Block,Block_Thread>>>(Vgg_Layer2_pool,Vgg_Layer3_Neurons,vgg_num,224,112,2,0,2);

    /* 3rd Layer */
	dim3 Block2_Block(128,16,16);
	vgg_conv<<<Block2_Block,Block_Thread>>>(Vgg_Layer3_bias,Vgg_Layer3_Neurons,Vgg_Layer3_Weights,Vgg_Layer4_Neurons,vgg_num,112,112,1,1,3,64,true,true);

	/* 4th Layer */    
	vgg_conv<<<Block2_Block,Block_Thread>>>(Vgg_Layer4_bias,Vgg_Layer4_Neurons,Vgg_Layer4_Weights,Vgg_Layer4_pool,vgg_num,112,112,1,1,3,128,true,true);

	dim3 Block2_Pool_Block(128,8,8);
    vgg_maxpool<<<Block2_Pool_Block,Block_Thread>>>(Vgg_Layer4_pool,Vgg_Layer5_Neurons,vgg_num,112,56,2,0,2);

    /* 5th Layer */
	dim3 Block3_Block(256,8,8);
	vgg_conv<<<Block3_Block,Block_Thread>>>(Vgg_Layer5_bias,Vgg_Layer5_Neurons,Vgg_Layer5_Weights,Vgg_Layer6_Neurons,vgg_num,56,56,1,1,3,128,true,true);

    /* 6th Layer */
    vgg_conv<<<Block3_Block,Block_Thread>>>(Vgg_Layer6_bias,Vgg_Layer6_Neurons,Vgg_Layer6_Weights,Vgg_Layer7_Neurons,vgg_num,56,56,1,1,3,256,true,true);

    /* 7th Layer */
	vgg_conv<<<Block3_Block,Block_Thread>>>(Vgg_Layer7_bias,Vgg_Layer7_Neurons,Vgg_Layer7_Weights,Vgg_Layer7_pool,vgg_num,56,56,1,1,3,256,true,true);

	dim3 Block3_Pool_Block(256,4,4);
    vgg_maxpool<<<Block3_Pool_Block,Block_Thread>>>(Vgg_Layer7_pool,Vgg_Layer8_Neurons,vgg_num,56,28,2,0,2);

    /* 8th Layer */
	dim3 Block4_Block(512,4,4);
	vgg_conv<<<Block4_Block,Block_Thread>>>(Vgg_Layer8_bias,Vgg_Layer8_Neurons,Vgg_Layer8_Weights,Vgg_Layer9_Neurons,vgg_num,28,28,1,1,3,256,true,true);

    /* 9th Layer */
    vgg_conv<<<Block4_Block,Block_Thread>>>(Vgg_Layer9_bias,Vgg_Layer9_Neurons,Vgg_Layer9_Weights,Vgg_Layer10_Neurons,vgg_num,28,28,1,1,3,512,true,true);

    /* 10th Layer */
    vgg_conv<<<Block4_Block,Block_Thread>>>(Vgg_Layer10_bias,Vgg_Layer10_Neurons,Vgg_Layer10_Weights,Vgg_Layer10_pool,vgg_num,28,28,1,1,3,512,true,true);

	dim3 Block4_Pool_Block(512,2,2);
    vgg_maxpool<<<Block4_Pool_Block,Block_Thread>>>(Vgg_Layer10_pool,Vgg_Layer11_Neurons,vgg_num,28,14,2,0,2);

    /* 11th Layer */
	dim3 Block5_Block(512,2,2);
	vgg_conv<<<Block5_Block,Block_Thread>>>(Vgg_Layer11_bias,Vgg_Layer11_Neurons,Vgg_Layer11_Weights,Vgg_Layer12_Neurons,vgg_num,14,14,1,1,3,512,true,true);

    /* 12th Layer */
    vgg_conv<<<Block5_Block,Block_Thread>>>(Vgg_Layer12_bias,Vgg_Layer12_Neurons,Vgg_Layer12_Weights,Vgg_Layer13_Neurons,vgg_num,14,14,1,1,3,512,true,true);

    /* 13th Layer */
    vgg_conv<<<Block5_Block,Block_Thread>>>(Vgg_Layer13_bias,Vgg_Layer13_Neurons,Vgg_Layer13_Weights,Vgg_Layer13_pool,vgg_num,14,14,1,1,3,512,true,true);

	dim3 Block5_Pool_Block(512,1,1);
    vgg_maxpool<<<Block5_Pool_Block,Block_Thread>>>(Vgg_Layer13_pool,Vgg_Layer14_Neurons,vgg_num,14,7,2,0,2);

    /* 14th Layer */
	dim3 FC1_Block(4096,1,1);
	dim3 FC1_Thread(1,1);
	vgg_fc<<<FC1_Block,FC1_Thread>>>(Vgg_Layer14_bias,Vgg_Layer14_Neurons,Vgg_Layer14_Weights,Vgg_Layer15_Neurons,vgg_num,(7*7*512),true);

    /* 15th Layer */
	dim3 FC2_Block(4096,1,1);
	dim3 FC2_Thread(1,1);
	vgg_fc<<<FC2_Block,FC2_Thread>>>(Vgg_Layer15_bias,Vgg_Layer15_Neurons,Vgg_Layer15_Weights,Vgg_Layer16_Neurons,vgg_num,4096,true);

    /* 16th Layer */
	dim3 FC3_Block(1000,1,1);
	dim3 FC3_Thread(1,1);
	vgg_fc<<<FC3_Block,FC3_Thread>>>(Vgg_Layer16_bias,Vgg_Layer16_Neurons,Vgg_Layer16_Weights,Vgg_Result_Neurons,vgg_num,4096,false);

    cudaDeviceSynchronize();

}

} // extern "C"
