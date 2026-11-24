__global__
void matmul_fusion(
    float* m1_in, float* m1_w, float* m1_b, float* m1_out, // matrix multiplication
    float* m2_in, float* m2_w, float* m2_b, float* m2_out, // matrix multiplication
    int m1_in_h, int m1_in_w, int m1_w_w, // matrix multiplication
    int m2_in_h, int m2_in_w, int m2_w_w, // matrix multiplication
    int m1_merge, int m2_merge, // number of merged kernels
    int m1_warpPerSM, int m2_warpPerSM, // allocated warps per SM
    int m1_numOps, int m2_numOps // number of iterations
)
{
    __shared__ float m1_product[5 * 24 * 32];
    __shared__ float m2_product[5 * 24 * 32];

    int m1_warpPerTB, m2_warpPerTB; // number of warps per Thread Block 

    // calculate number of warps per thread block
    m1_warpPerTB = m1_warpPerSM / 2; // 2 blocks per SM
    m2_warpPerTB = m2_warpPerSM / 2;

	if (blockIdx.x < SM_COUNT) {
		m1_warpPerTB += m1_warpPerSM & 1;
	}
	else {
		m2_warpPerTB += m2_warpPerSM & 1;
	}

    if (threadIdx.x < m1_warpPerTB * WARP_SIZE) {
        int m1_offset = SM_COUNT * WARP_SIZE * m1_warpPerSM;
    
        int m1_in_size = m1_in_h * m1_in_w;
        int m1_out_size = m1_in_h * m1_w_w;

        int m1_out_index = blockIdx.x * (WARP_SIZE * m1_warpPerTB) + threadIdx.x;
		if (blockIdx.x >= SM_COUNT) {
			m1_out_index += (m1_warpPerSM & 1) * SM_COUNT * WARP_SIZE;
		}

        for (int loops = 0; loops < m1_numOps; loops++) {
            if (m1_out_index < m1_out_size) {
                int m1_out_i, m1_out_j;

                m1_out_i = m1_out_index / m1_w_w;
                m1_out_j = m1_out_index % m1_w_w;

                for (int m = 0; m < m1_merge; m++) {
                    m1_product[threadIdx.x * 5 + m] = 0.;
                }

                for (int k = 0; k < m1_in_w; k++) {
                    float m1_w_param = m1_w[k * m1_w_w + m1_out_j];

                    #pragma unroll (1)
                    for (int m = 0; m < m1_merge; m++) {
                        m1_product[threadIdx.x * 5 + m] += m1_in[m1_in_size * m + m1_out_i * m1_in_w + k] * m1_w_param;
                    }
                }

                for (int m = 0; m < m1_merge; m++) {
                    m1_out[m * m1_out_size + m1_out_index] = m1_product[threadIdx.x * 5 + m];
                }
            }

            m1_out_index += m1_offset;
        }
    }
	else if (threadIdx.x < (m1_warpPerTB + m2_warpPerTB) * WARP_SIZE) {
        int m2_offset = SM_COUNT * WARP_SIZE * m2_warpPerSM;

        int m2_in_size = m2_in_h * m2_in_w;
        int m2_out_size = m2_in_h * m2_w_w;
		
		int m2_out_index = blockIdx.x * (WARP_SIZE * m2_warpPerTB) + (threadIdx.x - m1_warpPerTB * WARP_SIZE);
		if (blockIdx.x >= SM_COUNT) {
			m2_out_index -= (m2_warpPerSM & 1) * SM_COUNT * WARP_SIZE;
		}

        for (int loops = 0; loops < m2_numOps; loops++) {
            if (m2_out_index < m2_out_size) {
                int m2_out_i, m2_out_j;

                m2_out_i = m2_out_index / m2_w_w;
                m2_out_j = m2_out_index % m2_w_w;

                for (int m = 0; m < m2_merge; m++) {
                    m2_product[threadIdx.x * 5 + m] = 0.;
                }

                for (int k = 0; k < m2_in_w; k++) {
                    float m2_w_param = m2_w[k * m2_w_w + m2_out_j];

                    #pragma unroll (1)
                    for (int m = 0; m < m2_merge; m++) {
                        m2_product[threadIdx.x * 5 + m] += m2_in[m2_in_size * m + m2_out_i * m2_in_w + k] * m2_w_param;
                    }
                }

                for (int m = 0; m < m2_merge; m++) {
                    m2_out[m * m2_out_size + m2_out_index] = m2_product[threadIdx.x * 5 + m];
                }
            }

            m2_out_index += m2_offset;
        }
	}
}

__global__
void matmul_QKV(float* A, float* B, float* C, int model_num, bool QK) {
    __shared__ float tile_A[12][2][TILE_SIZE];
    __shared__ float tile_B[12][TILE_SIZE][32];

    int x = blockIdx.x * blockDim.x + threadIdx.x; // blockDim.x = 32 (warp size)
    int y = blockIdx.y * blockDim.y + threadIdx.y; // blockDim.y = 2
    int z = threadIdx.z;

    int M, N, K;

    M = 197;
    if (QK) {
        K = 64;
        N = 197;
    }
    else {
        K = 197;
        N = 64;
    }

    for (int num = 0; num < model_num; num++) {
        float value = 0.f;
        for (int i = 0; i < (K - 1) / TILE_SIZE + 1; i++) {
            // load A
            if (threadIdx.x < TILE_SIZE) {
                if (i * TILE_SIZE + threadIdx.x < K && y < M) {
                    tile_A[threadIdx.z][threadIdx.y][threadIdx.x] = A[(num * M * K * 12) + (z * M*K) + (y * K) + (i*TILE_SIZE + threadIdx.x)];
                }
                else {
                    tile_A[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
                }
            }

            // load B
            for (int j = 0; j < TILE_SIZE/blockDim.y; j++) {
                if (x < N && i * TILE_SIZE + j*blockDim.y + threadIdx.y < K) {
                    tile_B[threadIdx.z][j*blockDim.y + threadIdx.y][threadIdx.x] = B[(num * N * K * 12) + (z * K*N) + ((i*TILE_SIZE + j*blockDim.y + threadIdx.y) * N) + x];
                }
                else {
                    tile_B[threadIdx.z][j*blockDim.y + threadIdx.y][threadIdx.x] = 0;
                }
            }

            __syncthreads();

            for (int k = 0; k < TILE_SIZE; k++) {
                value += tile_A[threadIdx.z][threadIdx.y][k] * tile_B[threadIdx.z][k][threadIdx.x];
            }

            __syncthreads();
        }

        if (y < M && x < N) {
            C[z * M * N + y * N + x] = value;
        }
    }
}