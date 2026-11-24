#include <stdio.h>
#include <math.h>
#include <time.h>


/* Convolution */
__global__ void vgg_first_conv(
	float *bias,	float *in_nu,	float *in_w,	float *out_nu,
	int model_num,
	int in_fm,		int out_fm,		int str,		int pad,
	int ker,		int ker_channel,
	bool b,			bool relu)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out)
					 + (out_fm*row)
					 + col;

	int input_size = in_fm * in_fm * ker_channel;
	int output_size = out_fm * out_fm * gridDim.x;

	//Stride
    int x_str = 0, y_str = 0;
    x_str = 3*(row*str-pad)*in_fm;
    x_str = x_str < 0 ? 0 : x_str;
    y_str = 3*(col*str-pad);
    y_str = y_str < 0 ? 0 : y_str;

	//Padding
	int x_pad = 0, y_pad = 0;
	int loopr = ker, loopc = ker;

	//Upper
	if(row*str < pad){
		x_pad = pad - row*str;
		loopr = ker - x_pad;
	}
	//Bottom
	if(row >= out_fm - pad){
		loopr = in_fm - x_str/(3*in_fm);
	}
	//Left
	if(col*str < pad){
		y_pad = pad - col*str;
		loopc = ker - y_pad;
	}
	//Right
	if(col >= out_fm - pad){
		loopc = in_fm -  y_str/3;
	}

	float product[10] = {0.0};
	for(int l = 0; l < model_num; l++){
		for(int i = 0; i < loopr; i++){
			for(int j = 0; j < loopc; j++){
				for(int k = 0; k < ker_channel; k++){
					product[l] += in_nu[input_size*l + i*in_fm*ker_channel + j*ker_channel + k + x_str + y_str] 
							*in_w[num_out*ker*ker*ker_channel + i*ker + j + k*ker*ker + x_pad*ker + y_pad];
				}
			}
		}
		if(loopc > 0 && loopr > 0){
			if(b == true)
				product[l] += bias[num_out];

			//ReLU
			if(relu == true){
				if(product[l] < 0)
					product[l] = 0;
			}
			out_nu[output_size*l + out_position] = product[l];
		}
	}
}

__global__ void vgg_conv(
	float *bias,	float *in_nu,	float *in_w,	float *out_nu,	
	int model_num,
	int in_fm,		int out_fm,		int str,		int pad,		
	int ker,		int ker_channel,
	bool b,			bool relu)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

	int input_size = in_fm * in_fm * ker_channel;
	int output_size = out_fm * out_fm * gridDim.x;

    //Stride
    int x_str = 0, y_str = 0;
    x_str = (row*str-pad)*in_fm;
    x_str = x_str < 0 ? 0 : x_str;
    y_str = col*str-pad;
    y_str = y_str < 0 ? 0 : y_str;

	//Padding
	int x_pad = 0, y_pad = 0;
	int loopr = ker, loopc = ker;

	//Upper
	if(row*str < pad){
		x_pad = pad - row*str;
		loopr = ker - x_pad;
	}
	//Bottom
	if(row >= out_fm - pad){
		loopr = in_fm - x_str/in_fm;
	}
	//Left
	if(col*str < pad){
		y_pad = pad - col*str;
		loopc = ker - y_pad;
	}
	//Right
	if(col >= out_fm - pad){
		loopc = in_fm -  y_str;
	}

	float product[5] = {0.0};
	for(int l = 0; l < model_num; l++){
		for(int i = 0; i < ker_channel; i++){
			for(int j = 0; j < loopr; j++){
				for(int k = 0; k < loopc; k++){
					product[l] += in_nu[input_size*l + in_fm*in_fm*i + in_fm*j + k + x_str + y_str] 
							*in_w[num_out*ker_channel*ker*ker + i*ker*ker + j*ker + k + x_pad*ker + y_pad];
				}
			}
		}
		if(loopc > 0 && loopr > 0){
			if(b == true)
				product[l] += bias[num_out];

			//ReLU
			if(relu == true){
				product[l] = max(product[l], 0.);
			}
			out_nu[output_size*l + out_position] = product[l];
		}
	}
}

/* Maxpooling */
__global__ void vgg_maxpool(
	float *in_nu,	float *out_nu,
	int model_num,
	int in_fm,		int out_fm,		int str,		int pad,		int ker)
{
    int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

	int input_size = in_fm * in_fm * gridDim.x;
	int output_size = out_fm * out_fm * gridDim.x;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

    //Stride
    int x_str = 0, y_str = 0;
    x_str = (row*str-pad)*in_fm;
    x_str = x_str < 0 ? 0 : x_str;
    y_str = col*str-pad;
    y_str = y_str < 0 ? 0 : y_str;

	//Padding
	int loopr = ker, loopc = ker;

	//Upper
	if(row < pad){
		loopr = ker - pad;
	}
	//Bottom
	if(row >= out_fm - pad){
		loopr = in_fm - x_str/in_fm;
	}
	//Left
	if(col < pad){
		loopc = ker - pad;
	}
	//Right
	if(col >= out_fm - pad){
		loopc = in_fm -  y_str;
	}

    float max_value[5] = {0.0};
	for(int i = 0; i < model_num; i++){
		for(int j = 0; j < loopr; j++){
			for(int k = 0; k < loopc; k++){
				max_value[i] = max(max_value[i], in_nu[input_size*i + num_out*in_fm*in_fm + j*in_fm + k + x_str + y_str]);
			}
		}
    	out_nu[output_size*i + out_position] = max_value[i];
	}
}



/* Fully Connected */
__global__ void vgg_fc(float *bias,float *in_nu,float *in_w,float *out_nu,int model_num,int input, bool relu)
{
    int num_out = blockIdx.x;
	int weight = num_out * input;

	int input_size = input;
	int output_size = gridDim.x;

	float result[5] = {0.0};
	for(int i = 0; i < model_num; i++){
		for(int j = 0; j < input; j++){
			result[i] += in_nu[input_size*i + j] * in_w[weight+j];
		}
		result[i] += bias[num_out];
		//ReLU
		if(relu == true){
			result[i] = max(result[i], 0.);
		}
		out_nu[output_size*i + num_out] = result[i];
	}
}
