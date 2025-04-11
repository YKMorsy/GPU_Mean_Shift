#include <vector>
#include <cuda.h>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include "lodepng.h"

#define NUM_THREADS 256
#define TILE_WIDTH 256
#define MAX_ITERATIONS 15
#define BW 10
#define rev_sqrt_two_pi 0.3989422804
#define rev_two_pi rev_sqrt_two_pi*rev_sqrt_two_pi

struct float6 {
    float x, y, z, w, u, v;

    __host__ __device__ float6 operator-(const float6& other) const {
        float6 result;
        result.x = x - other.x;
        result.y = y - other.y;
        result.z = z - other.z;
        result.w = w - other.w;
        result.u = u - other.u;
        result.v = v - other.v;
        return result;
    }

    // Addition assignment operator (+=)
    __host__ __device__ float6& operator+=(const float6& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        w += other.w;
        u += other.u;
        v += other.v;
        return *this;
    }

    // Scalar multiplication operator (*)
    __host__ __device__ float6 operator*(float scalar) const {
        float6 result;
        result.x = x * scalar;
        result.y = y * scalar;
        result.z = z * scalar;
        result.w = w * scalar;
        result.u = u * scalar;
        result.v = v * scalar;
        return result;
    }

    // Compound scalar multiplication operator (*=)
    __host__ __device__ float6& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        w *= scalar;
        u *= scalar;
        v *= scalar;
        return *this;
    }

    __host__ __device__ float6& operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        w /= scalar;
        u /= scalar;
        v /= scalar;
        return *this;
    }
};

__host__ __device__ float6 make_float6(float x, float y, float z, float w, float u, float v) {
    float6 result;
    result.x = x;
    result.y = y;
    result.z = z;
    result.w = w;
    result.u = u;
    result.v = v;
    return result;
}

__device__ __host__ float gaussian_kernel(float dist2, float bandwidth) {
    const float rev_bandwidth = 1.0 / (bandwidth * sqrtf(2.0f * M_PI));
    return expf(-0.5f * dist2 / (bandwidth * bandwidth)) * rev_bandwidth;
}

__global__ void cuda_MeanShift_SharedMemory_2D(float *X, const float *I, const float * originalPoints, const int N, const int dim) {

	__shared__ float tile[TILE_WIDTH][6];

	// for each pixel
	int tx = threadIdx.x;
	int row = blockIdx.x*blockDim.x + tx;

    float6 numerator = make_float6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	float denominator = 0.0;
	int it = row * dim;

	for (int tile_i = 0; tile_i < (N - 1) / TILE_WIDTH + 1; ++tile_i) {
		//loading phase - each thread load something into shared memory
		int row_t = tile_i * TILE_WIDTH + tx;

		int index = row_t * dim;
		if (row_t < N) {
			tile[tx][0] = originalPoints[index];
			tile[tx][1] = originalPoints[index + 1];
			tile[tx][2] = originalPoints[index + 2];
			tile[tx][3] = originalPoints[index + 3];
			tile[tx][4] = originalPoints[index + 4];
			tile[tx][5] = originalPoints[index + 5];
		}
		else {
			tile[tx][0] = 0.0;
			tile[tx][1] = 0.0;
			tile[tx][2] = 0.0;
			tile[tx][3] = 0.0;
			tile[tx][4] = 0.0;
			tile[tx][5] = 0.0;
		}
		__syncthreads();
		//end of loading into shared memory

		if (row < N) // only the threads inside the bounds do some computation
		{
			float6 x_i = make_float6(I[it], I[it + 1], I[it + 2], I[it + 3], I[it + 4], I[it + 5]); //load input point

			//computing phase
			for (int j = 0; j < TILE_WIDTH; ++j) {
				float6 x_j = make_float6(tile[j][0], tile[j][1], tile[j][2], tile[j][3], tile[j][4], tile[j][5]); //from shared memory
				float6 sub = x_i - x_j;
                float distance6 = sub.x * sub.x + sub.y * sub.y + sub.z * sub.z + sub.w * sub.w + sub.u * sub.u + sub.v * sub.v;
				float weight = gaussian_kernel(distance6, BW);
				numerator += x_j * weight; //accumulating
				denominator += weight;

			}
		}
		__syncthreads();
		//end of computing phase for tile_ij
	}

	if (row < N) {
		//storing
		numerator /= denominator;
        X[row * dim] = numerator.x;
        X[row * dim + 1] = numerator.y;
        X[row * dim + 2] = numerator.z;
        X[row * dim + 3] = numerator.w;
        X[row * dim + 4] = numerator.u;
        X[row * dim + 5] = numerator.v;
	}

}


__global__ void cuda_MeanShift_2D(float *X, float *I, float * originalPoints, int N, int dim) 
{
    int tx = threadIdx.x;
    int row = blockIdx.x * blockDim.x + tx;

    float6 numerator = make_float6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    float denominator = 0.0;

    if (row < N) {
        float6 y_i = make_float6(I[row * dim], I[row * dim + 1], I[row * dim + 2], 
                                  I[row * dim + 3], I[row * dim + 4], I[row * dim + 5]); // load input point

        // computing mean shift
        for (int j = 0; j < N; ++j) {
            float6 x_j = make_float6(originalPoints[j * dim], originalPoints[j * dim + 1], 
                                      originalPoints[j * dim + 2], originalPoints[j * dim + 3],
                                      originalPoints[j * dim + 4], originalPoints[j * dim + 5]); // from central gpu memory
            float6 sub = y_i - x_j;
            float distance6 = sub.x * sub.x + sub.y * sub.y + sub.z * sub.z + sub.w * sub.w + sub.u * sub.u + sub.v * sub.v;
            float weight = gaussian_kernel(distance6, BW);
            numerator += x_j * weight; // accumulating
            denominator += weight;
        }

        numerator /= denominator;
        // Update the output data with the new position
        X[row * dim] = numerator.x;
        X[row * dim + 1] = numerator.y;
        X[row * dim + 2] = numerator.z;
        X[row * dim + 3] = numerator.w;
        X[row * dim + 4] = numerator.u;
        X[row * dim + 5] = numerator.v;
    }
}

int main(int argc, char **argv)
{
    std::vector<unsigned char> colorImage;
    unsigned int w, h;
    unsigned int error = lodepng::decode(colorImage, w, h, "color_png/00000.png");
    if (error) {
        std::cout << "1. Decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        return -1;
    }

    std::vector<unsigned char> depthImage;
    unsigned int w2, h2;
    error = lodepng::decode(depthImage, w2, h2, "depth/00000.png");
    if (error) {
        std::cout << "2. Decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        return -1;
    }

    if (w != w2 || h != h2) {
        std::cerr << "Error: Dimension mismatch between color and depth images" << std::endl;
        return -1;
    }

    float* inputData = (float*)malloc(6 * w * h * sizeof(float));

    for (unsigned int y = 0; y < h; y++) {
        for (unsigned int x = 0; x < w; x++) {
            // Set x and y coordinates
            inputData[(y * w + x) * 6] = (float) x;
            inputData[(y * w + x) * 6 + 1] = (float) y;
            
            // Set depth value
            inputData[(y * w + x) * 6 + 2] = (float) depthImage[(y * w + x) * 4]; 

            // Set color values
            inputData[(y * w + x) * 6 + 3] = (float) colorImage[(y * w + x) * 4];     
            inputData[(y * w + x) * 6 + 4] = (float) colorImage[(y * w + x) * 4 + 1]; 
            inputData[(y * w + x) * 6 + 5] = (float) colorImage[(y * w + x) * 4 + 2]; 
        }
    }

    int vecDim = 6;
    int datasetDim = 6*w*h/vecDim;

    auto start_total = std::chrono::high_resolution_clock::now();

    float* inputData_d;
    float* originalData_d;
    float* outputData_d;

    cudaMalloc((void **)&inputData_d, 6 * w * h * sizeof(float));
    cudaMalloc((void **)&originalData_d, 6 * w * h * sizeof(float));
    cudaMalloc((void **)&outputData_d, 6 * w * h * sizeof(float));

	cudaMemcpy(inputData_d, inputData, sizeof(float) * 6 * w * h, cudaMemcpyHostToDevice);
    cudaMemcpy(originalData_d, inputData, sizeof(float) * 6 * w * h, cudaMemcpyHostToDevice);
    cudaMemcpy(outputData_d, inputData, sizeof(float) * 6 * w * h, cudaMemcpyHostToDevice);

    dim3 blockDim = dim3(NUM_THREADS);
    dim3 gridDim((datasetDim + blockDim.x - 1) / blockDim.x);
	
    for (int i = 0; i < MAX_ITERATIONS; i++) 
    {
		std::cout << "Iteration n: " << i << " started." << std::endl;
        cuda_MeanShift_SharedMemory_2D <<<gridDim, blockDim >>> (outputData_d, inputData_d, originalData_d, datasetDim, vecDim);
		cudaDeviceSynchronize();
		std::swap(inputData_d, outputData_d);
	}
    std::swap(inputData_d, outputData_d);

    float* outputData = (float*)malloc(6 * w * h * sizeof(float));
    cudaMemcpy(outputData, outputData_d, sizeof(float) * 6 * w * h, cudaMemcpyDeviceToHost);

    std::vector<unsigned char> outputImage(4 * w * h);
    for (unsigned int y = 0; y < h; y++) {
        for (unsigned int x = 0; x < w; x++) {
            int idx = (y * w + x) * 6;
            outputImage[(y * w + x) * 4] = static_cast<unsigned char>(outputData[idx + 3]);
            outputImage[(y * w + x) * 4 + 1] = static_cast<unsigned char>(outputData[idx + 4]);
            outputImage[(y * w + x) * 4 + 2] = static_cast<unsigned char>(outputData[idx + 5]);
            outputImage[(y * w + x) * 4 + 3] = 255;
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    auto duration_total = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total);
    std::cout << "Total execution time: " << duration_total.count()/1000 << " seconds" << std::endl;


    lodepng::encode("segmented_output.png", outputImage, w, h);

    cudaFree(inputData_d);
    cudaFree(originalData_d);
    cudaFree(outputData_d);
    free(inputData);
    free(outputData);

    return 0;

}