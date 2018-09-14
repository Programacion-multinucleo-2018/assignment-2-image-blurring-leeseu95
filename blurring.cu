//Seung Hoon Lee - A01021720
//Tarea 2 - Blurring CUDA Version

#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include <cuda_runtime.h>

using namespace std;

// input - input image one dimensional array
// ouput - output image one dimensional array
// width, height - width and height of the images
// colorWidthStep - number of color bytes (cols * colors)
// grayWidthStep - number of gray bytes 
__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep)
{
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("width: %d", width);
    // printf("height: %d", height);

	// Only valid threads perform memory I/O
	if ((xIndex < width) && (yIndex < height))
	{
		//Location of colored pixel in input
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

		// Location of gray pixel in output
		// const int gray_tid = yIndex * grayWidthStep + xIndex;
		// const unsigned char blue = input[color_tid];
		// const unsigned char green = input[color_tid + 1];
		// const unsigned char red = input[color_tid + 2];
        //Funciones de clases, no la vamos a usar porque no es graying

        //Definiciones de nuestros pixeles RGB
        int redPixel = 0;
        int greenPixel = 0;
		int bluePixel = 0;
		int size = 9;
		int texel = 0;
		
		//Iteramos a traves del filter matrix 3 x 3 como lo dice el algoritmo en wikipedia
		//https://en.wikipedia.org/wiki/Box_blur
		for(int uvX = -1; uvX <= 1; uvX ++) { //Itermaos desde posicion del pixel -1, el pixel y el pixel + 1
			for(int uvY = -1; uvY <= 1; uvY ++) {
				//Como son pixeles de RGB, iteramos a traves de uvX * 3 (ya que son 3 pixeles horizontalmente y asi iteramos)
				//Igual multiplicamos el width y * 3 en uvY para iterar verticalmente (Arthur me ayudo con la explicacion para encontrar el texel)
				//Si le sumas al texel el uvY * 3 * width, agarras el de abajo o arriba depeendiendo de uvY
				texel = color_tid+(uvX*3)+(uvY*width*3); //Direccion de los pixeles alrededor de la matriz

				redPixel += input[texel];
				greenPixel += input[texel+1];
				bluePixel += input[texel+2];
			}
		}

		//Blurring con una matriz de 10 x 10 para ver si se hace mas fuerte el blurring
		//Iteramos a traves del filter matrix 3 x 3 como lo dice el algoritmo en wikipedia
		// for(int uvX = -5; uvX <= 5; uvX ++) { //Itermaos desde posicion del pixel -1, el pixel y el pixel + 1
		// 	for(int uvY = -5; uvY <= 5; uvY ++) {
		// 		texel = (yIndex+uvY) * colorWidthStep + (3 * (xIndex+uvX)); //Direccion de los pixeles alrededor de la matriz

		// 		redPixel += input[texel+2];
		// 		greenPixel += input[texel+1];
		// 		bluePixel += input[texel];
		// 	}
		// }

		output[color_tid] = static_cast<unsigned char>(redPixel/size);
		output[color_tid+1] = static_cast<unsigned char>(greenPixel/size);
		output[color_tid+2] = static_cast<unsigned char>(bluePixel/size);


        //No lo vamos a usar porque n oes graying
		// The standard NTSC conversion formula that is used for calculating the effective luminance of a pixel (https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems)
        // const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;
		// Alternatively, use an average
		//const float gray = (red + green + blue) / 3.f;

		// output[gray_tid] = static_cast<unsigned char>(gray);
	}
}

void convert_to_blur(const cv::Mat& input, cv::Mat& output)
{
	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors	
	size_t colorBytes = input.step * input.rows;

	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, colorBytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	int xBlock = 16;
	int yBlock = 1024;
	// Specify a reasonable block size
	const dim3 block(xBlock, yBlock);

	// Calculate grid size to cover the whole image
	// const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));
	printf("blur_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);
	
	// Launch the color conversion kernel
	auto start_cpu =  chrono::high_resolution_clock::now();
	blur_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step));
	auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

	printf("La cantidad de tiempo que se tarda cada ejecucion es alrededor de: %f ms con bloque de %d y %d\n", duration_ms.count(), xBlock, yBlock);

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, colorBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}

int main(int argc, char *argv[])
{
	string imagePath;
	
	if(argc < 2)
		imagePath = "imageHD.jpg";
  	else
  		imagePath = argv[1];

	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	//Create output image
	cv::Mat output(input.rows, input.cols, CV_8UC3); //Se tiene que cambiar a CV_8UC3 en vez de CV_8UC1

	//Call the wrapper function
	convert_to_blur(input, output);

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
