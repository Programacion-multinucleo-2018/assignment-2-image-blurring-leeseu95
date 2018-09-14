//Seung Hoon Lee - A01021720
//Tarea 2 - Blurring OMP Version
//g++ -o blurring_OMP blurring_OMP.cpp -lopencv_core -lopencv_highgui -lope ncv_imgproc -std=c++11

#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "omp.h"

using namespace std;

void blurring(cv::Mat& input, cv::Mat& output) //Le pasamos de parametro solo el input , output con CV
{
	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors	
	// size_t colorBytes = input.step * input.rows;
	//Codigo que no vamos a utilizar del programa de cuda

    //Variables para OMP
    int i, j, texelX, texelY;
    #pragma omp parallel private(i, j, texelX, texelY) shared (input, output)
    {
        for(i = 0; i < input.rows; i++) {
            for(j = 0; j < input.cols; j++) { //Iteramos a traves de la matriz
                //Definiciones de nuestros pixeles RGB
                int redPixel = 0;
                int greenPixel = 0;
                int bluePixel = 0;

                //Definicion de tamanos y texels
                int size = 9;
                texelX = 0;
                texelY = 0;
                
                // Location of colored pixel in input
                // const int color_tid = j * input.step + (3 * i);

                //Iteramos a traves del filter matrix 3 x 3 como lo dice el algoritmo en wikipedia
                //https://en.wikipedia.org/wiki/Box_blur
                for(int uvX = -1; uvX <= 1; uvX ++) { //Itermaos desde posicion del pixel -1, el pixel y el pixel + 1
                    for(int uvY = -1; uvY <= 1; uvY ++) {
                        texelX = uvX + i; //Direccion de los pixeles alrededor de la matriz
                        texelY = uvY + j; //Direccion de los pixeles alrededor de la matriz
                        // if(texelX <= 0 && texelX > input.rows && texelY <= 0 && texelY > input.cols){ //Si se encuentra fuera del espacio
                        //     cout << "fuera del espacio" << endl;
                        // } else {
                        //     // cout << "gsadga" << endl;
                        //     redPixel += input.at<cv::Vec3b>(texelX, texelY)[0];
                        //     greenPixel += input.at<cv::Vec3b>(texelX, texelY)[1];
                        //     bluePixel += input.at<cv::Vec3b>(texelX, texelY)[2];
                        // }
                        if(texelX > 0 && texelX < input.rows && texelY > 0 && texelY < input.cols) { //Se puede cambiar lo de arriba a esto
						redPixel += input.at<cv::Vec3b>(texelX, texelY)[0];
						greenPixel += input.at<cv::Vec3b>(texelX, texelY)[1];
						bluePixel += input.at<cv::Vec3b>(texelX, texelY)[2];
					    }
                    }
                }

                output.at<cv::Vec3b>(i, j)[0] = (redPixel/size);
                output.at<cv::Vec3b>(i, j)[1] = (greenPixel/size);
                output.at<cv::Vec3b>(i, j)[2] = (bluePixel/size);
            }
	    }
    }
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
	auto start_cpu =  chrono::high_resolution_clock::now();
	blurring(input, output);
	auto end_cpu =  chrono::high_resolution_clock::now();
	chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

	printf("La cantidad de tiempo que se tarda cada ejecucion es alrededor de: %f ms\n", duration_ms.count());

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
