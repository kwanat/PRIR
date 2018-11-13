#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>

#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
 
using namespace cv;
using namespace std;
 
__global__ void gaussianBlur(unsigned char* R, unsigned char* G, unsigned char* B,
                             unsigned char* resultRed, unsigned char* resultGreen, unsigned char* resultBlue,
                             int blockSize, int threadSize, int rowsNumber, int columnsNumber) {
    
    //Initialize Data
const int mask[5][5] = {            //druga wersja filtru dolnoprzepustowego wykorzystującego funkcję Gaussa
        {1, 1, 2, 1, 1},
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1},
        {1, 1, 2, 1, 1}
    };
    int SumMask = 52;   //suma liczb w filtrze
    unsigned int row, column, x, y, red, green, blue;
    int  startForBlock, startForThread;
    //Calculate start position for Block:
    if (blockIdx.x < (rowsNumber + 4) % gridDim.x) {
        startForBlock = blockIdx.x * blockSize + blockIdx.x;
        blockSize++;
    }
    else {
        startForBlock = blockIdx.x * blockSize + (rowsNumber+4) % gridDim.x;
    }
    //Calcualte start position for Thread
    if (threadIdx.x < (columnsNumber + 4) % blockDim.x) {
        startForThread = threadIdx.x*threadSize + threadIdx.x;
        threadSize++;
    }
    else {
        startForThread = threadIdx.x*threadSize + (columnsNumber+4) % blockDim.x;
    }
    //Do Calculations
    for (row = startForBlock; row < (startForBlock + blockSize); row++) {
        if (row >= rowsNumber) {
            break;
        }
        for (column = startForThread; column < (startForThread + threadSize); column++) {
            if (column >= columnsNumber) {
                break;
            }
            red = 0;
            green = 0;
            blue = 0;
            for (x = 0; x < 5; x++) //normalizacja
                for (y = 0; y < 5; y++) {
                    red += R[(row + x) * columnsNumber + column + y] * mask[x][y]; //nowy pixel (R)
                    green += G[(row + x) * columnsNumber + column + y] * mask[x][y]; //nowy pixel (G)
                    blue += B[(row + x) * columnsNumber + column + y] * mask[x][y]; //nowy pixel (B)
                }
            resultRed[row * columnsNumber + column] = red / SumMask; // dzielenie przez sume wszystkich wag maski
            resultGreen[row * columnsNumber + column] = green / SumMask;
            resultBlue[row * columnsNumber + column] = blue / SumMask;   
        }
    }
 
}
 
int main(int argc, char **argv)
{
    int blockNumber = 500; //cuda cores
    int threadNumber = 1024;
    int threadsCount; //liczba watkow
    cudaEvent_t start, stop; //deklaracja zmiennych licznika
    float elapsedTime; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (argc != 4) {                //sprawdzenie ilosci argumentow podanych przy wywolaniu programu
        cout << "Niepoprawna liczba oargumentow"<<endl;
        exit(1);
    } 

    threadsCount = atoi(argv[1]);
    if (threadsCount <= 0 || threadsCount > 15) {
        cout << "Liczba rdzeni niepoprawna";
        exit(-1);
    }

    blockNumber=threadsCount;
    //Load and create image
    char *imgName = argv[2];
    char *imgOutName = argv[3];
    
    Mat img;
    img = imread(imgName, CV_LOAD_IMAGE_COLOR); // wczytanie obrazu wejsciowego
 
    if (!img.data) {
        cout << "Nie mozna wczytac" << imgName;
        getchar();
        return -1;
    }
 
    Mat background[3];   //destination array
    split(img, background); //split source
 
    //Prepare size of image and fragment sizes:
    int rows = img.rows;
    int columns = img.cols;
    int sizeForBlock = rows / blockNumber;
    int sizeForThread = columns / threadNumber;
   //cout << rows << "  " << columns << "  "<< sizeForBlock <<"  "<< sizeForThread  << endl;
 
    //Prepare data to upload
    unsigned char* R;
    unsigned char* G;
    unsigned char* B;
    unsigned char* cudaR;
    unsigned char* cudaG;
    unsigned char* cudaB;
 
    //Load data on GPU memory
    cudaMalloc(&cudaR, rows*columns*sizeof(unsigned char));
    cudaMalloc(&R, (rows)*(columns) * sizeof(unsigned char));
    cudaMemcpy(cudaR, &background[2].data[0], rows*columns * sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMalloc(&cudaG, rows*columns * sizeof(unsigned char));
    cudaMalloc(&G, (rows)*(columns) * sizeof(unsigned char));
    cudaMemcpy(cudaG, &background[1].data[0], rows*columns * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMalloc(&cudaB, rows*columns * sizeof(unsigned char));
    cudaMalloc(&B, (rows)*(columns ) * sizeof(unsigned char));
    cudaMemcpy(cudaB, &background[0].data[0], rows*columns * sizeof(unsigned char), cudaMemcpyHostToDevice);
 
    //Do calculations
    cudaEventRecord(start);
    gaussianBlur <<< blockNumber, threadNumber >>> (cudaR, cudaG, cudaB, R, G, B, sizeForBlock, sizeForThread, rows-4, columns-4); //uruchomienie jądra
    cudaEventRecord(stop); //zatrzymanie licznika i zczytanie czasu obliczen
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Czas : %f ms\n", elapsedTime);
 
    //Wczytanie rozmytego obrazu z GPU
    unsigned char* resultBlue;
    unsigned char* resultRed;
    unsigned char* resultGreen;
 
    resultBlue = new unsigned char[(rows)*(columns)];
    resultRed = new unsigned char[(rows)*(columns)];
    resultGreen = new unsigned char[(rows)*(columns)];
 
    cudaMemcpy(resultRed,R,(rows)*(columns) * sizeof(unsigned char),cudaMemcpyDeviceToHost);
    cudaMemcpy(resultBlue, B, (rows)*(columns) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultGreen, G, (rows)*(columns) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
 
    Mat imgResult;
    Mat Red = Mat(rows, columns, CV_8UC1, resultRed);
    Mat Green = Mat(rows, columns, CV_8UC1, resultGreen);
    Mat Blue = Mat(rows, columns, CV_8UC1, resultBlue);
    vector<Mat> tab; //tablica do scalenia
 
    tab.push_back(Blue);
    tab.push_back(Green);
    tab.push_back(Red);
    merge(tab, imgResult); //scalanie 
    imwrite(imgOutName, Mat(imgResult, Rect(0, 0, columns - 4, rows - 4)));
 
    cudaFree(&R);
    cudaFree(&B);
    cudaFree(&G);
    cudaFree(&cudaR);
    cudaFree(&cudaB);
    cudaFree(&cudaG);
 

 
    return 0;
}
