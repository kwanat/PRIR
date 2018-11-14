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
    
    if (blockIdx.x < (rowsNumber + 4) % gridDim.x) {			//Obliczanie pozycji startowej dla bloku, gridDim podaje liczbe blokow w siatce
        startForBlock = blockIdx.x * blockSize + blockIdx.x; //blockIdx -indeks bloku w siatce
        blockSize++;
    }
    else {
        startForBlock = blockIdx.x * blockSize + (rowsNumber+4) % gridDim.x;
    }

    if (threadIdx.x < (columnsNumber + 4) % blockDim.x) {		//Obliczanie pozycji startowej dla watku, blockDim podaje liczbę wątków w bloku, w określonym kierunku
        startForThread = threadIdx.x*threadSize + threadIdx.x;
        threadSize++;
    }
    else {
        startForThread = threadIdx.x*threadSize + (columnsNumber+4) % blockDim.x;
    }
    //Obliczenia 
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
    int blockNumber; //cuda cores
    int threadNumber = 1;
    cudaEvent_t start, stop; //deklaracja zmiennych licznika
    float elapsedTime; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    if (argc != 4) {                //sprawdzenie ilosci argumentow podanych przy wywolaniu programu
        cout << "Niepoprawna liczba oargumentow"<<endl;
        exit(1);
    } 

    int threadsCount = atoi(argv[1]);
    if (threadsCount <= 0) {
        cout << "Niepoprawna liczba argumentow\n";
        exit(-1);
    }
    blockNumber = threadsCount;

    //Load and create image
    char *imgName = argv[2];
    char *imgOutName = argv[3];
    
    Mat img;
    img = imread(imgName, CV_LOAD_IMAGE_COLOR); // wczytanie obrazu wejsciowego
 
    if (!img.data) {
        cout << "Nie mozna wczytac" << imgName;
        return -1;
    }
 
    Mat frame[3];   //destination array - tablica docelowa
    split(img, frame); //split source - dzielenie na tablice
 
    //Przygotowanie rozmiarów obrazka i rozmiary fragmentów:
    int rows = img.rows; //liczba wierszy w obrazku
    int columns = img.cols;	//liczba kolumn w obrazku
    int sizePerBlock = rows / blockNumber; //rozmiar na blok
    int sizeForThread = columns / threadNumber; //rozmiar dla wątku
   //cout << rows << "  " << columns << "  "<< sizePerBlock <<"  "<< sizeForThread  << endl;
 
    //Prepare data to upload
    unsigned char* R;
    unsigned char* G;
    unsigned char* B;
    unsigned char* cudaR;
    unsigned char* cudaG;
    unsigned char* cudaB;
 
    //Load data on GPU memory
    cudaMalloc(&cudaR, rows*columns*sizeof(unsigned char)); //Alokuje pamięć na karcie graficznej.
    cudaMalloc(&R, (rows)*(columns) * sizeof(unsigned char));
    cudaMemcpy(cudaR, &frame[2].data[0], rows*columns * sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMalloc(&cudaG, rows*columns * sizeof(unsigned char));
    cudaMalloc(&G, (rows)*(columns) * sizeof(unsigned char));
    cudaMemcpy(cudaG, &frame[1].data[0], rows*columns * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMalloc(&cudaB, rows*columns * sizeof(unsigned char));
    cudaMalloc(&B, (rows)*(columns ) * sizeof(unsigned char));
    cudaMemcpy(cudaB, &frame[0].data[0], rows*columns * sizeof(unsigned char), cudaMemcpyHostToDevice);
 
    //Do calculations
    cudaEventRecord(start);
    gaussianBlur <<< blockNumber, threadNumber >>> (cudaR, cudaG, cudaB, R, G, B, sizePerBlock, sizeForThread, rows-4, columns-4); //uruchomienie jądra
    cudaEventRecord(stop); //zatrzymanie licznika i zczytanie czasu obliczen
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Czas : %f ms\n", elapsedTime);
 
    //Wczytanie rozmytego obrazu z GPU
    unsigned char* resultBlue; //deklaracja wskaznikow 
    unsigned char* resultRed;
    unsigned char* resultGreen;
 
    resultBlue = new unsigned char[(rows)*(columns)]; //alokacja pamieci dla wskaznikow pikseli
    resultRed = new unsigned char[(rows)*(columns)];
    resultGreen = new unsigned char[(rows)*(columns)];
 
    cudaMemcpy(resultRed,R,(rows)*(columns) * sizeof(unsigned char), cudaMemcpyDeviceToHost); //kopiowanie miedzy karta a RAM, (1 arg to obszar do ktorego jest kopiowane, drugi to obszar pamieci z ktorego jest kopiowane, rozmiar do skopowania, kierunek transferu)
    cudaMemcpy(resultBlue, B, (rows)*(columns) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultGreen, G, (rows)*(columns) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
 
    Mat imgResult;
    Mat Red = Mat(rows, columns, CV_8UC1, resultRed);
    Mat Green = Mat(rows, columns, CV_8UC1, resultGreen);
    Mat Blue = Mat(rows, columns, CV_8UC1, resultBlue);
    vector<Mat> tab; //tablica do scalenia
 
    tab.push_back(Blue); //dodaje do końca tablicy nowy px B
    tab.push_back(Green); //dodaje do końca tablicy nowy px G
    tab.push_back(Red); //dodaje do końca tablicy nowy px R
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
