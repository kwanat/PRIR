#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <omp.h>
#include "opencv2/imgproc/imgproc.hpp" //pliki naglowkowe OpenCv
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "mpi.h"
using namespace std;
using namespace cv; //przestrzen nazw OpenCv

MPI_Comm comm;

const int mask[5][5] = {			//druga wersja filtru dolnoprzepustowego wykorzystującego funkcję Gaussa 
   				{1,1,2,1,1},
    		    {1,2,4,2,1},
    		    {2,4,8,4,2},
    		    {1,2,4,2,1},
  				{1,1,2,1,1}
				}; 
int SumMask = 52; //suma liczb w filtrze
Mat Gaussianblur(Mat img, Mat imgScore){
int column, row, R, G, B, x, y;
    int columns = img.cols;  //oblicza liczbe kolumn w obrazku
	int rows = img.rows;	//oblicza liczbe wierszy w obrazku
	
	for(row = 2; row < rows-2; row++) // przeszukiwanie calego obrazka
	{
		for(column = 2; column < columns-2; column++)
		{
			int startA = row -2;
			int startB = column -2;
			Vec3b &pixelResult = imgScore.at<Vec3b>(row, column); //wybrany piksel
			R = 0;
			G = 0;
			B = 0;
			for(x=0; x < 5; x++) // normalizacja maski
				for(y=0; y < 5; y++){
					Vec3b pixelSource = img.at<Vec3b>(startA + x, startB + y);
                    R += pixelSource.val[0] * mask[x][y]; //nowy pixel (R)
                    G += pixelSource.val[1] * mask[x][y]; //nowy pixel (G)
                    B += pixelSource.val[2] * mask[x][y]; //nowy pixel (B)
				}
            pixelResult.val[0] = R/SumMask;  // dzielenie przez sume wszystkich wag maski
            pixelResult.val[1] = G/SumMask;
            pixelResult.val[2] = B/SumMask;
		}
	}
return imgScore;

}

int main(int argc, char **argv)
{
	
int rank, size;
   	
	double begin_time, end_time, total_time; //zmienne wykorzystywane do zliczania czasu dzialania programu

MPI_Init(&argc, &argv);
comm=MPI_COMM_WORLD;
MPI_Comm_rank(comm, &rank);
MPI_Comm_size(comm, &size);

	if(argc!=4){
		cout << "Niepoprawna liczba oargumentow" << argc; //sprawdzenie ilosci argumentow podanych przy wywolaniu programu
		exit(-1);
	}
	int n_thr = atoi(argv[1]); //rzutowanie char do int
	if(n_thr <= 1) //Zabezpieczenie przed niepodaniem liczby wątków
	{
		cout << "Liczba procesow musi byc wieksza niz 1";
		exit(-1);
	}
	

	//Wczytanie i tworzenie obrazka
	char* imgName = argv[2]; 
	char* imgOutName = argv[3];
	Mat img;
	img = imread(imgName, CV_LOAD_IMAGE_COLOR);
Mat imgScore = Mat(img.rows, img.cols, CV_8UC3); // CV_8UC3 - 8-bit unsigned integer matrix/image with 3 channels

	if(!img.data) {										// sprawdzdenie czy wejściowy obrazek istnieje
		cout << "Nie mozna wczytac" << imgName; 
		return -1;
	}

	begin_time = MPI_Wtime(); //rozpoczecie liczenia czasu
imgScore=Gaussianblur(img,imgScore);

	end_time = MPI_Wtime(); // zakonczenie liczenia czasu
	total_time = ((end_time - begin_time)*1000); // przeksztalcenie otrzymanego czasu do postaci ms
	cout << "Czas: " << total_time << " ms" << endl; //wyświetlanie czasu
	
	imwrite(imgOutName, imgScore); //zapis obrazka
	return 0;
}