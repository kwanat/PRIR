#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <omp.h>
#include "opencv2/imgproc/imgproc.hpp" //pliki naglowkowe OpenCv
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv; //przestrzen nazw OpenCv

int DivSumMask(int value, int SumMask) {

	value = value/SumMask;
	if(value > 255)
	{
		value = 255;
	}
	if(value < 0)
	{
		value = 0;
	}
	return value;
}

int main(int argc, char **argv)
{
	int column, row, R, G, B, x, y;
   	const int mask[5][5] = {
   		{1,1,2,1,1},
        {1,2,4,2,1},
        {2,4,8,4,2},
        {1,2,4,2,1},
  		{1,1,2,1,1}
	}; 
int SumMask = 52;
double begin_time, end_time, total_time;

	if(argc!=4){
		cout << "Niepoprawna liczba oargumentow" << argc;
		exit(-1);
	}
	int n_thr = atoi(argv[1]);
	if(n_thr <= 0 || n_thr > 15)
	{
		cout << "Numer watku musi byc poprawny i mniejszy niz 16";
		exit(-1);
	}
	

	//Wczytanie i tworzenie obrazka
	char* imgName = argv[2];
	char* imgOutName = argv[3];
	Mat img;
	img = imread(imgName, CV_LOAD_IMAGE_COLOR);

	if(!img.data) {
		cout << "Nie mozna wczytac" << imgName;
		return -1;
	}

    int columns = img.cols;
	int rows = img.rows;
	Mat imgScore = Mat(rows-4, columns-4, CV_8UC3);
	//Macierz
	begin_time = omp_get_wtime();
	#pragma omp parallel for shared(img, imgScore) firstprivate(mask, SumMask) private(x, y , column, row) num_threads(n_thr)
	for(row = 2; row < rows-2; row++)
	{
		for(column = 2; column < columns-2; column++)
		{
			int startA = row -2;
			int startB = column -2;
			Vec3b &pixelResult = imgScore.at<Vec3b>(startA, startB);
			R = 0;
			G = 0;
			B = 0;
			for(x=0; x < 5; x++)
				for(y=0; y < 5; y++){
					Vec3b pixelSource = img.at<Vec3b>(startA + x, startB + y);
                    R += pixelSource.val[0] * mask[x][y];
                    G += pixelSource.val[1] * mask[x][y];
                    B += pixelSource.val[2] * mask[x][y];
				}
			R = R/SumMask;
			G = G/SumMask;
			B = B/SumMask;
            pixelResult.val[0] = R;
            pixelResult.val[1] = G;
            pixelResult.val[2] = B;
		}
	}
	end_time = omp_get_wtime();
	total_time = ((end_time - begin_time)*1000);
	cout << "Czas: " << total_time << " ms" << endl;
	//zapis obrazka
	imwrite(imgOutName, imgScore);
	return 0;
}
