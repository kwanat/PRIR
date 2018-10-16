#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "opencv2/imgproc/imgproc.hpp" //pliki naglowkowe OpenCv
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv; //przestrzen nazw OpenCb

int DivSumMask(int value, int SumMask) {

	value = value/SumMask;
	if(value < 0)
	{
		value = 0;
	}
	if(value > 255)
	{
		value = 255;
	}
	return value;
}

int main(int argc, char **argv)
{
	int column, row, R, G, B, x, y;
    int mask[5][5] = {{1,1,2,1,1},
                    				    {1,2,4,2,1},
                    				    {2,4,8,4,2},
                    				    {1,2,4,2,1},
									  	{1,1,2,1,1}
					 					}; 
int SumMask = 52;
	struct timeval start, end;
	long sec, usec;

	if(argc!=4){
		cout << "Must be three arguments" << argc;
		exit(-1);
	}
	int n_thr = atoi(argv[1]);
	if(n_thr <= 0 || n_thr > 15)
	{
		cout << "Number of core must be correct and not fewer than 16";
		exit(-1);
	}
	

	//Wczytanie i tworzenie obrazka
	char* imgName = argv[2];
	char* imgOutName = argv[3];
	Mat img;
	img = imread(imgName, CV_LOAD_IMAGE_COLOR);

	if(!img.data) {
		cout << "Coudn't load" << imgName;
		return -1;
	}

    int columns = img.cols;
	int rows = img.rows;
	Mat imgScore = Mat(rows-4, columns-4, CV_8UC3);
	//Macierz
	gettimeofday(&start, NULL);
	omp_set_dynamic(0);
	#pragma omp parallel for shared(img, imgScore) firstprivate(mask, SumMask) private(x, y , column, row) num_threads(n_thr)
	for(row = 2; rows-2 > row; row++)
	{
		for(column = 2; columns-2 > column; column++)
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
	gettimeofday(&end, NULL);
	//czas ktory uplynal
	sec = end.tv_sec - start.tv_sec;
	usec = end.tv_usec - start.tv_usec;
	cout << "Czas: " << (((sec) * 1000 + usec / 1000.0) + 0.5) << "ms" << endl;
	//zapis obrazka
	imwrite(imgOutName, imgScore);
	return 0;
}
