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

const int mask[5][5] = {            //druga wersja filtru dolnoprzepustowego wykorzystującego funkcję Gaussa
        {1, 1, 2, 1, 1},
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1},
        {1, 1, 2, 1, 1}
};
int SumMask = 52; //suma liczb w filtrze
Mat Gaussianblur(Mat img, Mat imgScore) {
    int column, row, R, G, B, x, y;
    int columns = img.cols;  //oblicza liczbe kolumn w obrazku
    int rows = img.rows;    //oblicza liczbe wierszy w obrazku

    for (row = 2; row < rows - 2; row++) // przeszukiwanie calego obrazka
    {
        for (column = 2; column < columns - 2; column++) {
            int startA = row - 2;
            int startB = column - 2;
            Vec3b &pixelResult = imgScore.at<Vec3b>(row, column); //wybrany piksel
            R = 0;
            G = 0;
            B = 0;
            for (x = 0; x < 5; x++) // normalizacja maski
                for (y = 0; y < 5; y++) {
                    Vec3b pixelSource = img.at<Vec3b>(startA + x, startB + y);
                    R += pixelSource.val[0] * mask[x][y]; //nowy pixel (R)
                    G += pixelSource.val[1] * mask[x][y]; //nowy pixel (G)
                    B += pixelSource.val[2] * mask[x][y]; //nowy pixel (B)
                }
            pixelResult.val[0] = R / SumMask;  // dzielenie przez sume wszystkich wag maski
            pixelResult.val[1] = G / SumMask;
            pixelResult.val[2] = B / SumMask;
        }
    }
    return imgScore;

}

int main(int argc, char** argv) {

    int rank, size;

    double begin_time, end_time, total_time; //zmienne wykorzystywane do zliczania czasu dzialania programu

    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (rank == 0) {

        if (argc != 3) {
            cout << "Niepoprawna liczba oargumentow"
                 << argc; //sprawdzenie ilosci argumentow podanych przy wywolaniu programu
            exit(-1);
        }

        //Wczytanie i tworzenie obrazka
        char *imgName = argv[2];
        char *imgOutName = argv[3];
        Mat img, slice;
        img = imread(imgName, CV_LOAD_IMAGE_COLOR);
	cout<<"img: "<<img.cols<<" "<<img.rows<<endl<<endl;
	cout<<img.data<<endl<<endl;
        if (!img.data) {                                        // sprawdzdenie czy wejściowy obrazek istnieje
            cout << "Nie mozna wczytac" << imgName;
            return -1;
        }
        copyMakeBorder(img, img, 2, 2, 2, 2, BORDER_REPLICATE);

        Mat imgScore = Mat(img.rows - 4, img.cols-4, CV_8UC3); // CV_8UC3 - 8-bit unsigned integer matrix/image with 3 channels
        int start = 0, width, end;
        width = end = img.cols / (size - 1) + 2;
        width += 2;
        begin_time = MPI_Wtime(); //rozpoczecie liczenia czasu
        for (int i = 1; i < size; i++) { //rozeslanie obrazka do procesow
            if (i == (size - 1))
                end = img.cols;
            slice = Mat(width, img.rows, CV_8UC3);
            slice = img(Rect(start, 0, end, img.rows)).clone();
            start = end - 2;
            end = start + width;
//ROZESLANIE OBRAZKA

            int sliceColumn = slice.cols;
            int sliceRows = slice.rows;
cout<<"wymiary slice: "<<sliceColumn<<" "<<sliceRows<<endl;
            MPI_Send(&sliceColumn, sizeof(int), MPI_LONG, i, 0, comm);
            MPI_Send(&sliceRows, sizeof(int), MPI_LONG, i, 1, comm);
            MPI_Send(slice.data, sliceColumn * sliceRows * 3, MPI_BYTE, i, 2, comm);
	cout<<"obraz wyslany do procesu: "<<i<<endl;
        }



//ODEBRANIE OBRAZKA

        for (int i = 1; i < size; i++) {
            int tempcols, temprows;
            MPI_Recv(&tempcols, sizeof(int), MPI_LONG, i, 0, comm, MPI_STATUS_IGNORE);
cout<<"dziala 1"<<endl;
            MPI_Recv(&temprows, sizeof(int), MPI_LONG, i, 1, comm, MPI_STATUS_IGNORE);
cout<<"dziala 2"<<endl<<tempcols<<" "<<temprows<<endl;           
 Mat tempimg = Mat(temprows, tempcols, CV_8UC3);
            MPI_Recv(tempimg.data, tempcols * temprows * 3, MPI_BYTE, i, 2, comm, MPI_STATUS_IGNORE);
cout<<"dziala 3"<<" "<<tempimg.cols<<" "<<tempimg.rows<<" "<<tempimg.data<<endl;        
imgScore=tempimg;   
//imwrite(imgOutName, tempimg); //zapis obrazka
// hconcat(imgScore, tempimg, imgScore);
cout<<"obraz odebrany od: "<<i<<endl;
        }


            end_time = MPI_Wtime(); // zakonczenie liczenia czasu
            total_time = ((end_time - begin_time) * 1000); // przeksztalcenie otrzymanego czasu do postaci ms
            cout << "Czas: " << total_time << " ms" << endl; //wyświetlanie czasu

            imwrite(imgOutName, img); //zapis obrazka
        }else{
//INNE WATKI

//ODEBRANIE DANYCH
            int proccols, procrows;
            MPI_Recv(&proccols, sizeof(int), MPI_LONG, 0, 0, comm, MPI_STATUS_IGNORE);
            MPI_Recv(&procrows, sizeof(int), MPI_LONG, 0, 1, comm, MPI_STATUS_IGNORE);
            Mat procimg = Mat(proccols, procrows, CV_8UC3);
            Mat outimg = Mat(proccols, procrows, CV_8UC3);
cout<<"get: "<<proccols<<" "<<procrows<<endl;
            MPI_Recv(procimg.data, proccols * procrows * 3, MPI_BYTE, 0, 2, comm, MPI_STATUS_IGNORE);
cout<<"obraz odebrany przez proces: "<<rank<<endl;
//FILTR GAUSSA
            outimg = Gaussianblur(procimg, outimg);
cout<<"procimg: "<<procimg.cols<<" "<<procimg.rows<<endl;
cout<<"blur: "<<outimg.cols<<" "<<outimg.rows<<endl;
cout<<"obraz rozmyty przez proces: "<<rank<<endl;
            Mat sendimg = Mat(procrows - 4, proccols - 4, CV_8UC3);
//WYSLANIE DANYCH

            sendimg = outimg(Rect(2, 2, outimg.cols - 4, outimg.rows - 4)).clone();
	proccols-=4;
	procrows-=4; 
cout<<"sending: "<<proccols<<" "<<procrows<<endl;
cout<<"sending: "<<sendimg.cols<<" "<<sendimg.rows<<endl;
            MPI_Send(&proccols, sizeof(int), MPI_LONG, 0, 0, comm);
            MPI_Send(&procrows, sizeof(int), MPI_LONG, 0, 1, comm);
            MPI_Send(sendimg.data, proccols  * procrows  * 3, MPI_BYTE, 0, 2, comm);


        }
        MPI_Finalize();
        return 0;
    }
