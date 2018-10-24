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

int main(int argc, char **argv) {

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
            MPI_Finalize();
            exit(-1);
        }
        if (size <= 1) {
            cout << "Za mala liczba procesow" << endl;
            MPI_Finalize();
            exit(1);
        }

        //Wczytanie i tworzenie obrazka
        char *imgName = argv[1];
        char *imgOutName = argv[2];
        Mat img, slice, imgScore;
        img = imread(imgName, CV_LOAD_IMAGE_COLOR);
        cout << "img: " << img.cols << " " << img.rows << endl << endl;
        if (!img.data) {                                        // sprawdzdenie czy wejściowy obrazek istnieje
            cout << "Nie mozna wczytac" << imgName;
            MPI_Finalize();
            return -1;
        }
        copyMakeBorder(img, img, 2, 2, 2, 2, BORDER_REPLICATE);

//        Mat imgScore = Mat(img.rows - 4, img.cols-4, CV_8UC3); // CV_8UC3 - 8-bit unsigned integer matrix/image with 3 channels
        int start = 0, width, end;
        width = end = img.cols / (size - 1) + 2;
        width += 2;

        begin_time = MPI_Wtime(); //rozpoczecie liczenia czasu
        for (int i = 1; i < size; i++) { //rozeslanie obrazka do procesow
            if (i == (size - 1))
                width = img.cols - start;
            slice = Mat(width, img.rows, CV_8UC3);
            slice = img(Rect(start, 0, width, img.rows)).clone();
            start = end - 4;
            end = start + width;
//ROZESLANIE OBRAZKA

            int sliceColumn = slice.cols;
            int sliceRows = slice.rows;
            cout << "wymiary slice: " << sliceColumn << " " << sliceRows << endl;
            MPI_Send(&sliceColumn, 1, MPI_INT, i, 0, comm);
            MPI_Send(&sliceRows, 1, MPI_INT, i, 1, comm);
            MPI_Send(slice.data, sliceColumn * sliceRows * 3, MPI_BYTE, i, 2, comm);
            cout << "obraz wyslany do procesu: " << i << endl;
        }



//ODEBRANIE OBRAZKA

        for (int i = 1; i < size; i++) {
            int tempcols, temprows;
            MPI_Recv(&tempcols, 1, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);
            MPI_Recv(&temprows, 1, MPI_INT, i, 1, comm, MPI_STATUS_IGNORE);
            Mat tempimg = Mat(temprows, tempcols, CV_8UC3);
            MPI_Recv(tempimg.data, tempcols * temprows * 3, MPI_BYTE, i, 2, comm, MPI_STATUS_IGNORE);
            if (i == 1) {
                imgScore = tempimg.clone();
            } else {
                hconcat(imgScore, tempimg, imgScore);
            }
            cout << "rozmiar obrazu po " << i << " odbiorach wynosi: " << imgScore.cols << " " << imgScore.rows << endl;
            cout << "obraz odebrany od: " << i << endl;
        }


        end_time = MPI_Wtime(); // zakonczenie liczenia czasu
        total_time = ((end_time - begin_time) * 1000); // przeksztalcenie otrzymanego czasu do postaci ms
        cout << "Czas: " << total_time << " ms" << endl; //wyświetlanie czasu

        imwrite(imgOutName, imgScore); //zapis obrazka
    } else {
//INNE WATKI

//ODEBRANIE DANYCH
        int proccols, procrows;
        MPI_Recv(&proccols, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&procrows, 1, MPI_INT, 0, 1, comm, MPI_STATUS_IGNORE);

        Mat procimg = Mat(procrows, proccols, CV_8UC3);
        Mat outimg = Mat(procrows, proccols, CV_8UC3);

        MPI_Recv(procimg.data, proccols * procrows * 3, MPI_BYTE, 0, 2, comm, MPI_STATUS_IGNORE);
        cout << "obraz odebrany przez proces: " << rank << endl;

//FILTR GAUSSA
        outimg = Gaussianblur(procimg, outimg);

        Mat sendimg = Mat(procrows - 4, proccols - 4, CV_8UC3);
//WYSLANIE DANYCH

        sendimg = outimg(Rect(2, 2, outimg.cols - 4, outimg.rows - 4)).clone();
        proccols -= 4;
        procrows -= 4;

        MPI_Send(&proccols, 1, MPI_INT, 0, 0, comm);
        MPI_Send(&procrows, 1, MPI_INT, 0, 1, comm);
        MPI_Send(sendimg.data, proccols * procrows * 3, MPI_BYTE, 0, 2, comm);


    }
    MPI_Finalize();
    return 0;
}
