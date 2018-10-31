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

    int rank, size;     // numer procesu oraz liczba wszystkich procesow

    double begin_time, end_time, total_time; //zmienne wykorzystywane do zliczania czasu dzialania programu

    MPI_Init(&argc, &argv);     //inicjalizacja MPI
    comm = MPI_COMM_WORLD;      //pobranie komunikatora
    MPI_Comm_rank(comm, &rank); //pobranie numeru procesu
    MPI_Comm_size(comm, &size); //pobranie liczby wszystkich procesow
    if (rank == 0) {            // jesli jestesmy w procesie macierzystym

        if (argc != 3) {            //sprawdzenie liczby argumentow programu
            cout << "Niepoprawna liczba oargumentow"
                 << argc; //sprawdzenie ilosci argumentow podanych przy wywolaniu programu
            MPI_Finalize();  //Terminates MPI execution environment
            exit(-1);
        }
        if (size <= 1) { //Zbyt mała liczba procesow
            cout << "Za mala liczba procesow" << endl;
            MPI_Finalize();
            exit(-1);
        }

        //Wczytanie i tworzenie obrazka
        char *imgName = argv[1];    //obraz wejsciowy - nazwa
        char *imgOutName = argv[2]; //obraz wyjsciowy - nazwa
        Mat img, slice, imgScore;
        img = imread(imgName, CV_LOAD_IMAGE_COLOR); // wczytanie obrazu wejsciowego
        cout << "img: " << img.cols << " " << img.rows << endl << endl;
        if (!img.data) {                                        // sprawdzdenie czy wejściowy obrazek istnieje
            cout << "Nie mozna wczytac" << imgName;
            MPI_Finalize();
            return -1;
        }
        copyMakeBorder(img, img, 2, 2, 2, 2, BORDER_REPLICATE); //powielenie granic obrazu o 2 w kazda strone

        //PRZYGOTOWANIE KAWALKOW OBRAZU
        int start = 0, width, end;
        width = end = img.cols / (size - 1) + 2; //szerokosc kawalka
        width += 2; //zakladka pikselowa

        begin_time = MPI_Wtime(); //rozpoczecie liczenia czasu
        for (int i = 1; i < size; i++) { //rozeslanie obrazka do procesow
            if (i == (size - 1)) //jesli ostatni proces to szerokosc rowna odleglosci punktu startowego od liczby kolumn calego obrazu
                width = img.cols - start;
            slice = Mat(width, img.rows, CV_8UC3); // kawalek obrazu
            slice = img(Rect(start, 0, width, img.rows)).clone(); //wyciecie kawalka z obrazu wejsciowego
            start = end - 4; // przejscie do nastepnego punktu startowego
            end = start + width; // przejscie do kolejnego punktu koncowego

            //ROZESLANIE OBRAZKA
            int sliceColumn = slice.cols;   //liczba kolumn kawalka
            int sliceRows = slice.rows;     //liczba wierszy kawałka
            cout << "wymiary slice: " << sliceColumn << " " << sliceRows << endl;
            MPI_Send(&sliceColumn, 1, MPI_INT, i, 0, comm);     // wyslanie liczby kolumn typu int do procesu i
            MPI_Send(&sliceRows, 1, MPI_INT, i, 1, comm);       // wyslanie liczby wierszy typu int do procesu i
            MPI_Send(slice.data, sliceColumn * sliceRows * 3, MPI_BYTE, i, 2, comm);    //wyslanie obrazka jako typu byte
            cout << "obraz wyslany do procesu: " << i << endl;
        }



//ODEBRANIE OBRAZKA OD PROCESOW 1-n

        for (int i = 1; i < size; i++) {
            int tempcols, temprows;                                         //liczba kolumn i wierszy
            MPI_Recv(&tempcols, 1, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE); //odebranie liczby kolumn
            MPI_Recv(&temprows, 1, MPI_INT, i, 1, comm, MPI_STATUS_IGNORE); //odebranie liczby wierszy
            Mat tempimg = Mat(temprows, tempcols, CV_8UC3);                 //definicja obrazu wynikowego
            MPI_Recv(tempimg.data, tempcols * temprows * 3, MPI_BYTE, i, 2, comm, MPI_STATUS_IGNORE); // odebranie obrazu wynikowego
            if (i == 1) {                   //jesli to pierwszy kawalek to inicjujemy obraz wyjsciowy
                imgScore = tempimg.clone();
            } else {
                hconcat(imgScore, tempimg, imgScore);   //kolejne obrazki doklejane sa do przygotowanego juz obrazu wynikowego
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
        MPI_Recv(&proccols, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE); //odebranie liczby kolumn
        MPI_Recv(&procrows, 1, MPI_INT, 0, 1, comm, MPI_STATUS_IGNORE); //odebranie liczby wierszy

        Mat procimg = Mat(procrows, proccols, CV_8UC3); //definicja obrazu odebranego
        Mat outimg = Mat(procrows, proccols, CV_8UC3);  //definicja obrazu rozmytego

        MPI_Recv(procimg.data, proccols * procrows * 3, MPI_BYTE, 0, 2, comm, MPI_STATUS_IGNORE); //odebranie kawalka obrazu z proccesu 0
        cout << "obraz odebrany przez proces: " << rank << endl;

//FILTR GAUSSA
        outimg = Gaussianblur(procimg, outimg);

        Mat sendimg = Mat(procrows - 4, proccols - 4, CV_8UC3); // definicja obrazu wyjsciowego
//WYSLANIE DANYCH

        sendimg = outimg(Rect(2, 2, outimg.cols - 4, outimg.rows - 4)).clone(); //okrojenie obrazu wyjsciowego z powielonych krawedzi
        proccols -= 4;
        procrows -= 4;

        MPI_Send(&proccols, 1, MPI_INT, 0, 0, comm);                                //wyslanie liczby kolumn
        MPI_Send(&procrows, 1, MPI_INT, 0, 1, comm);                                //wyslanie liczby wierszy
        MPI_Send(sendimg.data, proccols * procrows * 3, MPI_BYTE, 0, 2, comm);      //wyslanie obrazka


    }
    MPI_Finalize();
    return 0;
}
