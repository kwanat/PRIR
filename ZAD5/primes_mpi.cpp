# include <iostream>
# include <fstream>
# include <cstdlib>
# include <cmath>
# include <vector>
# include "mpi.h"

using namespace std;

MPI_Comm comm;

struct number {        //struktura wykorzystywana w wektorze danych - zawiera informacje o wartosci liczby oraz o tym czy jest pierwsza
    unsigned long int value;
    bool prime;
};

vector <number> primeTesting(vector <number> tab, uint sqr, uint d)         //Funkcja testująca pierwszoć liczb, przyjmuje jako argumenty zbior do tesowania, pierwiastek do ktorego testujemy oraz wielkosc zbioru testowanego
{                                                                          //Funkcja zwraca przetestowany zbior
    uint i, j;
    for (i = 2; i <= sqr; i++) {            //petla zrownoleglana - kolejne liczby od 2 do pierwiastka kwadratowego z najwiekszego elementu zbioru wczytywanego
        for (j = 0; j < d; j++) {        //petla wewnetrzna sprawdzajaca czy kolejne liczby wektora tab dziela sie przez aktualna wartosc zmiennej i
            if ((tab[j].value % i == 0) & (tab[j].value != i)) //jesli tak liczba uznawana jest za zlozona, dodatkowo sprawdzamy czy liczba nie jest rowna obecnemu dzielnikowi (zasada pierwszosci)
                tab[j].prime = false;
        }
    }
    return tab;
}

int main(int argc, char **argv) {
    ifstream file;    //plik wejsciowy
    unsigned int maxval = 0;  //zmienna przechowująca wartosc maksymalna z testowanego pliku
    number fromfile;          // pojedyncza liczba z pliku wraz z informacja o pierwszosci
    int rank, size;     // numer procesu oraz liczba wszystkich procesow

    double begin_time, end_time, total_time; //zmienne wykorzystywane do zliczania czasu dzialania programu

    MPI_Init(&argc, &argv);     //inicjalizacja MPI
    comm = MPI_COMM_WORLD;      //pobranie komunikatora
    MPI_Comm_rank(comm, &rank); //pobranie numeru procesu
    MPI_Comm_size(comm, &size); //pobranie liczby wszystkich procesow
    //przygotowanie nowego typu danych dla MPI, aby mozliwe bylo przesylanie wektora typu number
    int lengths[2] = {1, 1};                                                 // ilosc zmiennych w strukturze
    MPI_Aint offsets[2] = {offsetof(number, value), offsetof(number, prime)};  // przesuniecie bitowe zmiennych struktury
    MPI_Datatype types[2] = {MPI_LONG,MPI::BOOL};                           // podstawowe typy skladajace sie na nasz typ
    MPI_Datatype myType;                                                    // deklaracja nowego typu
    MPI_Type_create_struct(2, lengths, offsets, types, &myType);            // utworzenie nowego typu
    MPI_Type_commit(&myType);


    //PROCES ZEROWY
    if (rank == 0) {
        if (argc != 2) {                //sprawdzenie ilosci argumentow podanych przy wywolaniu programu
            cout << "The number of arguments is invalid" << endl;
            exit(1);
        }
        file.open(argv[1]);
        if (file.fail()) {            //Sprawdzenie poprawnosci otwartego pliku
            cout << "Could not open file to read." << endl;
            exit(1);
        }
        if (size <= 1) { //Zbyt mała liczba procesow
            cout << "Za mala liczba procesow" << endl;
            MPI_Finalize();
            exit(-1);
        }
        vector <number> tab;                //utworzenie wektora liczb

        while (file >> fromfile.value) {        //pobranie danych z pliku do wektora tab
            fromfile.prime = true;            //domniemanie pierwszosci liczby
            tab.push_back(fromfile);        //zapisanie liczby w wektorze tab
            if (fromfile.value > maxval)        //poszukiwanie liczby maksymalnej ze zbioru
                maxval = fromfile.value;

        }


        uint sqr = sqrt(maxval);            //pierwiastek z liczby maksymalnej
        begin_time = MPI_Wtime();        //rozpoczecie liczenia czasu
        uint d = tab.size();          //zmienna pomocnicza rozmiar wektora danych

        int begin, end, step;         // zmienne pomocnicze do manipulacji podzbiorami danych
        begin = 0;                    // początek
        step = d / (size - 1);            // krok
        end = step;                   // koniec


        for (int i = 1; i < size; i++) {   // rozeslanie do procesow podzbiorow liczb pierwszych
            if (i == (size - 1))             // jesli ostatni podzbior to znacznik konca podzbioru rowny jest znacznikowi konca calego zbioru
                end = tab.size();
            vector <number> templateTable(tab.begin() + begin, tab.begin() + end);       // przygotowanie podzbioru danych
            begin = end;                                                                  // zwiekszenie zmiennych pomocniczych
            end += step;

            uint tableSize = templateTable.size();                                        // rozmiar podzbioru danych
            MPI_Send(&tableSize, 1, MPI_INT, i, 0, comm);                               // wyslanie rozmiaru danych
            MPI_Send(&sqr, 1, MPI_INT, i, 1, comm);                                     // wyslanie pierwiastka kwadratowego
            MPI_Send(templateTable.data(), tableSize, myType, i, 2, comm);               // wyslanie podzbioru danych
        }
        vector <number> tab2;         // wektor do odbierania danych
        vector <number> concatTable;  // wektor wynikowy


        for (int i = 1; i < size; i++) {    // ODBIERANIE DANYCH Z PROCESOW

            MPI_Recv(&d, 1, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);                    //odebranie rozmiaru danych
            tab2.resize(d);                                                             //zmiana rozmiaru wektora abydopasować go do rozmiaru danych
            MPI_Recv(tab2.data(), d, myType, i, 1, comm, MPI_STATUS_IGNORE);            // odebranie danych
            concatTable.insert(concatTable.end(), tab2.begin(), tab2.end());            // dolaczenie odebranych danych do wektora wynikowego
        }

        end_time = MPI_Wtime(); // zakonczenie liczenia czasu
        total_time = ((end_time - begin_time) * 1000); // przeksztalcenie otrzymanego czasu do postaci ms
        cout << "Czas: " << total_time << " ms" << endl; //wyświetlanie czasu
        for (uint i = 0; i < concatTable.size(); i++)                //wypisanie liczb z  wektora tab wraz z informacją czy są pierwsze
            if (concatTable[i].prime == true)
                cout << concatTable[i].value << ": prime" << endl;
            else
                cout << concatTable[i].value << ": composite" << endl;
    } else {// KONIEC PROCESU 0
        vector <number> tab3, tab4;      // wektory do odbierania oraz wysylania danych
        uint d, sqr;                    // wielkosc wektora oraz pierwiastek


        MPI_Recv(&d, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);                    // odebranie rozmiaru danych
        MPI_Recv(&sqr, 1, MPI_INT, 0, 1, comm, MPI_STATUS_IGNORE);                  // odebranie pierwiastka
        tab3.resize(d);                                                             // przygotowanie rozmiaru wektora
        MPI_Recv(tab3.data(), d, myType, 0, 2, comm, MPI_STATUS_IGNORE);            // odebranie danych do testowania

        tab4 = primeTesting(tab3, sqr, d);                                              // testowanie pierwszosci liczb

        MPI_Send(&d, 1, MPI_INT, 0, 0, comm);                                       // wyslanie rozmiaru danych wynikowych
        MPI_Send(tab4.data(), d, myType, 0, 1, comm);                                // wyslanie danych wynikowych
    }

    MPI_Type_free(&myType);                                                     // wyrejestrowanie typu 
    MPI_Finalize();
    return 0;

}
