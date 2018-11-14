#include "cuda_runtime.h"
#include "device_launch_parameters.h"
# include <iostream>
# include <fstream>
# include <cstdlib>
# include <cmath>
# include <vector>

using namespace std;

struct number{		//struktura wykorzystywana w wektorze danych - zawiera informacje o wartosci liczby oraz o tym czy jest pierwsza
   unsigned long int value;
    bool prime;
};

__global__ void primeTesting (number* tab, uint sqr, uint d)         //Funkcja testująca pierwszoć liczb, przyjmuje jako argumenty wskaźnik na zbior do tesowania, pierwiastek do ktorego testujemy oraz wielkosc zbioru testowanego
{    
uint tid=blockIdx.x;                                                                          //Pobranie numeru bloku w jakim się znajdujemy
uint i,j;                                                                                     //Zmienne pomocnicze
for (i=2;i<=sqr;i++) {			// kolejne liczby od 2 do pierwiastka kwadratowego z najwiekszego elementu zbioru wczytywanego
        for (j = tid; j <d; j+=gridDim.x) {		//petla wewnetrzna sprawdzajaca czy kolejne liczby wektora tab dziela sie przez aktualna wartosc zmiennej i (zmienna j inkrementowana o rozmiar siatki blokow)
                if((tab[j].value%i==0)&&(tab[j].value!=i)) //jesli tak liczba uznawana jest za zlozona, dodatkowo sprawdzamy czy liczba nie jest rowna obecnemu dzielnikowi (zasada pierwszosci)
                    tab[j].prime=false;    
        }
    }                                      
}

int main(int argc, char** argv) {
    int blockNumber=100;      //liczba blokow
    
    ifstream file;  	//plik wejsciowy
    unsigned int maxval=0;  //zmienna przechowująca wartosc maksymalna z testowanego pliku
    number fromfile;	      // pojedyncza liczba z pliku wraz z informacja o pierwszosci
    
    cudaEvent_t start, stop; //deklaracja zmiennych licznika
    float elapsedTime;       //czas wynikowy
    cudaEventCreate(&start); //zdarzenia CUDA
    cudaEventCreate(&stop);

    

    if (argc != 3) {				//sprawdzenie ilosci argumentow podanych przy wywolaniu programu
        cout << "The number of arguments is invalid"<<endl;
        exit(1);
    }
    blockNumber=atoi(argv[1]);
    file.open(argv[2]);
	if (file.fail()){			//Sprawdzenie poprawnosci otwartego pliku
		cout<<"Could not open file to read."<<endl;
		exit(1);
	}

    vector<number> tab;				//utworzenie wektora liczb

    while (file >> fromfile.value) {		//pobranie danych z pliku do wektora tab
        fromfile.prime=true;			//domniemanie pierwszosci liczby
        tab.push_back(fromfile);		//zapisanie liczby w wektorze tab
        if(fromfile.value>maxval)		//poszukiwanie liczby maksymalnej ze zbioru
            maxval=fromfile.value;

    }

   				
    uint sqr=sqrt(maxval);			//pierwiastek z liczby maksymalnej
    uint d=tab.size();          //zmienna pomocnicza rozmiar wektora danych

    number* tab2;               //wskaźnik na typ number
    number* temp = tab.data();  //konwersja wektora do tablicy 
            
    cudaMalloc( (void**)&tab2, d * sizeof(number) );                     //alokacja miejsca w pamięci urządzenia
    cudaMemcpy(tab2, temp, d * sizeof(number), cudaMemcpyHostToDevice);  //kopiowanie danych z pamięci RAM do pamięci GPU
    cudaEventRecord(start);                                              //rozpoczęcie liczenia czasu
    primeTesting <<< blockNumber, 1 >>> (tab2, sqr,d);                   //wywolanie funkcji na urządzeniu 

    cudaEventRecord(stop);                                              //zatrzymanie licznika czasu
    cudaEventSynchronize(stop);                                         //synchronizacja zdarzenia stop

    number * result;                                                            //wskaźnik na typ number do przechowywania wyniku
    result= (number *) malloc (d*sizeof(number));                               //alokacja miejsca
    cudaMemcpy(result, tab2, d * sizeof(number), cudaMemcpyDeviceToHost);       //kopiowanie danych wynikowych z pamięci urzadzenia do pamięci Hosta
    cudaEventElapsedTime(&elapsedTime, start, stop);                            //obliczenie czasu pracy programu
    printf("Czas : %f ms\n", elapsedTime);                                      //wypisanie czasu obliczeń
    

       
    for (uint i=0;i<10;i++)				//wypisanie liczb z  wektora tab wraz z informacją czy są pierwsze
        if(result[i].prime==true)
            cout<< result[i].value<<": prime"<<endl;
        else
            cout<< result[i].value<<": composite"<<endl;   
    cudaFree(tab2);                                                             // zwolnienie pamięci na urządzeniu
    free(result);                                                               // zwolnienie pamięci na hoscie
    return 0;                  

}
