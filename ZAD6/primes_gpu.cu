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

__global__ void primeTesting (number* tab, uint sqr, uint d)         //Funkcja testująca pierwszoć liczb, przyjmuje jako argumenty zbior do tesowania, pierwiastek do ktorego testujemy oraz wielkosc zbioru testowanego
{    
uint tid=blockIdx.x;                                                                          //Funkcja zwraca przetestowany zbior
uint i,j;
for (i=2;i<=sqr;i++) {			//petla zrownoleglana - kolejne liczby od 2 do pierwiastka kwadratowego z najwiekszego elementu zbioru wczytywanego
        for (j = tid; j <d; j+=blockDim.x) {		//petla wewnetrzna sprawdzajaca czy kolejne liczby wektora tab dziela sie przez aktualna wartosc zmiennej i
                if((tab[j].value%i==0)&(tab[j].value!=i)) //jesli tak liczba uznawana jest za zlozona, dodatkowo sprawdzamy czy liczba nie jest rowna obecnemu dzielnikowi (zasada pierwszosci)
                    tab[j].prime=false;    
        }
    }                                      
}

int main(int argc, char** argv) {
    int blockNumber=1;
    
    ifstream file;  	//plik wejsciowy
    unsigned int maxval=0;  //zmienna przechowująca wartosc maksymalna z testowanego pliku
    number fromfile;	      // pojedyncza liczba z pliku wraz z informacja o pierwszosci
    
    cudaEvent_t start, stop; //deklaracja zmiennych licznika
    float elapsedTime; 
    cudaError error;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    

    if (argc != 2) {				//sprawdzenie ilosci argumentow podanych przy wywolaniu programu
        cout << "The number of arguments is invalid"<<endl;
        exit(1);
    }
    file.open(argv[1]);
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

    number* tab2;
    error=cudaMalloc( (void**)&tab2, d * sizeof(number) );
    error = cudaMemcpy(tab2, &tab[0], d * sizeof(number), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    cout<<d<<endl<<sizeof(tab)<<endl;
    //primeTesting <<< blockNumber, 1 >>> (tab2, sqr,d);

    cudaEventRecord(stop); //zatrzymanie licznika i zczytanie czasu obliczen
    error = cudaEventSynchronize(stop);
    vector<number>* result=new vector<number>();
    //vector<number> * result;
    //result= (vector<number> *) malloc (d*sizeof(number));
    
    //vector<number> result;
    result->resize( d );
    error = cudaMemcpy(result, &tab2, d * sizeof(number), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Czas : %f ms\n", elapsedTime);
    
    cout<<result->size()<<endl;
    cout<<sizeof(result)<<endl;
    cout<<sizeof(number)<<endl;
    cout<<endl<<endl<<(*result)[0].value<<endl;
       
    for (uint i=0;i<10;i++)				//wypisanie liczb z  wektora tab wraz z informacją czy są pierwsze
        if((* result)[i].prime==true)
            cout<<(* result)[i].value<<": prime"<<endl;
        else
            cout<<(* result)[i].value<<": composite"<<endl;
                 
           /*          
            for (uint i=0;i<10;i++)				//wypisanie liczb z  wektora tab wraz z informacją czy są pierwsze
        if(tab[i].prime==true)
            cout<<tab[i].value<<": prime"<<endl;
        else
            cout<<tab[i].value<<": composite"<<endl;
	         */          
    return 0;                  

}
