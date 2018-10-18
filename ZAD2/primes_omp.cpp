# include <omp.h>
# include <iostream>
# include <fstream>
# include <cstdlib>
# include <cmath>
# include <vector>


using namespace std;

struct number{		//struktura wykorzystywana w wektorze danych - zawiera informacje o wartosci liczby oraz o tym czy jest pierwsza
   unsigned int value;
    bool prime;
};

int main(int argc, char** argv) {
    ifstream file;  	//plik wejsciowy
    int threadsCount;	//liczba watkow
   unsigned int maxval=0;
    number fromfile;	
    double begin_time, end_time, total_time;	//zmienne wykorzystywane do zliczania czasu dzialania programu
    if (argc != 3) {				//sprawdzenie ilosci argumentow podanych przy wywolaniu programu
        cout << "The number of arguments is invalid"<<endl;
        exit(1);
    }
    threadsCount = atoi(argv[1]);		//rzutowanie char do int
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

    uint i,j;					//zmienne sterujace petla
    uint sqr=sqrt(maxval);			//pierwiastek z liczby maksymalnej
    begin_time = omp_get_wtime();		//rozpoczecie liczenia czasu
	uint d=tab.size();
//CZESC ZASADNICZA PROGRAMU
    #pragma omp parallel for default(shared) private(i,j) num_threads(threadsCount) schedule(runtime)
    for (i=2;i<=sqr;i++) {			//petla zrownoleglana - kolejne liczby od 2 do pierwiastka kwadratowego z najwiekszego elementu zbioru wczytywanego
        for (j = 0; j <d; j++) {		//petla wewnetrzna sprawdzajaca czy kolejne liczby wektora tab dziela sie przez aktualna wartosc zmiennej i
                if((tab[j].value%i==0)&(tab[j].value!=i)) //jesli tak liczba uznawana jest za zlozona, dodatkowo sprawdzamy czy liczba nie jest rowna obecnemu dzielnikowi (zasada pierwszosci)
                    tab[j].prime=false;
        }
    }
    end_time = omp_get_wtime(); // zakonczenie liczenia czasu
    total_time=(end_time-begin_time)*1000; // przeksztalcenie otrzymanego czasu do postaci ms

    cout<<"Time: "<<total_time<<"ms"<<endl;  //wypisanie czasu obliczen



    for (i=0;i<tab.size();i++)				//wypisanie liczb z  wektora tab wraz z informacją czy są pierwsze
        if(tab[i].prime==true)
            cout<<tab[i].value<<": prime"<<endl;
        else
            cout<<tab[i].value<<": composite"<<endl;





}
