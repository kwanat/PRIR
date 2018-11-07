# include <iostream>
# include <fstream>
# include <cstdlib>
# include <cmath>
# include <vector>
# include "mpi.h"

using namespace std;

MPI_Comm comm;

struct number{		//struktura wykorzystywana w wektorze danych - zawiera informacje o wartosci liczby oraz o tym czy jest pierwsza
   unsigned long int value;
    bool prime;
};

MPI_Datatype register_mpi_type(example const&) {
constexpr std::size_t num_members=2;
int lengths[num_members] = {1,1};
MPI_Aint offsets[num_members] = {offsetof(example,x),offsetof(example,y)};
MPI_Datatype types[num_members] = {MPI_LONG,MPI_BOOL};
MPI_Datatype type;
MPI_Type_struct(num_members, lengths, offsets, types, &type);
MPI_Type_commit(&type);
return type;
}

vector<number> primeTesting (vector<number> tab, uint sqr, uint d)
{
uint i,j;
for (i=2;i<=sqr;i++) {			//petla zrownoleglana - kolejne liczby od 2 do pierwiastka kwadratowego z najwiekszego elementu zbioru wczytywanego
        for (j = 0; j <d; j++) {		//petla wewnetrzna sprawdzajaca czy kolejne liczby wektora tab dziela sie przez aktualna wartosc zmiennej i
                if((tab[j].value%i==0)&(tab[j].value!=i)) //jesli tak liczba uznawana jest za zlozona, dodatkowo sprawdzamy czy liczba nie jest rowna obecnemu dzielnikowi (zasada pierwszosci)
                    tab[j].prime=false;
        }
    }
return tab;
}

int main(int argc, char** argv) {
    ifstream file;  	//plik wejsciowy
    unsigned int maxval=0;
    number fromfile;	
    int rank, size;     // numer procesu oraz liczba wszystkich procesow

    double begin_time, end_time, total_time; //zmienne wykorzystywane do zliczania czasu dzialania programu

    MPI_Init(&argc, &argv);     //inicjalizacja MPI
    comm = MPI_COMM_WORLD;      //pobranie komunikatora
    MPI_Comm_rank(comm, &rank); //pobranie numeru procesu
    MPI_Comm_size(comm, &size); //pobranie liczby wszystkich procesow
    
    if(rank==0){
    if (argc != 2) {				//sprawdzenie ilosci argumentow podanych przy wywolaniu programu
        cout << "The number of arguments is invalid"<<endl;
        exit(1);
    }
    file.open(argv[1]);
	if (file.fail()){			//Sprawdzenie poprawnosci otwartego pliku
		cout<<"Could not open file to read."<<endl;
		exit(1);
	}
  if (size <= 1) { //Zbyt mała liczba procesow
            cout << "Za mala liczba procesow" << endl;
            MPI_Finalize();
            exit(-1);
        }
    vector<number> tab;				//utworzenie wektora liczb

    while (file >> fromfile.value) {		//pobranie danych z pliku do wektora tab
        fromfile.prime=true;			//domniemanie pierwszosci liczby
        tab.push_back(fromfile);		//zapisanie liczby w wektorze tab
        if(fromfile.value>maxval)		//poszukiwanie liczby maksymalnej ze zbioru
            maxval=fromfile.value;

    }

   				//zmienne sterujace petla
    uint sqr=sqrt(maxval);			//pierwiastek z liczby maksymalnej
    begin_time = MPI_Wtime();		//rozpoczecie liczenia czasu
    MPI_Datatype type= register_mpi_type(&tab[0]);
    uint d=tab.size();
    MPI_Send(&d, 1, MPI_INT, 1, 0, comm);       // wyslanie liczby wierszy typu int do procesu i
    MPI_Send(&sqr, 1, MPI_INT, 1, 1, comm); 
    MPI_Send(tab.data, d,type, 1, 2, comm);    //wyslanie obrazka jako typu byte

    vector<number> tab2;
    
    MPI_Recv(&d, 1, MPI_INT, 1, 0, comm, MPI_STATUS_IGNORE); //odebranie liczby wierszy
    MPI_Datatype type= register_mpi_type(&tab[0]);
    tab2.resize(d);
    MPI_Recv(tab.data,d , type, 1, 1, comm, MPI_STATUS_IGNORE); // odebranie obrazu wynikowego
    
    
    
    
    
     
    end_time = MPI_Wtime(); // zakonczenie liczenia czasu
    total_time = ((end_time - begin_time) * 1000); // przeksztalcenie otrzymanego czasu do postaci ms
    cout << "Czas: " << total_time << " ms" << endl; //wyświetlanie czasu

   


    for (uint i=0;i<tab2.size();i++)				//wypisanie liczb z  wektora tab wraz z informacją czy są pierwsze
        if(tab2[i].prime==true)
            cout<<tab2[i].value<<": prime"<<endl;
        else
            cout<<tab2[i].value<<": composite"<<endl;
	  }else{// KONIEC PROCESU 0             
    vector<number> tab;
    uint d,sqr;
    
    
    MPI_Recv(&d, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE); //odebranie liczby wierszy
    MPI_Recv(&sqr, 1, MPI_INT, 0, 1, comm, MPI_STATUS_IGNORE);
    MPI_Datatype type= register_mpi_type(&tab[0]);
    tab.resize(d);
    MPI_Recv(tab.data,d , type, 0, 2, comm, MPI_STATUS_IGNORE); // odebranie obrazu wynikowego

    
    
    
    tab=primeTesting(tab,sqr,d); 
    
    MPI_Send(&d, 1, MPI_INT, 0, 0, comm);       // wyslanie liczby wierszy typu int do procesu i
    MPI_Send(tab.data, d,type, 0, 1, comm);    //wyslanie obrazka jako typu byte








    }
                



    MPI_Finalize();
    return 0;                  

}
