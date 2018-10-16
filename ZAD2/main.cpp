# include <omp.h>
# include <iostream>
# include <fstream>
# include <cstdlib>
# include <cmath>
# include <vector>


using namespace std;

struct number{
    int value;
    bool prime;
};

int main(int argc, char** argv) {
    ifstream file;
    int threadsCount;
    int maxval=0;
    number fromfile;
    double begin_time, end_time, total_time;
    if (argc != 3) {
        cout << "Niepoprawna liczba argumentow"<<endl;
        exit(1);
    }
    threadsCount = atoi(argv[1]);
    file.open(argv[2]);
    vector<number> tab;

    while (file >> fromfile.value) {
        fromfile.prime=true;
        tab.push_back(fromfile);
        if(fromfile.value>maxval)
            maxval=fromfile.value;

    }


    //bool *table = new bool[maxval+1];

    //for(int j=0;j<=maxval;j++)
    //    tab[j][0]=true;

    begin_time = omp_get_wtime();

    int i,j;
    int sqr=sqrt(maxval);
    #pragma omp parallel for default(shared) private(i,j) num_threads(threadsCount) schedule(static)
    for (i=2;i<=sqr;i++) {
        for (j = 0; j <tab.size(); j++) {
                if((tab[i].value%i==0)&(tab[i].value!=2))
                    tab[i].prime=false;
        }
    }
    end_time = omp_get_wtime();
    total_time=(end_time-begin_time)*1000;

    cout<<"time: "<<total_time<<endl;


/*
    for (i=0;i<tab.size();i++)
        if(tab[i].prime==true)
            cout<<tab[i].value<<":prime"<<endl;
        else
            cout<<tab[i].value<<":composed"<<endl;


*/


}
