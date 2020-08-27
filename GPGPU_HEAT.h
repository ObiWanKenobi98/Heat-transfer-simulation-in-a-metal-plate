#include <vector_types.h>
/*fisier care contine declaratii de structuri/ constante/ prototipuri de functii*/
 /*structura folosita pentru a retine temperatura in kelvin sub forma RGB*/
typedef struct {
    unsigned char R, G, B;
}RGBtemp;

/*dimensiunea placii de metal pentru care se efectueaza simularea*/
#define N 254
#define M 254
/*diferenta maxima absoluta totala tolerata de la un pas la simularii la altul pentru a finaliza calculul*/
#define EPS 30
/*variabile necesare afisarii ferestrei cu valorile termice pe ecran*/
#define WIDTH (N + 2)
#define HEIGHT (M + 2)
/*datele de intrare ale problemei*/
#define ts 10000
#define tr 20
#define x0 64
#define y0 64
#define loadFactor 75
/*structura ce urmeaza sa fie pasata unui thread CPU in vederea calcului
asociat unui bloc de la un anumit pas al simularii*/
typedef struct {
    int rowIndex;
    int columnIndex;
    int rowSizex;
    int columnSizey;
    int numblocksx;
    int numblocksy;
    double a[M + 2][N + 2];
    double b[M + 2][N + 2];
    double diff[M + 2][N + 2];
    RGBtemp RGBtemperature[M + 2][N + 2];
}myCPUdata;
/*referinte necesare compilarii*/
extern "C" void runComputeTemp(double(*pa)[M + 2], double(*pb)[M + 2]);
extern "C" void runComputeDiff(double(*pa)[M + 2], double(*pb)[M + 2], double(*pdiff)[M + 2]);
extern "C" void runComputeRGBfromKelvin(double(*pa)[M + 2], RGBtemp(*prgb)[M + 2]);

#endif
