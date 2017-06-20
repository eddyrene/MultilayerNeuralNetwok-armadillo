#ifndef NETWORK_H
#define NETWORK_H
#include "layer.h"
#include <cmath>
#include <cstdio>

class Network
{
private:
    vector<Layer*> * vectLayer;
    int numCapas;
    int numOcultas, numSalidas, numEntradas;
    vector<int> vectOcultas;
    vector<double> VectOrders;
    vector<double> Y;
    vector<mat*> Fweights;

    //vector<vector<double> > Y;
    double threshold;
    double ratioL;
public:
    vector<mat> vectDeltas;
    Network();
    Network(int capas, int entradas, int ocultas,  int salida );
    Network(int capas, int entradas, vector<int> ocultas,  int salida );
    void fill();
    bool isCorrect();
    void init(vector<double> input, vector<double> expected , double err);
    void init2(vector<double> input, double expected , double err);
    mat * derVectNeuron(vector<Neuron *> * v );
    void printVector(string a, vector<double> t);
    void printMat(string a, vector<vector<double> > M);
    void forward();
    void forward2();
    void backpropagation();
    void backpropagationMomentum();
    void backpropagationBatches();
    void bactchUpdate(vector<mat> deltas);
    void createVectDeltas(int tam, int a , int b , int c);
    void createWeights();
    void printAll();
    vector<double> getVectorOrders();
    vector<Layer*> * getVectorLayers();
    mat * vectNeurontoMatrix( vector<Neuron *> * v );
    void matrixtoVectNeuron(mat *c, vector<Neuron *> * v);
    void printWeight();
    double sumSquareError();
    double getThreshold(){return threshold;}
    int getNumEntradas(){return numEntradas;}
    int getNumSalidas(){return numSalidas;}
    bool testSet(vector<double>  I, vector<double> O);
    void loadDataNumbers(string name, int a, vector< vector<double >> &training, vector< vector<double >> &test);
    void loadDataFlowers(string name, int Es, vector< vector<double >> &training, vector< vector<double >> &test);
    void normalize(vector<vector<double>> & A , vector<vector<double>> B);
};

#endif // NETWORK_H
