#ifndef LAYER_H
#define LAYER_H
#include "neuron.h"
//#include "weight.h"
#include <vector>

class Layer
{
    private:
        vector<Neuron *> * vectNeuron;
        mat * pMat;
        int size;
        int mfil,mcol;
        mat * matError;
        mat * weightBias;
    public:
        Layer();
        Layer(int tam);
        int getSize();
        void sigmod();
        void sigmod2();
        mat * getMat();
        void setMat(mat &m);
        void  update(int a , int b);
        mat * getMatError();
        void setMatError(mat *m);
        mat * getWeightBias();
        void binarizacion();
        vector<Neuron *> * getVectNeuron();
};

#endif // LAYER_H
