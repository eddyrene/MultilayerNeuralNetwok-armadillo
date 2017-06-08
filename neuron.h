#ifndef NEURON_H
#define NEURON_H
#include <armadillo>
#include <iostream>

using namespace std;
using namespace arma;

class Neuron
{
private:
    double val;
public:
    Neuron();
    Neuron(double v);
    void setVal(double v);
    double getVal();
};

#endif // NEURON_H
