#include "neuron.h"

Neuron::Neuron()
{
    val = 0;
}

Neuron::Neuron(double v)
{
    val=v;
}

void Neuron::setVal(double v)
{
    val=v;
}

double Neuron::getVal()
{
    return val;
}
