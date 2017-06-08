#include "layer.h"

Layer::Layer()
{
       vectNeuron = NULL;
}

Layer::Layer(int tam)
{
    //pWeight= new Weight();
    size=tam;
    vectNeuron = new vector<Neuron *> ();
    //cout<<"tam : "<<tam<<endl;
    vectNeuron->resize(size);
    pMat = new mat(0,0,fill::randu);
    mcol=mfil=0;
    matError = new mat(1,size-1,fill::zeros);
    weightBias = new mat(1,size,fill::zeros);
}

int Layer::getSize()
{
    return size;
}

void Layer::sigmod()
{
    //cout<<"\n se activo \n "<<endl;
    for(int i =0; i<vectNeuron->size();i++ )
    {
        float exp_value;
        float return_value;
        double x = vectNeuron->at(i)->getVal();
        exp_value = exp((double) -x);
        /*** Final sigmoid value ***/
        return_value = 1 / (1 + exp_value);
        vectNeuron->at(i)->setVal(return_value);
        vectNeuron->at(0)->setVal(1);
       // cout<<vectNeuron->at(i)->getVal()<<"  ";
    }
}

void Layer::sigmod2()
{
    //cout<<"\n se activo \n "<<endl;
    for(int i =0; i<vectNeuron->size();i++ )
    {
        float exp_value;
        float return_value;
        double x = vectNeuron->at(i)->getVal();
        exp_value = exp((double) -x);
        /*** Final sigmoid value ***/
        return_value = 1 / (1 + exp_value);
        vectNeuron->at(i)->setVal(return_value);
        vectNeuron->at(0)->setVal(1);
        cout<<vectNeuron->at(i)->getVal()<<"  ";
    }


    cout<<endl;
}
void Layer::binarizacion()
{
    for(int i =0; i<vectNeuron->size();i++ )
    {
        double x = vectNeuron->at(i)->getVal();
        if(x>= 0.5)
            vectNeuron->at(i)->setVal(1);
        else
            vectNeuron->at(i)->setVal(0);
        cout<<vectNeuron->at(i)->getVal()<<"  ";
    }
    cout<<"se binarizo"<<endl;
}


mat * Layer::getMat()
{
    return pMat;
}

void Layer::setMat(mat &m)
{
    cout<<"se deberia guardar"<<endl;
    pMat= &m;
    cout<<&m<<endl;
}

void Layer::update(int a, int b)
{
    mfil=a; mcol=b;
    pMat = new mat(mfil,mcol,fill::randu);
}

mat *Layer::getMatError()
{
    return matError;
}

void Layer::setMatError(mat * m)
{
    matError = m;
}

mat *Layer::getWeightBias()
{
    return weightBias;
}


vector<Neuron *> *Layer::getVectNeuron()
{
    return vectNeuron;
}
