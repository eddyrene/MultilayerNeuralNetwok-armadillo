#include "network.h"

Network::Network()
{

}
Network::Network(int capas, int entradas, int ocultas, int salidas)
{
    numCapas= capas;
    numOcultas= ocultas+1;
    numEntradas= entradas+1;
    numSalidas= salidas+1;
    vectLayer = new vector<Layer *>();
    //capa de entrada
    Layer * e = new Layer(numEntradas);
    Layer * l =new Layer(numSalidas);
    vectLayer->push_back(e);
    VectOrders.push_back(numEntradas);
    //capas intermedias
    for(int i =1 ;i<capas-1; i++ )
    {
        Layer * m = new Layer(numOcultas);
        //cout<<"hidden"<<endl;
        vectLayer->push_back(m);
        VectOrders.push_back(numOcultas);
    }
    // capa salida
    vectLayer->push_back(l);
    VectOrders.push_back(numSalidas);
    fill();
    createWeights();
    ratioL=1;
}

Network::Network(int capas, int entradas, vector<int> ocultas, int salidas)
{
    //cout<<"salio"<<endl;
    numCapas= capas;
    //numOcultas= ocultas+1;
    numEntradas= entradas+1;
    numSalidas= salidas+1;
    vectLayer = new vector<Layer *>();
    //capa de entrada
    Layer * e = new Layer(numEntradas);
    Layer * l =new Layer(numSalidas);
    vectLayer->push_back(e);
    VectOrders.push_back(numEntradas);
    //capas intermedias
    vectOcultas=ocultas;
    for(auto a :vectOcultas )
    {
        a++;  cout<<"vectOcultas"<<a<<endl;
    }
    cout<<"tam  "<<vectOcultas.size();
    for(int i =0 ;i<ocultas.size(); i++ )
    {
        Layer * m = new Layer(ocultas[i]);
        vectLayer->push_back(m);
        VectOrders.push_back(ocultas[i]);
    }
    // capa salida
    vectLayer->push_back(l);
    VectOrders.push_back(numSalidas);
    fill();
    createWeights();
    ratioL=1;

}

void Network::fill()
{
     Layer * in= vectLayer->at(0);
     cout<<" s"<<in->getVectNeuron()->size()<<"ca "<< numEntradas<<endl;
     Layer * out= vectLayer->at(numCapas-1);
    for(int i =0; i< numEntradas ; i++)
    {
        Neuron * n = new Neuron(-1);
        in->getVectNeuron()->at(i)= n;
        //cout<<"##"<<in->getVectNeuron()->at(i)->getVal()<<endl;
    }
    for(int i =0; i< numSalidas ; i++)
    {
        Neuron * n = new Neuron(rand() % 2);
        out->getVectNeuron()->at(i)=n;
       // cout<<"**"<<out->getVectNeuron()->at(i)->getVal()<<endl;
    }
    for(int i =1 ; i< numCapas-1 ; i++)
    {
       // cout<<"capa: "<<i<<endl;
        //Layer * hidden = vectLayer->at(i);
        for(int j=0; j < vectOcultas[i-1] ; j++)
        {
            Neuron * n = new Neuron(-1);
            vectLayer->at(i)->getVectNeuron()->at(j)=n;
          //  cout<<"%%%%%"<<vectLayer->at(i)->getVectNeuron()->at(j)->getVal()<<endl;
        }
    }
    Fweights.resize(vectLayer->size()-1);
    for(int i=0 ; i<vectLayer->size(); i++ )
    {
        if(i==vectLayer->size()-1)
            Fweights[i] = new mat(1,VectOrders[i]-1,fill::zeros);
        else
            Fweights[i] = new mat(1,VectOrders[i],fill::zeros);
        cout<<"dim  "<<VectOrders[i]<<endl;
    }

}

void Network::init(vector<double> input, vector<double> expected, double err)
{
    //vectLayer->at(0)->getVectNeuron()->at(0)->setVal(1);
    for(int i =0; i< numEntradas-1 ; i++)
    {
        vectLayer->at(0)->getVectNeuron()->at(i+1)->setVal(input[i]);
    }
    Y.clear();
    for(int i =0; i< numSalidas -1; i++)
    {
        //Y.push_back(expected);
        Y.push_back(expected[i]);
    }
    threshold=err;
}

void Network::init2(vector<double> input, double expected, double err)
{
    //vectLayer->at(0)->getVectNeuron()->at(0)->setVal(1);

    for(int i =0; i< numEntradas-1 ; i++)
    {
        vectLayer->at(0)->getVectNeuron()->at(i+1)->setVal(input[i]);
    }

    Y.clear();
   // Y.push_back(1);
    for(int i =0; i< numSalidas -1; i++)
    {
        Y.push_back(expected);
        //Y.push_back(expected[i]);
    }
    threshold=err;

}
void Network::printVector(string a, vector<double> t)
{
    cout<<a;
    for(auto y : t)
        cout<<" "<<y;
}

void Network::printMat(string a, vector<vector<double> > M)
{
    cout<<a<<endl;
    for(int i =0 ; i< M.size();i++)
    {
        for(int j=0;j<M[i].size();j++)
            cout<<M[i][j]<<" ";
        cout<<endl;
    }
}
void Network::forward()
{
    mat * r;
   // cout<<"numero de capas "<<vectLayer->size()-1<<endl;
    for(int i =0; i< vectLayer->size()-1; i++)
    {
       // cout<<"faefa "<<i<<endl;
        if(i!=0)
        {
            matrixtoVectNeuron(r,vectLayer->at(i)->getVectNeuron());
            vectLayer->at(i)->sigmod();
            delete r;
        }
        mat * neu=  vectNeurontoMatrix( vectLayer->at(i)->getVectNeuron());
        mat * wght =  vectLayer->at(i)->getMat();
        //cout<<"i ones \n"<<*wght<<endl;
       // cout<<"dimensiones A \n"<<neu->n_rows<<" "<<neu->n_cols<<endl;
       // cout<<"dimensiones B \n"<<wght->n_rows<<" "<<wght->n_cols<<endl;
        r = new mat((*neu)*(*wght));
       // cout<<"\nla entrada \n "<<*neu<<"\n los pesos \n "<<*wght<<"\n resultado \n "<<*r<<endl;
       //cout<<"\n los pesos \n "<<*wght<<endl;
        delete neu;

    }
    matrixtoVectNeuron(r,vectLayer->at(vectLayer->size()-1)->getVectNeuron());
    vectLayer->at(vectLayer->size()-1)->sigmod();
    delete r;
}
void Network::forward2()
{
    mat * r;
    for(int i =0; i< vectLayer->size()-1; i++)
    {
        if(i!=0)
        {
            matrixtoVectNeuron(r,vectLayer->at(i)->getVectNeuron());
            vectLayer->at(i)->sigmod();
            // vectLayer->at(i)->binarizacion();
        }
        mat * neu=  vectNeurontoMatrix( vectLayer->at(i)->getVectNeuron());
        mat * wght =  vectLayer->at(i)->getMat();
        //cout<<"i ones \n"<<*wght<<endl;
       // cout<<"dimensiones A \n"<<neu->n_rows<<" "<<neu->n_cols<<endl;
       // cout<<"dimensiones B \n"<<wght->n_rows<<" "<<wght->n_cols<<endl;
        r = new mat((*neu)*(*wght));delete neu;
        //cout<<"\nla entrada \n "<<*neu<<"\n los pesos \n "<<*wght<<"\n resultado \n "<<*r<<endl;
        //cout<<"\n resultado \n"<<*r<<endl;
    }
    matrixtoVectNeuron(r,vectLayer->at(vectLayer->size()-1)->getVectNeuron());
    vectLayer->at(vectLayer->size()-1)->sigmod2();
    //printVector("Vector Final ");
    delete r;

    int pos =vectLayer->size()-1;
    cout<<"\n  imprimiendo salida \n "<<endl;
    for(auto k : *(vectLayer->at(pos)->getVectNeuron()))
    {
        cout<<k->getVal()<<" ";
    }
    //vectLayer->at(vectLayer->size()-1)->binarizacion();
}

void Network::backpropagation()
{
   // cout<<"\n \n backpropagation \n \n"<<endl;
    for(int i =numCapas-1; i>= 0;i--)
    {
        vector<Neuron *> * VN= vectLayer->at(i)->getVectNeuron();
        //E = new mat(1,VN->size());
        //cout<<"i->: "<<i<<endl;
        if(i==numCapas-1 )
        {
            //cout<<"ggggggg"<<endl;
            mat * E = vectLayer->at(i)->getMatError();
            //cout<<"llllllll"<<endl;
            //cout<<"size 1 "<<Y.size()<<endl;
            //cout<<"size 2 "<<VN->size()<<endl;
          //  cout<<"E:  "<< vectLayer->at(i)->getMatError()<<"  "<<*E<<endl;
            //E->at(0)=0;
            for(int j =0 ;j < Y.size(); j++)
            {
              //  cout<<"y "<<Y[j]<<endl;
                //cout<<"s "<<(VN->at(j)->getVal())* VN->at(j)->getVal()*( 1 - VN->at(j)->getVal())<<endl;
                double a1= VN->at(j+1)->getVal();
                //cout<<"a1:      "<<a1<<endl;
                E->at(j)=(Y[j]-a1)*a1*(1-a1);
                //E->at(j)=(Y[j]-a1);
               // cout<<"Y"<<Y[j]<<endl;
            }
          //  cout<<"\n primer error calculado \n "<<*E<<endl;
        }
        else
        {
            //cout<<"else"<<endl;
            mat * E = vectLayer->at(i+1)->getMatError();
         //  cout<<"\n Este error debe coincidir con el anterior \n "<<*E<<endl;
            mat * weight = vectLayer->at(i)->getMat();
         //  cout<<"\n Los pesos actuales \n "<<*weight<<endl;
            mat * X = vectNeurontoMatrix( vectLayer->at(i)->getVectNeuron());
        //   cout<<"\n Las neruronas actuales \n "<<*X<<endl;
            //cout<<"X -- \n"<<(*X)<<" "<<trans(*X)<<" "<<X<<endl;
            //cout<<"E -- \n"<<*E<<" "<<E<<endl;
            //cout<<"W -- \n"<<*weight<<" "<<weight<<endl;
            *(weight) += ratioL*trans(*X)*(*E);delete X;
         //    cout<<"\n Los nuevos pesos \n "<<*weight<<endl;
            if(i!=0)
            {
               //cout<<"\n Los nuevos pesos \n "<<*weight<<endl;
                mat * R = new mat((*E)*(trans(*weight)));
              // cout<<"R --\n"<< *R<<R<<endl;
                mat * D = derVectNeuron(VN);
              //  cout<<"D --\n"<< *D<<D<<endl;
                mat * S  = new mat((*D)%(*R));
              // cout<<"S recortado -- \n"<< *S<<S<<endl;
                E=vectLayer->at(i)->getMatError();
                for(int j = 0 ; j<E->n_cols ; j++)
                    E->at(j)= S->at(j+1);
                delete R;
                delete D;
                delete S;
              // cout<<"\n EL nuevo error: \n "<<*E<<endl;
            }
        }

    }
}


void  Network::backpropagationMomentum()
{
       // cout<<"\n \n backpropagation \n \n"<<endl;
    for(int i =numCapas-1; i>= 0;i--)
    {
        vector<Neuron *> * VN= vectLayer->at(i)->getVectNeuron();
        //E = new mat(1,VN->size());
        //cout<<"i->: "<<i<<endl;
        if(i==numCapas-1)
        {
            //cout<<"ggggggg"<<endl;
            mat * E = vectLayer->at(i)->getMatError();
            //mat * E = Fweights[i];
            //cout<<"llllllll"<<endl;
            //cout<<"size 1 "<<Y.size()<<endl;
            //cout<<"size 2 "<<VN->size()<<endl;
          //  cout<<"E:  "<< vectLayer->at(i)->getMatError()<<"  "<<*E<<endl;
            //E->at(0)=0;
            for(int j =0 ;j < Y.size(); j++)
            {
              //  cout<<"y "<<Y[j]<<endl;
                //cout<<"s "<<(VN->at(j)->getVal())* VN->at(j)->getVal()*( 1 - VN->at(j)->getVal())<<endl;
                double a1= VN->at(j+1)->getVal();
                //cout<<"a1:      "<<a1<<endl;
                E->at(j)=(Y[j]-a1)*a1*(1-a1);
                //E->at(j)=(Y[j]-a1);
               // cout<<"Y"<<Y[j]<<endl;
            }
            *E +=  0.1 * *(Fweights[i]);
            *(Fweights[i])= *(E);
           // cout<<"\n primer error calculado \n "<<*E<<endl;
        }
        else
        {
            //cout<<"else"<<endl;
            mat * E = vectLayer->at(i+1)->getMatError();
         //  cout<<"\n Este error debe coincidir con el anterior \n "<<*E<<endl;
            mat * weight = vectLayer->at(i)->getMat();
         //  cout<<"\n Los pesos actuales \n "<<*weight<<endl;
            mat * X = vectNeurontoMatrix( vectLayer->at(i)->getVectNeuron());
        //   cout<<"\n Las neruronas actuales \n "<<*X<<endl;
            //cout<<"X -- \n"<<(*X)<<" "<<trans(*X)<<" "<<X<<endl;
            //cout<<"E -- \n"<<*E<<" "<<E<<endl;
            //cout<<"W -- \n"<<*weight<<" "<<weight<<endl;
            //int en=0;cin>>en;
        //    cout<<"W --"<<i<<endl<<*(Fweights[i])<<endl<<endl;
                *(weight) += ratioL*trans(*X)*(*E);
                delete X;
            //cout<<"\n Los nuevos pesos \n "<<*weight<<endl;
            if(i!=0)
            {
               //cout<<"\n Los nuevos pesos \n "<<*weight<<endl;
                mat * W = new mat((*E)*(trans(*weight)));
              // cout<<"R --\n"<< *R<<R<<endl;
                mat * Der = derVectNeuron(VN);
              //  cout<<"D --\n"<< *D<<D<<endl;
                mat * Delta  = new mat((*Der)%(*W));
                *Delta +=  0.1 * *(Fweights[i]);
                *(Fweights[i])= *(Delta);
              // cout<<"S recortado -- \n"<< *S<<S<<endl;
                E=vectLayer->at(i)->getMatError();
                for(int j = 0 ; j<E->n_cols ; j++)
                    E->at(j)= Delta->at(j+1);
                delete W;
                delete Der;
                delete Delta;
              // cout<<"\n EL nuevo error: \n "<<*E<<endl;
            }
        }
    }
}

void Network::createWeights()
{
    //Fweights.resize(vectLayer->size()-1);
    cout<<"tamoño de pess"<<Fweights.size()<<endl;
    for(int i=0 ; i<vectLayer->size()-1 ; i++ )
    {
        vectLayer->at(i)->update(VectOrders[i],VectOrders[i+1]-1);
        //Fweights[i] = new mat(1,VectOrders[i],fill::zeros);
    }

}

bool  Network::isCorrect()
{
    int pos =vectLayer->size()-1;
    //vectLayer->at(pos)->sigmod();
   // cout<<"\n  imprimiendo salida \n "<<endl;
    //cout<<"\n";
    int correctos=0;
    for(int j =1; j< vectLayer->at(pos)->getVectNeuron()->size() ; j++)
    {
        //cout<<vectLayer->at(pos)->getVectNeuron()->at(j)->getVal()<<" ";
        if(round((vectLayer->at(pos)->getVectNeuron()->at(j)->getVal()))== Y.at(j-1))
            correctos++;
    }
    //printVector("Esperado",Y);
    //cout<<"correctos x capa "<<correctos<<endl;
    if(correctos==vectLayer->at(pos)->getVectNeuron()->size()-1) return true;
    else
        return false;
}



void Network::printWeight()
{
    for(int i=0 ; i<vectLayer->size()-1 ; i++ )
    {
        cout<<"\n pesitos \n "<<*(vectLayer->at(i)->getMat())<<endl;
    }
}

double Network::sumSquareError()
{
    double r=0;
    for(int i =0;i< Y.size();i++ )
    {
        double Esp= Y.at(i);
        double Last =vectLayer->at(numCapas-1)->getVectNeuron()->at(i+1)->getVal();
        //cout<<"Esperado "<<Y.at(i)<<" Ultima capa "<<vectLayer->at(numCapas-1)->getVectNeuron()->at(i+1)->getVal()<<endl;
        r += pow(Esp-Last,2);
        //r= pow(r,2);
    }
    //printVector("La salida esperada es : \n ",Y);
   return r/2;
}

bool Network::testSet(vector<double>  I, vector<double> O)
{
    for(int i =0; i< numEntradas-1 ; i++)
    {
        vectLayer->at(0)->getVectNeuron()->at(i+1)->setVal(I[i]);
    }
    Y.clear();
    for(int i =0; i< numSalidas -1; i++)
    {
        Y.push_back(O[i]);
    }
    mat * r;
    for(int i =0; i< vectLayer->size()-1; i++)
    {
        if(i!=0)
        {
            matrixtoVectNeuron(r,vectLayer->at(i)->getVectNeuron());
            vectLayer->at(i)->sigmod();
            // vectLayer->at(i)->binarizacion();
        }
        mat * neu=  vectNeurontoMatrix( vectLayer->at(i)->getVectNeuron());
        mat * wght =  vectLayer->at(i)->getMat();
        r = new mat((*neu)*(*wght));delete neu;
       /// cout<<"\nla entrada \n "<<*neu<<"\n los pesos \n "<<*wght<<"\n resultado \n "<<*r<<endl;
    }
    matrixtoVectNeuron(r,vectLayer->at(vectLayer->size()-1)->getVectNeuron());
    int pos =vectLayer->size()-1;
    vectLayer->at(pos)->sigmod();
    cout<<"\n  imprimiendo salida \n "<<endl;
    cout<<"\n";
    int correctos=0;
    for(int j =1; j< vectLayer->at(pos)->getVectNeuron()->size() ; j++)
    {
        cout<<vectLayer->at(pos)->getVectNeuron()->at(j)->getVal()<<" ";
        if(round((vectLayer->at(pos)->getVectNeuron()->at(j)->getVal()))== Y.at(j-1))
            correctos++;
    }
    printVector("Esperado",Y);
    cout<<"correctos x capa "<<correctos<<endl;
    if(correctos==vectLayer->at(pos)->getVectNeuron()->size()-1) return true;
    else
        return false;
    delete r;
}

void Network::loadDataNumbers(string name, int a, vector< vector<double >> &training, vector< vector<double >> &test)
{
    training.resize(a);
    for(int i = 0 ; i< a ;i++)
        training[i].resize(getNumEntradas()-1);

    test.resize(a);
    for(int i = 0 ; i< a ;i++)
        test[i].resize(getNumSalidas()-1);

    /*for(int i=0;i<outputs.size();i++)
        for(int j=0; j<outputs[i].size();j++)
        {
            outputs[i][j]=0;
        }*/
    cout<<"imput  "<<training.size()<<endl;
    cout<<"output  "<<test.size()<<endl;

    ifstream file(name);
    int m=0;
    string line;

    while(getline(file,line) )
    {
        int n=0;
        bool flag= false;
        std::stringstream   linestream(line);
        std::string         value;
        //cout<<"entro"<<endl;
        while(getline(linestream,value,','))
        {
            double v = atoi(value.c_str());
            double vv=v/255;
            //double vv= v;
             if(!flag)
             {
                 test[m][v]=1;
                 flag=true;
             }
             else
             {
                 //cout<<"float "<<v<<endl;
                 training[m][n]=vv;
                 n++;
             }
        }
       // std::cout << "Line Finished" << std::endl;
     m++;
    }
    //my_net->printMat("\n Training: \n", inputs);
    //my_net->printMat("\n Expected: \n", outputs);
    cout<<"terminó de leer con exito"<<endl;
}

void Network::loadDataFlowers(string name, int Es, vector<vector<double> > &training, vector<vector<double> > &test)
{
    training.resize(Es);
    for(int i = 0 ; i< Es ;i++)
        training[i].resize(getNumEntradas()-1);
    test.resize(Es);
    for(int i = 0 ; i< Es ;i++)
        test[i].resize(getNumSalidas()-1);

    cout<<"imput"<<training.size()<<endl;
    cout<<"output"<<test.size()<<endl;
    float e1, e2,e3,e4;
    string s1;
    ifstream file(name);
    int m=0;
    while(!file.eof())
    {
        file>>e1>>e2>>e3>>e4>>s1;
        if(s1=="") break;
        training[m][0]= e1;
        training[m][1]= e2;
        training[m][2]= e3;
        training[m][3]= e4;
        for(int i =0 ;i<s1.size();i++)
        {
            test[m][i]=s1[i]-'0';
        }
        m++;
    }
    printMat("\n Training: \n", training);
    printMat("\n Expected: \n", test);
    cout<<"leyó"<<endl;

}

void Network::normalize(vector<vector<double> > &A, vector<vector<double> > B)
{
    A.resize(B.size());
    for(int i = 0 ; i< B.size() ;i++)
        A[i].resize(B[i].size());

    cout<<A.size()<<" "<<A[5].size()<<endl;
    for(int j=0;j<B[0].size();j++)
    {
        vector<double> V;
        for(int i=0;i<B.size();i++)
        {
            V.push_back(B[i][j]);
        }
        //cout<<"size v"<<V.size()<<endl;
        double max = *max_element(V.begin(),V.end());
        double min = *min_element(V.begin(),V.end());
        //cout<<"max  "<< max <<"min  "<<min<<endl;
        for(int i=0;i<B.size();i++)
        {
            A[i][j]= (B[i][j]-min)/(max-min);
        }
    }
}

void Network::printAll()
{
    cout<<"imprimiendo todo "<<endl;
    for(int i=0 ; i<vectLayer->size(); i++ )
    {
        for(int j=0; j< vectLayer->at(i)->getVectNeuron()->size(); j++)
        {
            cout<<vectLayer->at(i)->getVectNeuron()->at(j)->getVal()<<endl;
        }
        cout<<endl;
    }
}
vector<double> Network::getVectorOrders()
{
    return VectOrders;
}

vector<Layer*> * Network::getVectorLayers()
{
    return vectLayer;
}

mat *Network::vectNeurontoMatrix(vector<Neuron *> *v)
{
    mat * res;
    res= new mat(1,v->size(),fill::zeros);
    //cout<<"zeros \n "<<*res<<endl;
    //cout<<"tamano   "<<res->n_cols<<endl;
    for(int i =0 ; i< v->size(); i++)
    {
        double a = v->at(i)->getVal();
        //cout<< "a" << a <<endl;
        (*res)[i] =a;
        //cout<< "b" << (*res)[i] <<endl;
    }

    /*for(int i =0 ; i< v->size(); i++)
        cout<<(*res)[i]<<" ";*/

    //res->print();
    //cout<<*res<<endl;
    return res;
}
mat *Network::derVectNeuron(vector<Neuron *> *v)
{
    mat * result;
    result= new mat(1,v->size());
    //cout<<"tam del vector a derivar "<<v->size()<<endl;
    for(int i =0 ; i< result->size(); i++)
    {
        double a1 = v->at(i)->getVal();
        result->at(i)= (1-a1)*a1;
    }
    //cout<<"impriendo transformada \n "<<*result<<endl;
    //result
    return result;
}

void Network::matrixtoVectNeuron(mat *c, vector<Neuron *> * v)
{
    //cout<<"rango \n"<<c->n_cols<<"vector"<<v->size()<<endl;
    for(int i =0; i< c->n_cols; i++)
    {
            v->at(i+1)->setVal(c->at((i)));
    }
    //cout<<"valores actualizados"<<endl;
}
