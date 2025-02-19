/***********************************************
* Fecha: 8 agosto de 2022
* Autor: Daniel Velásquez
* Tema: Introducción al Algebra Lineal con Eigen
* Materia: Introducción HPC
* **********************************************/

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>

/*Se crean tipos de datos personalizados*/
/*Tipo de dato Matriz 3x3 flotante*/
typedef Eigen::Matrix<float, 3, 3> MiMatriz3x3f;
/*Tipo de dato Vector 3 flotante*/
typedef Eigen::Matrix<float, 3, 1> MiVector3f;
/*Tipo de dato Vector NxN flotante*/
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>MiMatrizd;

int main(){
   //Se declaran las variables de los tipos anteriores
   MiMatriz3x3f mA;
   MiVector3f vB;
   MiMatrizd mDyn(10,15);

   std::cout << "=='Se inicializan las variables anteriores'==" << std::endl;
   //mA en ceros
   mA = MiMatriz3x3f::Zero();
   std::cout<<"\n-mA en ceros \n" << mA <<std::endl;
   //mA Identidad
   mA = MiMatriz3x3f::Identity();
   std::cout<<"\n-mA Identidad \n" << mA<<std::endl;
   //vB inicialializado en numeros aleatorios
   vB = MiVector3f::Random();
   std::cout<<"\n-vB Aleatorio\n" << vB<<std::endl;
   //Se inicializa manualmente la mA
   mA << 1,2,3,4,5,6,7,8,9;
   std::cout <<"\n-mA inicializada manualmente\n" << mA << std::endl;
   //Se cambia el valor indice (1,1) de mA por 10
   mA(1,1) = 10;
   std::cout <<"\n mA (1,1) = 10 \n" << mA << std::endl;

   //Se imprime la Matriz Transpuesta
   std::cout <<"\n mA^T\n " << mA.transpose() <<std::endl;
   //Se imprime de nuevo la matriz mA
   std::cout <<"\n mA\n " << mA <<std::endl;
   //Se sobre escribe sobre matriz mA la transpuesta
   mA.transposeInPlace();
   std::cout <<"\n mA sobre escrita Transpuesta\n " << mA <<std::endl;


   std::cout <<"=='Funcion MAP'==" <<std::endl;
   //Se crea una data int
   int datosInt[]={1,2,3,4};
   //Se mapea en un ]Vector de Fila de enteros (4x1)
   Eigen::Map<Eigen::RowVectorXi> v_map(datosInt, 4);
   std::cout <<"\n Vector Map\n" << v_map << std::endl;

   //Se mapea en una Matriz una data dada en un vector flotantes
   std::vector<float> vdatos = {1,2,3,5,4,3,1,3,4};
   //std::cout << vdatos.data() <<std::endl;
   Eigen::Map<MiMatriz3x3f> M_map(vdatos.data());
   std::cout <<"\n Matrix Map \n" <<M_map <<std::endl;


   std::cout <<"\n =='Aritmetica'==\n" <<M_map <<std::endl;
   //Se crean dos matrices 2x2
   Eigen::Matrix2d a;
   a<< 1, 2, 3, 4;
   Eigen::Matrix2d b;
   b << 0, 1, 1, 0;
   Eigen::Matrix2d suma;
   //Se imprimen las matrices anteriores
   std::cout << "\n Matrix a \n" <<a << std::endl;
   std::cout << "\n Matrix b \n" <<b << std::endl;
   std::cout << " =='Operaciones Element Wise'== "<< std::endl;
   suma = a.array() + b.array();
   std::cout << "\n Matrix Element Wise a+b \n" << suma <<std::endl;
   Eigen::Matrix2d resta;
   resta = a.array() - b.array();
   std::cout << "\n Matrix Element Wise a-b \n" << resta <<std::endl;

   std::cout << "=='Acceso parcial a las estructuras'==" <<std::endl;

   //Se crea una matriz con numeros aleatorios flotante
   //Acepta cualquier cantidad de numeros
   Eigen::MatrixXf matriz_dinamica = Eigen::MatrixXf::Random(4,4);

   std::cout << "\n Matrix 4x4 aleatoria \n" << matriz_dinamica<<std::endl;
   //Se quiere extraer o copiar el bloque central de la matriz antertior

   return 0;
}
