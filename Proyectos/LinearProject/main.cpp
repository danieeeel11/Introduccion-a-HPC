#include "ClassExtraction/extractiondata.h"
#include "Regression/linearregression.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <list>
#include <iostream>
#include <vector>
#include <fstream>


int main(int argc, char* argv[])
{
    std::cout << "##################################################" << std::endl;
    /*Se necesitan 3 argumentos de entrada*/
    ExtractionData Obj_extraccion(argv[1],argv[2],argv[3]);
    std::cout << "Fichero entrada: " << argv[1] << std::endl;


    /*Lectura*/
    /*Se crea un vector de vectores del tipo string para cargar objeto ExData lectura*/
    std::vector<std::vector<std::string>> lec_datos = Obj_extraccion.LeerCSV();
    /*Cantidad de filas y columnas*/
    int filas = lec_datos.size() + 1;
    int columnas = lec_datos[0].size();
    std::cout << "Num filas: " << filas << std::endl;
    std::cout << "Num columnas: " << columnas << std::endl;
    std::cout << "##################################################" << std::endl;


    /*Se crea una matriz Eigen, para ingresar los valores a esa matriz*/
    Eigen::MatrixXd matData = Obj_extraccion.CSVtoEigen(lec_datos, filas, columnas);
    std::cout << "Promedios por columnas" << Obj_extraccion.Promedio(matData) << std::endl;
    std::cout << "Desviaciones STD por columnas" << Obj_extraccion.DevStand(matData) << std::endl;


    /*Se normaliza la matriz de datos*/
    Eigen::MatrixXd normData = Obj_extraccion.Norm(matData);


    /*Separacion grupos: Entrenamiento / Prueba*/
    /*Se divide en datos de entrenamiento y datos de prueba*/
    Eigen::MatrixXd X_train, y_train, X_test, y_test;
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> tupla_datos = Obj_extraccion.TrainTestSplit(normData, 0.8);
    /*Se descomprime nla tupla en cuatro conjuntos*/
    std::tie(X_train, y_train, X_test, y_test) = tupla_datos;
    /*Inspeccion visual de division conjunto de datos train / test*/
    std::cout << "##################################################" << std::endl;
    std::cout << "Conjunto Train" << std::endl;
    std::cout << X_train.cols() << std::endl;
    std::cout << X_train.rows() << std::endl;
    std::cout << y_train.cols() << std::endl;
    std::cout << y_train.rows() << std::endl;
    std::cout << "Conjunto Test" << std::endl;
    std::cout << X_test.cols() << std::endl;
    std::cout << X_test.rows() << std::endl;
    std::cout << y_test.cols() << std::endl;
    std::cout << y_test.rows() << std::endl;
    std::cout << "##################################################" << std::endl;


    //Se instancia la clase de regresion lineal en un objeto
    linearregression modeloLR;


    /*Se crea vectores auxiliares para prueba y entrenamiento, inicializamos en 1*/
    Eigen::VectorXd vector_train = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vector_test = Eigen::VectorXd::Ones(X_test.rows());

    //Se redimensiona la matriz de entrenamiento y de prueba para ser ajustada a los vectores auxiliares anteriores
    //Train: Se redimensiona a una columna adicional
    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    //Train: Se añade a la nueva columna el vector de ceros
    X_train.col(X_train.cols()-1) = vector_train;
    //Test: Se redimendiona a una columna adicional
    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    //Test: Se añade a la nueva columna el vector de ceros
    X_test.col(X_test.cols()-1) = vector_test;


    /*Parametros*/
    //Se crea el vector de coeficientes theta
    Eigen::VectorXd thetas = Eigen::VectorXd::Zero(X_train.cols());
    //Se establece el alpha como ratio de aprendizaje de tipo flotante
    float learning_rate = 0.01; //alpha
    int num_iter = 1000; //num iteraciones
    //Se crea un vector para almacenar las thetas de salida (parametros m y b)
    Eigen::VectorXd thetas_salida;
    //Se crea un vector sencillo (std) de flotantes para almacenar los valores del costo
    std::vector<float> costo;


    /*Optimizacion de parametros*/
    //Se calcula el gradiente descendiente
    std::tuple<Eigen::VectorXd, std::vector<float>> gradiente = modeloLR.GradientDescent(X_train,y_train,
                                                                                      thetas,
                                                                                      learning_rate,
                                                                                      num_iter);
    //Se desempaqueta el gradiente
    std::tie(thetas_salida,costo) = gradiente;


    /*Calculo de Promedio / Desviacion para y_hat
      adicional se desnormalizan los datos para calcular la metrica R2_score*/
    //Se almacenam los valores de thetas y costos en un fichero para posteriormente ser visualizados
    //Se ejecutan una sola vez para generar los archivos
    //ExData.VectortoFile(costo, "costos.txt");
    //ExData.EigentoFile(thetasOut, "thetas.txt");
    //A continuacion se extrae el promedio de la matriz entrada
    auto prom_data = Obj_extraccion.Promedio(matData);
    //Se extraen los valores de la variable independiente
    auto prom_independientes = prom_data(0,11);
    //Se escalan los datos
    auto escalado = matData.rowwise()-matData.colwise().mean();
    //Se extrae la desviacion estandar de los datos escalados
    auto desv_stand = Obj_extraccion.DevStand(escalado);
    //Se extraen los valores de la variable independiente de la devstand
    auto desv_independientes = desv_stand(0,11);


    /*Calculo de valores estimados (predicciones) y_hat,
        se desnormaliza y
        y = mX+b */
    //Se crea una matriz para almacenar los valores estimados de entrenamiento
    Eigen::MatrixXd y_train_hat = (X_train* thetas_salida * desv_independientes).array() + prom_independientes;
    //Matriz para los valores reales de y
    //Valores reales train no normalizados
    Eigen::MatrixXd y = matData.col(11).topRows(1279); //78 o 1279


    //Se revisa que tan bueno fue el modelo a traves de la metrica de rendimiento
    float metrica_R2 = modeloLR.R2_Score(y, y_train_hat);
    std::cout << "##################################################" << std::endl;
    std::cout << "Metrica R2 conjunto entrenamiento: " << metrica_R2 << std::endl;

    return EXIT_SUCCESS;
}

/*
 *     Se necesitan 3 argumentos de entrada
ExtractionData ExData(argv[1],argv[2],argv[3]);
//Se instancia la clase de regresion lineal en un objeto
linearregression modeloLR;
//Se crea un vector de vectores del tipo string para cargar objeto ExData lectura
std::vector<std::vector<std::string>> dataframe = ExData.LeerCSV();

//Cantidad de filas y columnas
int filas = dataframe.size();
int columnas = dataframe[0].size();

//Se crea una matriz Eigen, para ingresar los valores a esa matriz
Eigen::MatrixXd matData = ExData.CSVtoEigen(dataframe, filas, columnas);


//Se normaliza la matriz de datos
Eigen::MatrixXd matNorm = ExData.Norm(matData);

//Se divide en datos de entrenamiento y datos de prueba
Eigen::MatrixXd X_train, y_train, X_test, y_test;
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> div_datos = ExData.TrainTestSplit(matNorm, 0.8);

//Se descomprime nla tupla en cuatro conjuntos
std::tie(X_train, y_train, X_test, y_test) = div_datos;

//Se crea vectores auxiliares para prueba y entrenamiento, inicializamos en 1
Eigen::VectorXd V_train = Eigen::VectorXd::Ones(X_train.rows());
Eigen::VectorXd V_test = Eigen::VectorXd::Ones(X_test.rows());

//Se redimensiona la matriz de entrenamiento y de prueba para ser ajustada a los vectores auxiliares anteriores
X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
X_train.col(X_train.cols()-1) = V_train;
X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
X_test.col(X_test.cols()-1) = V_test;

//Se crea el vector de coeficientes theta
Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());
//Se establece el alpha como ratio de aprendizaje de tipo flotante
float alpha = 0.01;
int num_iter = 1000; //num iteraciones




//Se crea un vector para almacenar las thetas de salida (parametros m y b)
Eigen::VectorXd thetasOut;
//Se crea un vector sencillo (std) de flotantes para almacenar los valores del costo
std::vector<float> costo;
//Se calcula el gradiente descendiente
std::tuple<Eigen::VectorXd, std::vector<float>> g_descendiente = modeloLR.GradientDescent(X_train,y_train,
                                                                                  theta,
                                                                                  alpha,
                                                                                  num_iter);
//Se desempaqueta el gradiente
std::tie(thetasOut,costo) = g_descendiente;
//Se almacenam los valores de thetas y costos en un fichero para posteriormente ser visualizados
//Se ejecutan una sola vez para generar los archivos
//ExData.VectortoFile(costo, "costos.txt");
//ExData.EigentoFile(thetasOut, "thetas.txt");

//A continuacion se extrae el promedio de la matriz entrada
auto prom_data = ExData.Promedio(matData);
//Se extraen los valores de la variable independiente
auto var_prom_independientes = prom_data(0,11);
//Se escalan los datos
auto datos_escalados = matData.rowwise()-matData.colwise().mean();
//Se extrae la desviacion estandar de los datos escalados
auto desv_stand = ExData.DevStand(datos_escalados);
//Se extraen los valores de la variable independiente de la devstand
auto var_desv_independientes = desv_stand(0,11);
//Se crea una matriz para almacenar los valores estimados de entrenamiento
Eigen::MatrixXd y_train_hat = (X_train* thetasOut * var_desv_independientes).array() + var_prom_independientes;
//Matriz para los valores reales de y
Eigen::MatrixXd y = matData.col(11).topRows(78); //1279
//Se revisa que tan bueno fue el modelo a traves de la metrica de rendimiento
float metrica_R2 = modeloLR.R2_Score(y, y_train_hat);
std::cout << "Metrica R2: " << metrica_R2 << std::endl;

return EXIT_SUCCESS;
*/
