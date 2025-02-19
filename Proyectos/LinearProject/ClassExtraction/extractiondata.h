#ifndef EXTRACTIONDATA_H
#define EXTRACTIONDATA_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <list>
#include <iostream>
#include <vector>
#include <fstream>

/*
 * Clase extraccion de datos:
 * - Leer un fichero csv
 * - Entrar con argumentos a la clase:
 *      - Lugar del dataset (csv)
 *      - Separador
 *      - Si tiene cabecera o no
 * - Pasar a un vector de vectores del tipo string
 * - Pasar el vector de vector string a Eigen
 * - Promedio
 * - Desviacion
 * - Normalizacion
 * - Metricas
 */

class ExtractionData
{
    /*Argumentos de entrada a la clase*/
    std::string dataset;        //ruta del dataset
    std::string delimitador;    //separador entre datos
    bool header;                //cabecera o no

public:
    //ExtractionData();
    /*Se crea el constructor con los argumentos de entrada*/
    //Vector de vectores del tipo string
    ExtractionData(std::string data,
                   std::string separador,
                   bool cabecera):
        dataset(data), delimitador(separador), header(cabecera){}
    /*Prototipo de metodos y funciones*/
    std::vector<std::vector<std::string>> LeerCSV();
    Eigen::MatrixXd CSVtoEigen(
            std::vector<std::vector<std::string>> dataSet,
            int filas,
            int columnas);
    auto Promedio(Eigen::MatrixXd datos) -> decltype(datos.colwise().mean());
    auto DevStand(Eigen::MatrixXd datos) -> decltype(((datos.array().square().colwise().sum()) / (datos.rows()-1)).sqrt());
    Eigen::MatrixXd Norm(Eigen::MatrixXd datos);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd datos, float size_train);

    void VectortoFile(std::vector<float> vector, std::string file_name);
    void EigentoFile(Eigen::MatrixXd matriz, std::string file_name);
};

#endif // EXTRACTIONDATA_H
