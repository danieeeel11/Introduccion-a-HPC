#include "linearregression.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

/*
 * FUNCION DE COSTO
 *
 * Primera funcion: funcion de costo para la regresion lineal
 * Basada en los minimos cuadrados ordinarios demostrado en clase
 * Se necesita entrenar el modelo, lo que implica minimizar alguna
 * funcion de costo, y de esta forma se puede medir la precision
 * de la funcion de hipotesis. Una funcion de costo es la forma de
 * penalizar el modelo por cometer un error. Se implementa una
 * funcion del tipo flotante, que toma como entrada los valores (x,y)
 */
float linearregression::F_OLS_Costo(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::MatrixXd thetas){
    //Eigen::MatrixXd diferencia = pow((X*thetas - Y).array(),2);
    //return (diferencia.sum() / (2*X.rows()));
    Eigen::MatrixXd diferencia = pow((X*thetas - Y).array(),2);
    return (diferencia.sum() / (2*X.rows()));
}

/*
 * FUNCION DE OPTIMIZACION
 *
 * Funcion de gradiente descendiente: En funcion de un ratio de
 * aprendizaje, se avanza hasta encontrar el punto minimo que representa
 * el valor optimo para la funcion
 * Se necesita proveer al programa una funcion para dar al algoritmo un valor inicial para theta,
 * el cual cambiara iterativamente hasat que converja el valor al minimo de nuestra funcion de costo.
 * Basicamente describe el gradiente descendiente: la idea es calcular el gradiente para la funcion
 * de costo que es dado por la derivada parcial de la funcion. La funcion tendra un alpha que representa
 * el salto del gradiente. Las entradas para la funcion seran X, y, theta, alpha y numero de veces que se
 * necesita actualizar theta hasta que la funcion converja
 */
std::tuple<Eigen::VectorXd, std::vector<float>> linearregression::GradientDescent(Eigen::MatrixXd X,
                                                                                  Eigen::MatrixXd Y,
                                                                                  Eigen::MatrixXd thetas,
                                                                                  float alpha,
                                                                                  int num_iter){
    /*Almacenamiento temporal de parametros theta*/
    Eigen::MatrixXd temporal = thetas;
    /*Cantidad de parametros (m) features*/
    int parametros = thetas.rows();
    /*Ubicacion costo inicial, que se actualiza con los nuevos pesos*/
    std::vector<float> costo;
    //En costo ingresaremos los valores de la funcion de costo
    costo.push_back(F_OLS_Costo(X,Y,thetas));
    //Se itera segun el numero de iteraciones y el ratio de aprendizaje para encontrar los valores optimos
    /*
     * Por cada iteracion se calcula la funcion de error que se usa para multiplicar cada deature para obtener
     * el error de cada feature y asi almacenarlo en la variable tem. Se actualiza theta y se calcula el nuevo
     * valor de la funcion de costo basada en el nuevo valor de theta
     */
    for(int i=0; i<num_iter; i++){
        Eigen::MatrixXd error = X*thetas-Y;
        for(int j=0; j<parametros; j++){
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd termino = error.cwiseProduct(X_i);
            temporal(j,0) = thetas(j,0) - ((alpha/X.rows())*termino.sum()); //((alpha/X.rows())*termino.sum());
        }
        thetas = temporal;
        //En costo ingresaremos los valores de la funcion de costo
        costo.push_back(F_OLS_Costo(X,Y,thetas));
    }
    return std::make_tuple(thetas,costo);
}

/*
 * METRICA DE RENDIMIENTO
 *
 * A continuacion se presenta que tan bueno es nuestro proyecto
 * Se procede a crear la metrica de rendimiento:
 * R² score: coeficiente de determinación, es donde el mejor valor posible es 1
 * Como metrica de evaluacion se tiene el R2, que representa una medida de que
 * tan bueno es nuestro modelo
*/
float linearregression::R2_Score(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){
     auto numerador = pow((y-y_hat).array(),2).sum();
     auto denominador = pow(y.array() - y.mean(),2).sum();
     return (1 - (numerador/denominador));
}
