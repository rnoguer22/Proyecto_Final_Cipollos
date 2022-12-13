# <h1 align="center">Proyecto_Final_Cipollos</h1>

<p align="center">https://github.com/rnoguer22/Proyecto_Final_Cipollos.git</p>
<p align="center"><b>Maria Soto Gonzalez & Ruben Nogueras Gonzalez</b></p>

---

<h2 align="center">Librerías</h2>
<p align="center">Estas son las librerias utilizadas para la elaboracion de nuestro proyecto</p>

```python3
import pandas as pd 
import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from statsmodels.api import add_constant, OLS
from statsmodels.formula.api import ols

import pylab as plt
import seaborn as sns

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
````

---

<h2 align="center">Heatmap</h2>
<p align="center">El mapa de calor o heatmap de los datos es el siguiente</p>

![image](https://user-images.githubusercontent.com/115952580/207438942-502ea34f-0333-4d0b-a02d-0d8a3433ce6e.png)

---

<h2 align="center">Regresiones lineales</h2>
<p align="center">Hemos calculado las rectas de regresion, siendo la variable independiente la columna del precio y la variable dependiente las otras columnas del dataframe. El codigo realizado para la representacion de dicha regresion es</p>

```python3
def plot_regression_model(x,y):
    
    x_const = add_constant(x) # add a constant to the model
    
    modelo = OLS(y, x_const).fit() # fit the model
    
    pred = modelo.predict(x_const) # make predictions
    
    print(modelo.summary());
    try:
        const = modelo.params[0] # create a variable with the value of the constant given by the summary
        coef = modelo.params[1] # create a variable with the value of the coef given by the summary

        x_l=np.linspace(x.min(), x.max(), 50) 
        y_l= coef*x_l + const # function of the line

        plt.figure(figsize=(10, 10));

        # plot the line
        plt.plot(x_l, y_l, label=f'{x.name} vs {y.name}={coef}*{x.name}+{const}');

        # data
        plt.scatter(x, y, marker='x', c='g', label=f'{x.name} vs {y.name}');

        plt.title('Regresion lineal')
        plt.xlabel(f'{x.name}')
        plt.ylabel(f'{y.name}')
        plt.legend()
        plt.show();
        return modelo
    except:
        print('No se puede imprimir la recta de regresión para modelos multivariable')
        plt.show();
        return modelo
```

![image](https://user-images.githubusercontent.com/115952580/207439844-b120fe58-06e1-4ebd-99b7-89af96377ba0.png)

---

<h2 align="center">Train y Test de los datos</h2>
<p align="center">Por ultimo hemos hecho el entrenamiento y test de los datos, usando dos modelos de escalado de datos: Standar scaler y Min-Max scaler. Finalmente hemos hecho las predicciones y evaluacion de los datos, siendo el porcentaje de acierto del modelo considerablemente bajo y el error bastante alto. Con esto llegamos a la <b>conclusion</b> de que se trata de un modelo bastante generico, el cual necesita una serie de mejoras para su optimizacion. </p>
