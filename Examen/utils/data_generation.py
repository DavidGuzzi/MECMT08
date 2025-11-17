"""
Módulo para generar los Data Generating Processes (DGPs) de cada ejercicio.

Cada función genera datos sintéticos siguiendo las especificaciones del examen,
permitiendo realizar simulaciones de Monte Carlo para evaluar propiedades de
muestra finita de distintos estimadores.
"""

import numpy as np
import pandas as pd


def generate_dgp_exercise1(n, seed=None):
    """
    Genera datos para el Ejercicio 1: Diferencia de medias vs. 
    regresión OLS ajustando por covariables.

    DGP con selección sobre observables:
    - X ~ N(0, 1)
    - D = 1 si X + ε > 0, donde ε ~ N(0, 1) (sesgo de selección vía X)
    - Y = 2 + 3*D + 2*X + u, donde u ~ N(0, 1)
    - ATE verdadero = 3

    Parámetros
    ----------
    n : int
        Tamaño de muestra
    seed : int, optional
        Semilla para reproducibilidad

    Retorna
    -------
    dict
        Diccionario con claves:
        - 'Y': Variable de resultado (array)
        - 'D': Indicador de tratamiento (array)
        - 'X': Covariable observable (array)
        - 'true_ate': ATE verdadero (float)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generar covariable observable
    X = np.random.normal(0, 1, n)

    # Generar asignación de tratamiento con sesgo de selección
    epsilon = np.random.normal(0, 1, n)
    D = (X + epsilon > 0).astype(int)

    # Generar variable de resultado
    u = np.random.normal(0, 1, n)
    Y = 2 + 3 * D + 2 * X + u

    return {
        'Y': Y,
        'D': D,
        'X': X,
        'true_ate': 3.0
    }


def generate_dgp_exercise2(n, seed=None):
    """
    Genera datos para el Ejercicio 2: Propensity Score Matching (PSM).

    DGP con independencia condicional:
    - X1 ~ N(0, 1)
    - X2 ~ Bernoulli(0.5)
    - P(D=1|X) = 1 / (1 + exp(-(0.5 + X1 + 2*X2)))
    - D ~ Bernoulli(P(D=1|X))
    - Y = 1 + 4*D + X1 + 3*X2 + u, donde u ~ N(0, 1)
    - ATE verdadero = 4

    Parámetros
    ----------
    n : int
        Tamaño de muestra
    seed : int, optional
        Semilla para reproducibilidad

    Retorna
    -------
    dict
        Diccionario con claves:
        - 'Y': Variable de resultado (array)
        - 'D': Indicador de tratamiento (array)
        - 'X1': Primera covariable (array)
        - 'X2': Segunda covariable (array)
        - 'true_ate': ATE verdadero (float)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generar covariables
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.binomial(1, 0.5, n)

    # Calcular propensity score verdadero
    logit_p = 0.5 + X1 + 2 * X2
    true_ps = 1 / (1 + np.exp(-logit_p))

    # Generar tratamiento según propensity score
    D = np.random.binomial(1, true_ps, n)

    # Generar variable de resultado
    u = np.random.normal(0, 1, n)
    Y = 1 + 4 * D + X1 + 3 * X2 + u

    return {
        'Y': Y,
        'D': D,
        'X1': X1,
        'X2': X2,
        'true_ate': 4.0,
        'true_ps': true_ps
    }


def generate_dgp_exercise3(n, seed=None, weak_instrument=False):
    """
    Genera datos para el Ejercicio 3: Variables Instrumentales (IV).

    DGP con endogeneidad:
    - Z ~ Bernoulli(0.5) (instrumento)
    - X ~ N(0, 1) (covariable)
    - v ~ N(0, 1) (error estructural)
    - D = 0.2 + γ*Z + 0.5*X + v, donde γ = 0.3 (fuerte) o 0.05 (débil)
    - ε ~ N(0, 1) (error idiosincrático)
    - u = 0.8*v + ε (endogeneidad)
    - Y = 5 + 2*D + X + u
    - ATE verdadero = 2

    Parámetros
    ----------
    n : int
        Tamaño de muestra
    seed : int, optional
        Semilla para reproducibilidad
    weak_instrument : bool, default=False
        Si True, usa γ=0.05 (instrumento débil)
        Si False, usa γ=0.3 (instrumento fuerte)

    Retorna
    -------
    dict
        Diccionario con claves:
        - 'Y': Variable de resultado (array)
        - 'D': Variable endógena de tratamiento (array)
        - 'Z': Variable instrumental (array)
        - 'X': Covariable exógena (array)
        - 'true_ate': ATE verdadero (float)
        - 'instrument_strength': Fuerza del instrumento ('strong' o 'weak')
    """
    if seed is not None:
        np.random.seed(seed)

    # Coeficiente del instrumento
    gamma = 0.05 if weak_instrument else 0.3

    # Generar instrumento y covariable
    Z = np.random.binomial(1, 0.5, n)
    X = np.random.normal(0, 1, n)

    # Generar tratamiento endógeno (primera etapa)
    v = np.random.normal(0, 1, n)
    D = 0.2 + gamma * Z + 0.5 * X + v

    # Generar variable de resultado con endogeneidad
    epsilon = np.random.normal(0, 1, n)
    u = 0.8 * v + epsilon  # Correlación entre u y v
    Y = 5 + 2 * D + X + u

    return {
        'Y': Y,
        'D': D,
        'Z': Z,
        'X': X,
        'true_ate': 2.0,
        'instrument_strength': 'weak' if weak_instrument else 'strong'
    }


def generate_dgp_exercise4(n, seed=None, violate_parallel_trends=False):
    """
    Genera datos para el Ejercicio 4: Diferencia-en-Diferencias (DID).

    DGP panel con 2 períodos (pre/post):
    - N unidades, 2 períodos (t=1,2)
    - X_it ~ N(0, 1) (covariable que varía en el tiempo)
    - α_i ~ N(0, 1) (efectos fijos individuales)
    - D_it = 1 si unidad está en grupo tratamiento y t=2
    - Sin violación: λ_t = 0 para todos
    - Con violación: λ_t = 0.5*t para tratados, 0 para control (pre-tendencia)
    - Y_it = α_i + λ_t + 1.5*D_it + 2*X_it + u_it, u_it ~ N(0, 1)
    - ATT verdadero = 1.5

    Parámetros
    ----------
    n : int
        Número de unidades (individuos)
    seed : int, optional
        Semilla para reproducibilidad
    violate_parallel_trends : bool, default=False
        Si True, genera pre-tendencias diferenciales para grupo tratado

    Retorna
    -------
    pd.DataFrame
        Panel data con columnas:
        - 'unit_id': Identificador de unidad
        - 'time': Período (1=pre, 2=post)
        - 'treatment_group': Grupo de tratamiento (1=tratado, 0=control)
        - 'D': Indicador de tratamiento activo (1 si tratado en post)
        - 'X': Covariable que varía en el tiempo
        - 'Y': Variable de resultado
        - 'true_att': ATT verdadero (1.5)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generar efectos fijos individuales
    alpha_i = np.random.normal(0, 1, n)

    # Asignar mitad de unidades a tratamiento
    treatment_group = np.zeros(n)
    treatment_group[:n // 2] = 1

    # Crear panel con 2 períodos
    data = []

    for i in range(n):
        for t in [1, 2]:
            # Covariable que varía en el tiempo
            X_it = np.random.normal(0, 1)

            # Indicador de tratamiento (solo activo en t=2 para grupo tratado)
            D_it = 1 if (treatment_group[i] == 1 and t == 2) else 0

            # Tendencia temporal diferencial (si se viola supuesto)
            if violate_parallel_trends:
                if treatment_group[i] == 1:
                    lambda_t = 0.5 * t  # Pre-tendencia para tratados
                else:
                    lambda_t = 0
            else:
                lambda_t = 0  # Sin tendencias diferenciales

            # Variable de resultado
            u_it = np.random.normal(0, 1)
            Y_it = alpha_i[i] + lambda_t + 1.5 * D_it + 2 * X_it + u_it

            data.append({
                'unit_id': i,
                'time': t,
                'treatment_group': treatment_group[i],
                'D': D_it,
                'X': X_it,
                'Y': Y_it
            })

    df = pd.DataFrame(data)
    df['true_att'] = 1.5

    return df
