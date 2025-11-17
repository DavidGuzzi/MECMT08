"""
Módulo para calcular métricas de Monte Carlo.

Contiene funciones para evaluar propiedades de muestra finita de estimadores:
- Sesgo
- Varianza
- Error Medio Cuadrático (MSE)
- Cobertura de intervalos de confianza
"""

import numpy as np
from scipy import stats


def calculate_bias(estimates, true_value):
    """
    Calcula el sesgo de un estimador.

    Sesgo = E[θ̂] - θ = (1/R) Σ θ̂_r - θ

    Donde R es el número de replicaciones Monte Carlo.

    Parámetros
    ----------
    estimates : array-like
        Vector de estimaciones de las replicaciones Monte Carlo
    true_value : float
        Valor verdadero del parámetro

    Retorna
    -------
    float
        Sesgo del estimador
    """
    estimates = np.asarray(estimates)
    # Filtrar NaN si existen
    estimates_clean = estimates[~np.isnan(estimates)]

    if len(estimates_clean) == 0:
        return np.nan

    bias = estimates_clean.mean() - true_value
    return bias


def calculate_variance(estimates):
    """
    Calcula la varianza de un estimador.

    Var(θ̂) = (1/(R-1)) Σ (θ̂_r - θ̂̄)²

    Parámetros
    ----------
    estimates : array-like
        Vector de estimaciones de las replicaciones Monte Carlo

    Retorna
    -------
    float
        Varianza del estimador
    """
    estimates = np.asarray(estimates)
    # Filtrar NaN
    estimates_clean = estimates[~np.isnan(estimates)]

    if len(estimates_clean) <= 1:
        return np.nan

    variance = estimates_clean.var(ddof=1)
    return variance


def calculate_mse(estimates, true_value):
    """
    Calcula el Error Medio Cuadrático (MSE) de un estimador.

    MSE = Sesgo² + Varianza

    El MSE mide la precisión total del estimador combinando sesgo y varianza.

    Parámetros
    ----------
    estimates : array-like
        Vector de estimaciones de las replicaciones Monte Carlo
    true_value : float
        Valor verdadero del parámetro

    Retorna
    -------
    float
        MSE del estimador
    """
    estimates = np.asarray(estimates)
    # Filtrar NaN
    estimates_clean = estimates[~np.isnan(estimates)]

    if len(estimates_clean) == 0:
        return np.nan

    # Método directo: E[(θ̂ - θ)²]
    mse = ((estimates_clean - true_value) ** 2).mean()

    return mse


def calculate_coverage(estimates, standard_errors, true_value, alpha=0.05):
    """
    Calcula la tasa de cobertura de intervalos de confianza.

    La cobertura es la proporción de intervalos de confianza al nivel (1-α)
    que contienen el valor verdadero del parámetro.

    Para un IC bien calibrado, la cobertura debería ser cercana a (1-α).

    Parámetros
    ----------
    estimates : array-like
        Vector de estimaciones puntuales de las replicaciones
    standard_errors : array-like
        Vector de errores estándar correspondientes
    true_value : float
        Valor verdadero del parámetro
    alpha : float, default=0.05
        Nivel de significancia (0.05 para IC 95%)

    Retorna
    -------
    float
        Tasa de cobertura (entre 0 y 1)
    """
    estimates = np.asarray(estimates)
    standard_errors = np.asarray(standard_errors)

    # Filtrar pares donde ambos valores son válidos
    valid_mask = ~(np.isnan(estimates) | np.isnan(standard_errors))
    estimates_clean = estimates[valid_mask]
    se_clean = standard_errors[valid_mask]

    if len(estimates_clean) == 0:
        return np.nan

    # Calcular límites de IC usando distribución normal
    z_critical = stats.norm.ppf(1 - alpha/2)  # Valor crítico para nivel (1-α)
    ci_lower = estimates_clean - z_critical * se_clean
    ci_upper = estimates_clean + z_critical * se_clean

    # Verificar si el valor verdadero está dentro del IC
    coverage_indicator = (ci_lower <= true_value) & (true_value <= ci_upper)

    # Proporción de ICs que cubren el valor verdadero
    coverage_rate = coverage_indicator.mean()

    return coverage_rate


def calculate_monte_carlo_stats(estimates, standard_errors, true_value, alpha=0.05):
    """
    Calcula todas las métricas de Monte Carlo para un estimador.

    Esta función consolida el cálculo de sesgo, varianza, MSE y cobertura
    en una sola llamada.

    Parámetros
    ----------
    estimates : array-like
        Vector de estimaciones puntuales de las replicaciones
    standard_errors : array-like
        Vector de errores estándar correspondientes
    true_value : float
        Valor verdadero del parámetro
    alpha : float, default=0.05
        Nivel de significancia para cobertura

    Retorna
    -------
    dict
        Diccionario con claves:
        - 'bias': Sesgo del estimador
        - 'variance': Varianza del estimador
        - 'mse': Error Medio Cuadrático
        - 'coverage': Tasa de cobertura de IC
        - 'n_replications': Número de replicaciones válidas
        - 'true_value': Valor verdadero (para referencia)
    """
    estimates = np.asarray(estimates)
    standard_errors = np.asarray(standard_errors)

    # Contar replicaciones válidas
    n_valid = np.sum(~np.isnan(estimates))

    # Calcular métricas
    bias = calculate_bias(estimates, true_value)
    variance = calculate_variance(estimates)
    mse = calculate_mse(estimates, true_value)
    coverage = calculate_coverage(estimates, standard_errors, true_value, alpha)

    return {
        'bias': bias,
        'variance': variance,
        'mse': mse,
        'coverage': coverage,
        'n_replications': n_valid,
        'true_value': true_value
    }


def summarize_results(results_dict):
    """
    Resume resultados de múltiples estimadores en formato tabular.

    Parámetros
    ----------
    results_dict : dict
        Diccionario donde las claves son nombres de estimadores y los valores
        son diccionarios con métricas de Monte Carlo

    Retorna
    -------
    dict
        Diccionario con formato organizado para visualización
    """
    summary = {}

    for estimator_name, metrics in results_dict.items():
        summary[estimator_name] = {
            'Sesgo': metrics.get('bias', np.nan),
            'Varianza': metrics.get('variance', np.nan),
            'MSE': metrics.get('mse', np.nan),
            'Cobertura': metrics.get('coverage', np.nan),
            'N': metrics.get('n_replications', 0)
        }

    return summary
