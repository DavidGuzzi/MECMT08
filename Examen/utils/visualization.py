"""
Módulo para visualización de resultados de simulaciones Monte Carlo.

Contiene funciones para crear tablas formateadas y gráficos que resumen
las propiedades de muestra finita de los estimadores.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_results_table(results_dict, decimals=4):
    """
    Crea una tabla formateada con los resultados de Monte Carlo.

    Parámetros
    ----------
    results_dict : dict
        Diccionario donde las claves son nombres de estimadores y los valores
        son diccionarios con métricas ('bias', 'variance', 'mse', 'coverage')
    decimals : int, default=4
        Número de decimales para redondear

    Retorna
    -------
    pd.DataFrame
        DataFrame con resultados formateados
    """
    rows = []

    for estimator_name, metrics in results_dict.items():
        # Calcular media estimada como: valor_verdadero + sesgo
        true_value = metrics.get('true_value', np.nan)
        bias = metrics.get('bias', np.nan)
        mean_estimate = true_value + bias if not np.isnan(true_value) and not np.isnan(bias) else np.nan

        row = {
            'Estimador': estimator_name,
            'Verdadero': round(true_value, decimals),
            'Media': round(mean_estimate, decimals),
            'Sesgo': round(bias, decimals),
            'Varianza': round(metrics.get('variance', np.nan), decimals),
            'MSE': round(metrics.get('mse', np.nan), decimals),
            'Cobertura': round(metrics.get('coverage', np.nan), decimals),
            'R': int(metrics.get('n_replications', 0))
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ordenar por MSE (mejor a peor)
    df = df.sort_values('MSE', ascending=True)

    return df


def plot_estimates_distribution(estimates_dict, true_value, title='',
                                 bins=50, figsize=(12, 6)):
    """
    Grafica la distribución de estimaciones para múltiples estimadores.

    Parámetros
    ----------
    estimates_dict : dict
        Diccionario donde las claves son nombres de estimadores y los valores
        son arrays con las estimaciones de cada replicación
    true_value : float
        Valor verdadero del parámetro (se marca con línea vertical)
    title : str, optional
        Título del gráfico
    bins : int, default=50
        Número de bins para histogramas
    figsize : tuple, default=(12, 6)
        Tamaño de la figura

    Retorna
    -------
    matplotlib.figure.Figure
        Figura con los histogramas
    """
    n_estimators = len(estimates_dict)

    # Crear subplots
    fig, axes = plt.subplots(1, n_estimators, figsize=figsize, sharey=True)

    if n_estimators == 1:
        axes = [axes]

    for ax, (name, estimates) in zip(axes, estimates_dict.items()):
        # Filtrar NaN
        estimates_clean = estimates[~np.isnan(estimates)]

        # Histograma
        ax.hist(estimates_clean, bins=bins, alpha=0.7, edgecolor='black',
                density=True, label='Distribución')

        # Línea vertical en el valor verdadero
        ax.axvline(true_value, color='red', linestyle='--', linewidth=2,
                   label=f'Valor verdadero = {true_value}')

        # Línea vertical en la media de estimaciones
        mean_estimate = estimates_clean.mean()
        ax.axvline(mean_estimate, color='blue', linestyle='-', linewidth=2,
                   label=f'Media = {mean_estimate:.3f}')

        # Etiquetas
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Estimación', fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel('Densidad', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    return fig


def plot_convergence_by_sample_size(results_by_n, metric='mse',
                                    title='', figsize=(10, 6)):
    """
    Grafica la convergencia de una métrica con respecto al tamaño de muestra.

    Útil para comparar cómo diferentes estimadores mejoran con N más grande.

    Parámetros
    ----------
    results_by_n : dict
        Diccionario anidado:
        {
            n_value: {
                'estimator_name': {'bias': ..., 'mse': ..., ...},
                ...
            },
            ...
        }
    metric : str, default='mse'
        Métrica a graficar ('bias', 'variance', 'mse', 'coverage')
    title : str, optional
        Título del gráfico
    figsize : tuple, default=(10, 6)
        Tamaño de la figura

    Retorna
    -------
    matplotlib.figure.Figure
        Figura con las curvas de convergencia
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extraer nombres de estimadores (asumiendo mismos estimadores para todos los N)
    first_n = list(results_by_n.keys())[0]
    estimator_names = list(results_by_n[first_n].keys())

    # Para cada estimador, graficar métrica vs N
    for estimator_name in estimator_names:
        n_values = []
        metric_values = []

        for n, estimators_results in sorted(results_by_n.items()):
            if estimator_name in estimators_results:
                n_values.append(n)
                metric_values.append(estimators_results[estimator_name].get(metric, np.nan))

        # Graficar
        ax.plot(n_values, metric_values, marker='o', linewidth=2,
                markersize=8, label=estimator_name)

    # Configuración
    ax.set_xlabel('Tamaño de muestra (N)', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(title or f'Convergencia de {metric.upper()} por tamaño de muestra',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_bias_variance_tradeoff(results_dict, title='', figsize=(8, 6)):
    """
    Grafica el trade-off entre sesgo y varianza para diferentes estimadores.

    Cada estimador se representa como un punto en el espacio (Sesgo², Varianza).
    El MSE es la distancia al origen.

    Parámetros
    ----------
    results_dict : dict
        Diccionario donde las claves son nombres de estimadores y los valores
        son diccionarios con métricas
    title : str, optional
        Título del gráfico
    figsize : tuple, default=(8, 6)
        Tamaño de la figura

    Retorna
    -------
    matplotlib.figure.Figure
        Figura con el scatter plot
    """
    fig, ax = plt.subplots(figsize=figsize)

    for estimator_name, metrics in results_dict.items():
        bias = metrics.get('bias', np.nan)
        variance = metrics.get('variance', np.nan)
        mse = metrics.get('mse', np.nan)

        # Graficar punto
        ax.scatter(bias**2, variance, s=200, alpha=0.7, label=estimator_name)

        # Anotar con MSE
        ax.annotate(f'MSE={mse:.4f}',
                    xy=(bias**2, variance),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)

    # Línea de 45 grados (donde Sesgo² = Varianza)
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Sesgo² = Varianza')

    # Configuración
    ax.set_xlabel('Sesgo²', fontsize=12)
    ax.set_ylabel('Varianza', fontsize=12)
    ax.set_title(title or 'Trade-off Sesgo-Varianza',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def create_comparison_table(exercises_results, decimals=4):
    """
    Crea una tabla comparativa de resultados de múltiples ejercicios.

    Parámetros
    ----------
    exercises_results : dict
        Diccionario anidado:
        {
            'Ejercicio 1': {
                'Estimador A': {'bias': ..., 'mse': ...},
                'Estimador B': {...}
            },
            'Ejercicio 2': {...}
        }
    decimals : int, default=4
        Número de decimales

    Retorna
    -------
    pd.DataFrame
        DataFrame consolidado con todos los resultados
    """
    rows = []

    for exercise_name, estimators_dict in exercises_results.items():
        for estimator_name, metrics in estimators_dict.items():
            # Calcular media estimada
            true_value = metrics.get('true_value', np.nan)
            bias = metrics.get('bias', np.nan)
            mean_estimate = true_value + bias if not np.isnan(true_value) and not np.isnan(bias) else np.nan

            row = {
                'Ejercicio': exercise_name,
                'Estimador': estimator_name,
                'Verdadero': round(true_value, decimals),
                'Media': round(mean_estimate, decimals),
                'Sesgo': round(bias, decimals),
                'Varianza': round(metrics.get('variance', np.nan), decimals),
                'MSE': round(metrics.get('mse', np.nan), decimals),
                'Cobertura': round(metrics.get('coverage', np.nan), decimals)
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    return df


def print_summary_statistics(estimates, estimator_name='Estimador'):
    """
    Imprime estadísticas descriptivas de las estimaciones.

    Parámetros
    ----------
    estimates : array-like
        Vector de estimaciones
    estimator_name : str
        Nombre del estimador

    Retorna
    -------
    None
        Imprime estadísticas en consola
    """
    estimates_clean = estimates[~np.isnan(estimates)]

    print(f"\n{'='*60}")
    print(f"Estadísticas descriptivas: {estimator_name}")
    print(f"{'='*60}")
    print(f"N replicaciones válidas: {len(estimates_clean)}")
    print(f"Media:                   {estimates_clean.mean():.6f}")
    print(f"Mediana:                 {np.median(estimates_clean):.6f}")
    print(f"Desviación estándar:     {estimates_clean.std(ddof=1):.6f}")
    print(f"Mínimo:                  {estimates_clean.min():.6f}")
    print(f"Máximo:                  {estimates_clean.max():.6f}")
    print(f"Percentil 2.5%:          {np.percentile(estimates_clean, 2.5):.6f}")
    print(f"Percentil 97.5%:         {np.percentile(estimates_clean, 97.5):.6f}")
    print(f"{'='*60}\n")
