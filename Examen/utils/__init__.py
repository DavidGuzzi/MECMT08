"""
Módulo de utilidades para resolver el Examen de Propiedades de Muestra Finita.

Este módulo contiene funciones auxiliares para:
- Generación de datos (DGPs) de los ejercicios
- Implementación de estimadores econométricos
- Cálculo de métricas de Monte Carlo
- Visualización de resultados
"""

from .data_generation import (
    generate_dgp_exercise1,
    generate_dgp_exercise2,
    generate_dgp_exercise3,
    generate_dgp_exercise4
)

from .estimators import (
    estimate_diff_means,
    estimate_ols_ate,
    estimate_psm,
    estimate_2sls,
    estimate_did
)

from .metrics import (
    calculate_monte_carlo_stats,
    calculate_bias,
    calculate_variance,
    calculate_mse,
    calculate_coverage
)

from .visualization import (
    create_results_table,
    plot_estimates_distribution,
    create_comparison_table
)

__all__ = [
    # Data generation
    'generate_dgp_exercise1',
    'generate_dgp_exercise2',
    'generate_dgp_exercise3',
    'generate_dgp_exercise4',
    # Estimators
    'estimate_diff_means',
    'estimate_ols_ate',
    'estimate_psm',
    'estimate_2sls',
    'estimate_did',
    # Metrics
    'calculate_monte_carlo_stats',
    'calculate_bias',
    'calculate_variance',
    'calculate_mse',
    'calculate_coverage',
    # Visualization
    'create_results_table',
    'plot_estimates_distribution',
    'create_comparison_table'
]
