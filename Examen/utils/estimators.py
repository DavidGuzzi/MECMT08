"""
Módulo con implementaciones de estimadores econométricos.

Contiene funciones para estimar efectos causales usando diferentes estrategias
de identificación:
- Diferencia en medias
- Regresión OLS
- Propensity Score Matching (PSM)
- Variables Instrumentales (2SLS)
- Diferencia-en-Diferencias (DID)
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import Logit


def estimate_diff_means(y, d):
    """
    Estima el ATE mediante diferencia simple en medias.

    Este es el estimador naive que compara medias entre tratados y controles
    sin ajustar por covariables. Solo es insesgado bajo asignación aleatoria.

    Parámetros
    ----------
    y : array-like
        Variable de resultado
    d : array-like
        Indicador de tratamiento (1=tratado, 0=control)

    Retorna
    -------
    dict
        Diccionario con:
        - 'ate': Estimador de ATE
        - 'se': Error estándar
        - 'ci_lower': Límite inferior IC 95%
        - 'ci_upper': Límite superior IC 95%
    """
    y = np.asarray(y)
    d = np.asarray(d)

    # Separar grupos
    y1 = y[d == 1]
    y0 = y[d == 0]

    n1 = len(y1)
    n0 = len(y0)

    # Diferencia en medias
    ate = y1.mean() - y0.mean()

    # Error estándar asumiendo homocedasticidad
    var1 = y1.var(ddof=1)
    var0 = y0.var(ddof=1)
    se = np.sqrt(var1 / n1 + var0 / n0)

    # Intervalo de confianza al 95%
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se

    return {
        'ate': ate,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def estimate_ols_ate(y, d, x):
    """
    Estima el ATE mediante regresión OLS ajustando por covariables.

    Modelo: Y = β0 + β1*D + β2*X + ε
    Estimador: ATE = β1

    Bajo el supuesto de independencia condicional (CIA), este estimador
    es insesgado si se ajusta por todas las covariables relevantes.

    Parámetros
    ----------
    y : array-like
        Variable de resultado
    d : array-like
        Indicador de tratamiento
    x : array-like
        Covariable(s) observable(s). Puede ser vector o matriz.

    Retorna
    -------
    dict
        Diccionario con:
        - 'ate': Estimador de ATE (coeficiente de D)
        - 'se': Error estándar robusto
        - 'ci_lower': Límite inferior IC 95%
        - 'ci_upper': Límite superior IC 95%
    """
    y = np.asarray(y)
    d = np.asarray(d)
    x = np.asarray(x)

    # Asegurar que x es matriz (incluso si es univariado)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # Construir matriz de diseño: [constante, D, X]
    X_design = np.column_stack([np.ones(len(y)), d, x])

    # Estimar OLS
    model = OLS(y, X_design).fit(cov_type='HC3')  # Errores robustos

    # El coeficiente de D está en posición 1
    ate = model.params[1]
    se = model.bse[1]
    ci_lower = model.conf_int(alpha=0.05)[1, 0]
    ci_upper = model.conf_int(alpha=0.05)[1, 1]

    return {
        'ate': ate,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def estimate_psm(y, d, x1, x2, n_bootstrap=200, seed=None):
    """
    Estima el ATE mediante Propensity Score Matching (PSM).

    Procedimiento:
    1. Estimar propensity score P(D=1|X) usando Logit
    2. Matching nearest-neighbor 1:1 con reemplazo
    3. Calcular ATE como diferencia promedio entre pares matched
    4. Bootstrap para calcular errores estándar

    Parámetros
    ----------
    y : array-like
        Variable de resultado
    d : array-like
        Indicador de tratamiento
    x1 : array-like
        Primera covariable
    x2 : array-like
        Segunda covariable
    n_bootstrap : int, default=200
        Número de réplicas bootstrap
    seed : int, optional
        Semilla para bootstrap

    Retorna
    -------
    dict
        Diccionario con:
        - 'ate': Estimador de ATE
        - 'se': Error estándar bootstrap
        - 'ci_lower': Límite inferior IC 95%
        - 'ci_upper': Límite superior IC 95%
        - 'ps': Propensity scores estimados
    """
    y = np.asarray(y)
    d = np.asarray(d)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    n = len(y)

    # 1. Estimar propensity score con Logit
    X_logit = np.column_stack([np.ones(n), x1, x2])
    try:
        logit_model = Logit(d, X_logit).fit(disp=0, maxiter=100, method='bfgs')
        ps = logit_model.predict(X_logit)
    except:
        # Si falla, usar propensity score simple (proporción de tratados)
        ps = np.ones(n) * d.mean()

    # 2. Función auxiliar para calcular ATE con matching
    def calculate_ate_matching(y_data, d_data, ps_data):
        """Calcula ATE usando nearest-neighbor matching 1:1 con reemplazo."""
        # Separar tratados y controles
        idx_treated = np.where(d_data == 1)[0]
        idx_control = np.where(d_data == 0)[0]

        if len(idx_treated) == 0 or len(idx_control) == 0:
            return np.nan

        ps_treated = ps_data[idx_treated].reshape(-1, 1)
        ps_control = ps_data[idx_control].reshape(-1, 1)

        # Calcular distancias entre propensity scores
        distances = cdist(ps_treated, ps_control, metric='euclidean')

        # Para cada tratado, encontrar el control más cercano
        matched_control_idx = idx_control[distances.argmin(axis=1)]

        # Calcular diferencias
        y_treated = y_data[idx_treated]
        y_matched_control = y_data[matched_control_idx]

        # ATE es el promedio de diferencias
        ate = (y_treated - y_matched_control).mean()

        return ate

    # Calcular ATE principal
    ate = calculate_ate_matching(y, d, ps)

    # 3. Bootstrap para errores estándar
    if seed is not None:
        np.random.seed(seed)

    ate_bootstrap = []
    for _ in range(n_bootstrap):
        # Muestreo con reemplazo
        boot_idx = np.random.choice(n, size=n, replace=True)
        y_boot = y[boot_idx]
        d_boot = d[boot_idx]
        x1_boot = x1[boot_idx]
        x2_boot = x2[boot_idx]

        # Re-estimar propensity score
        X_boot = np.column_stack([np.ones(n), x1_boot, x2_boot])
        try:
            # Verificar que hay variación en tratamiento
            if d_boot.sum() > 0 and d_boot.sum() < n:
                logit_boot = Logit(d_boot, X_boot).fit(disp=0, maxiter=100,
                                                        method='bfgs', warn_convergence=False)
                ps_boot = logit_boot.predict(X_boot)

                # Calcular ATE bootstrap
                ate_b = calculate_ate_matching(y_boot, d_boot, ps_boot)
                if not np.isnan(ate_b):
                    ate_bootstrap.append(ate_b)
        except:
            # Si el modelo no converge, omitir esta réplica
            continue

    # Error estándar y CI bootstrap
    if len(ate_bootstrap) > 0:
        se = np.std(ate_bootstrap, ddof=1)
        ci_lower = np.percentile(ate_bootstrap, 2.5)
        ci_upper = np.percentile(ate_bootstrap, 97.5)
    else:
        se = np.nan
        ci_lower = np.nan
        ci_upper = np.nan

    return {
        'ate': ate,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ps': ps
    }


def estimate_2sls(y, d, z, x):
    """
    Estima el ATE mediante Two-Stage Least Squares (2SLS).

    Primera etapa:  D = π0 + π1*Z + π2*X + v
    Segunda etapa: Y = β0 + β1*D̂ + β2*X + u

    Estimador: ATE = β1

    Parámetros
    ----------
    y : array-like
        Variable de resultado
    d : array-like
        Variable endógena (tratamiento)
    z : array-like
        Instrumento(s)
    x : array-like
        Covariable(s) exógena(s)

    Retorna
    -------
    dict
        Diccionario con:
        - 'ate_ols': ATE usando OLS naive (sesgado)
        - 'ate_2sls': ATE usando 2SLS
        - 'se_2sls': Error estándar 2SLS
        - 'ci_lower': Límite inferior IC 95%
        - 'ci_upper': Límite superior IC 95%
        - 'first_stage_f': Estadístico F de primera etapa
    """
    y = np.asarray(y)
    d = np.asarray(d)
    z = np.asarray(z)
    x = np.asarray(x)
    n = len(y)

    # Asegurar formato matricial
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # OLS naive (para comparación)
    X_ols = np.column_stack([np.ones(n), d, x])
    ols_model = OLS(y, X_ols).fit()
    ate_ols = ols_model.params[1]

    # Primera etapa: D ~ Z + X
    X_first = np.column_stack([np.ones(n), z, x])
    first_stage = OLS(d, X_first).fit()

    # Calcular estadístico F de primera etapa
    # F-test de significancia conjunta de instrumentos
    k_instruments = z.shape[1]
    r_matrix = np.zeros((k_instruments, first_stage.params.shape[0]))
    for i in range(k_instruments):
        r_matrix[i, i + 1] = 1  # Posiciones de los instrumentos (después de constante)

    f_test = first_stage.f_test(r_matrix)
    first_stage_f = f_test.fvalue

    # Valores predichos de D
    d_hat = first_stage.predict(X_first)

    # Segunda etapa: Y ~ D̂ + X
    X_second = np.column_stack([np.ones(n), d_hat, x])
    second_stage = OLS(y, X_second).fit()

    # Nota: Los errores estándar de la segunda etapa no son correctos
    # porque no ajustan por la incertidumbre de la primera etapa.
    # Para errores correctos, usaríamos el estimador manual de 2SLS.

    # Cálculo manual de 2SLS para errores estándar correctos
    # Matriz de instrumentos: [1, Z, X]
    W = X_first
    # Matriz de regresores: [1, D, X]
    X_design = np.column_stack([np.ones(n), d, x])

    # Proyección: X̂ = W(W'W)^-1 W'X
    P_W = W @ np.linalg.inv(W.T @ W) @ W.T
    X_hat = P_W @ X_design

    # Estimador 2SLS: β = (X̂'X)^-1 X̂'y
    beta_2sls = np.linalg.inv(X_hat.T @ X_design) @ X_hat.T @ y

    # Residuos
    residuals = y - X_design @ beta_2sls

    # Varianza de errores
    sigma2 = (residuals ** 2).sum() / (n - X_design.shape[1])

    # Matriz de varianza-covarianza
    vcov = sigma2 * np.linalg.inv(X_hat.T @ X_design)
    se_2sls = np.sqrt(np.diag(vcov))

    ate_2sls = beta_2sls[1]
    se_ate = se_2sls[1]

    # Intervalo de confianza
    ci_lower = ate_2sls - 1.96 * se_ate
    ci_upper = ate_2sls + 1.96 * se_ate

    return {
        'ate_ols': ate_ols,
        'ate_2sls': ate_2sls,
        'se_2sls': se_ate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'first_stage_f': first_stage_f
    }


def estimate_did(df, y_var='Y', d_var='D', time_var='time',
                 treatment_group_var='treatment_group', x_var='X',
                 unit_id_var='unit_id'):
    """
    Estima el ATT mediante Diferencia-en-Diferencias (DID).

    Dos métodos:
    1. DID simple: (Ȳ_treated,post - Ȳ_treated,pre) - (Ȳ_control,post - Ȳ_control,pre)
    2. Regresión DID: Y = α + β*Post + γ*Treated + δ*(Post×Treated) + θ*X + ε
       con errores clustered a nivel de unidad

    ATT = δ

    Parámetros
    ----------
    df : pd.DataFrame
        Panel data con al menos 2 períodos
    y_var : str
        Nombre de columna de resultado
    d_var : str
        Nombre de columna de tratamiento activo
    time_var : str
        Nombre de columna de tiempo
    treatment_group_var : str
        Nombre de columna de grupo de tratamiento
    x_var : str
        Nombre de columna de covariable
    unit_id_var : str
        Nombre de columna de identificador de unidad

    Retorna
    -------
    dict
        Diccionario con:
        - 'att_simple': ATT usando diferencias simples
        - 'att_regression': ATT usando regresión DID
        - 'se': Error estándar clustered
        - 'ci_lower': Límite inferior IC 95%
        - 'ci_upper': Límite superior IC 95%
    """
    # Método 1: DID simple
    treated_pre = df[(df[treatment_group_var] == 1) & (df[time_var] == 1)][y_var].mean()
    treated_post = df[(df[treatment_group_var] == 1) & (df[time_var] == 2)][y_var].mean()
    control_pre = df[(df[treatment_group_var] == 0) & (df[time_var] == 1)][y_var].mean()
    control_post = df[(df[treatment_group_var] == 0) & (df[time_var] == 2)][y_var].mean()

    att_simple = (treated_post - treated_pre) - (control_post - control_pre)

    # Método 2: Regresión DID
    # Crear variables dummy
    df = df.copy()
    df['post'] = (df[time_var] == 2).astype(int)
    df['treated'] = df[treatment_group_var]
    df['post_x_treated'] = df['post'] * df['treated']

    # Construir matriz de diseño
    X_did = sm.add_constant(df[['post', 'treated', 'post_x_treated', x_var]])
    y = df[y_var]

    # Estimar con errores clustered a nivel de unidad
    model = OLS(y, X_did).fit(cov_type='cluster',
                               cov_kwds={'groups': df[unit_id_var]})

    # El coeficiente de interés es post_x_treated
    att_regression = model.params['post_x_treated']
    se = model.bse['post_x_treated']
    ci_lower = model.conf_int(alpha=0.05).loc['post_x_treated', 0]
    ci_upper = model.conf_int(alpha=0.05).loc['post_x_treated', 1]

    return {
        'att_simple': att_simple,
        'att_regression': att_regression,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }
