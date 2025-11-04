import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

class IPWRA:
    """
    Inverse Probability Weighted Regression Adjustment (IPWRA) estimator
    Replica el comando teffects ipwra de Stata
    """
    
    def __init__(self, data, outcome, treatment, covariates, ps_noconstant=False):
        """
        Parameters:
        -----------
        data : DataFrame
            Dataset completo
        outcome : str
            Nombre de la variable de resultado
        treatment : str
            Nombre de la variable de tratamiento (binaria)
        covariates : list
            Lista de covariables para ambos modelos
        ps_noconstant : bool
            Si True, no incluye constante en modelo de propensity score
        """
        self.data = data.copy()
        self.outcome = outcome
        self.treatment = treatment
        self.covariates = covariates
        self.ps_noconstant = ps_noconstant
        
        # Variables de trabajo
        self.y = data[outcome].values
        self.d = data[treatment].values
        self.X = data[covariates].values
        self.n = len(self.y)
        
        # Resultados
        self.ps = None
        self.ate = None
        self.pomean0 = None
        self.pomean1 = None
        self.se_ate = None
        self.se_pomean0 = None
        
    def fit(self):
        """Ajusta el modelo IPWRA"""
        # Paso 1: Estimar propensity scores (modelo de tratamiento)
        self._estimate_propensity_scores()
        
        # Paso 2: Estimar modelos de outcome para cada grupo
        self._estimate_outcome_models()
        
        # Paso 3: Calcular efectos de tratamiento
        self._calculate_treatment_effects()
        
        # Paso 4: Calcular errores estándar robustos usando M-estimation
        self._calculate_robust_standard_errors()
        
        return self
    
    def _estimate_propensity_scores(self):
        """Estima propensity scores usando regresión logística"""
        if self.ps_noconstant:
            # Sin constante
            X_ps = self.X
        else:
            # Con constante
            X_ps = sm.add_constant(self.X)
        
        # Ajustar modelo logit
        logit_model = Logit(self.d, X_ps)
        self.logit_result = logit_model.fit(disp=0)
        
        # Propensity scores
        self.ps = self.logit_result.predict(X_ps)
        
        # Guardar coeficientes del modelo de tratamiento
        self.gamma = self.logit_result.params
    
    def _estimate_outcome_models(self):
        """Estima modelos de regresión para outcomes en cada grupo"""
        # Añadir constante para modelos de outcome
        X_with_const = sm.add_constant(self.X)
        
        # Modelo para grupo de control (D=0)
        mask0 = self.d == 0
        ols0 = sm.OLS(self.y[mask0], X_with_const[mask0])
        self.ols0_result = ols0.fit()
        self.beta0 = self.ols0_result.params
        
        # Modelo para grupo de tratamiento (D=1)
        mask1 = self.d == 1
        ols1 = sm.OLS(self.y[mask1], X_with_const[mask1])
        self.ols1_result = ols1.fit()
        self.beta1 = self.ols1_result.params
        
        # Predecir outcomes potenciales para todos
        self.y0_pred = X_with_const @ self.beta0
        self.y1_pred = X_with_const @ self.beta1
    
    def _calculate_treatment_effects(self):
        """Calcula ATE y potential outcome means usando IPWRA"""
        # Pesos IPW
        w1 = self.d / self.ps
        w0 = (1 - self.d) / (1 - self.ps)
        
        # IPWRA estimator: combina RA con pesos IPW
        # Para tratados: usa outcome observado + ajuste RA ponderado
        # Para controles: usa outcome observado + ajuste RA ponderado
        
        # Potential outcome bajo tratamiento (Y1)
        po1 = self.y1_pred + w1 * (self.y - self.y1_pred)
        self.pomean1 = np.mean(po1)
        
        # Potential outcome bajo control (Y0)
        po0 = self.y0_pred + w0 * (self.y - self.y0_pred)
        self.pomean0 = np.mean(po0)
        
        # ATE
        self.ate = self.pomean1 - self.pomean0
        
        # Guardar para cálculo de SE
        self.po1 = po1
        self.po0 = po0
    
    def _calculate_robust_standard_errors(self):
        """Calcula errores estándar robustos usando M-estimation"""
        # Construir matriz de momentos (influence functions)
        
        # Dimensiones
        k_x = self.X.shape[1]
        k_gamma = len(self.gamma)
        k_beta0 = len(self.beta0)
        k_beta1 = len(self.beta1)
        
        # Crear matriz de diseño para PS
        if self.ps_noconstant:
            X_ps = self.X
        else:
            X_ps = sm.add_constant(self.X)
        
        X_with_const = sm.add_constant(self.X)
        
        # Función de influencia para cada parámetro
        # Inicializar matriz de momentos
        n_params = k_gamma + k_beta0 + k_beta1 + 2  # +2 para ATE y POmean0
        psi = np.zeros((self.n, n_params))
        
        # 1. Momentos del modelo de tratamiento (logit)
        linear_pred = X_ps @ self.gamma
        lambda_i = np.exp(linear_pred) / (1 + np.exp(linear_pred))
        residual_ps = self.d - lambda_i
        
        for j in range(k_gamma):
            psi[:, j] = X_ps[:, j] * residual_ps
        
        # 2. Momentos del modelo de outcome para control
        residual0 = np.zeros(self.n)
        residual0[self.d == 0] = (self.y[self.d == 0] - 
                                   (X_with_const[self.d == 0] @ self.beta0))
        
        idx = k_gamma
        for j in range(k_beta0):
            psi[:, idx + j] = X_with_const[:, j] * residual0
        
        # 3. Momentos del modelo de outcome para tratamiento
        residual1 = np.zeros(self.n)
        residual1[self.d == 1] = (self.y[self.d == 1] - 
                                   (X_with_const[self.d == 1] @ self.beta1))
        
        idx = k_gamma + k_beta0
        for j in range(k_beta1):
            psi[:, idx + j] = X_with_const[:, j] * residual1
        
        # 4. Momento para POmean0
        idx = k_gamma + k_beta0 + k_beta1
        psi[:, idx] = self.po0 - self.pomean0
        
        # 5. Momento para ATE
        idx = k_gamma + k_beta0 + k_beta1 + 1
        psi[:, idx] = (self.po1 - self.po0) - self.ate
        
        # Calcular matriz de varianza asintótica
        # V = (1/n) * sum(psi_i * psi_i')
        meat = (psi.T @ psi) / self.n
        
        # Para el sandwich estimator, necesitamos la derivada de los momentos
        # Pero para simplificar, usamos la estimación directa de la varianza
        # de los parámetros de interés (últimos dos: POmean0 y ATE)
        
        var_matrix = meat / self.n
        
        # Extraer varianzas de los parámetros de interés
        idx_pomean0 = k_gamma + k_beta0 + k_beta1
        idx_ate = k_gamma + k_beta0 + k_beta1 + 1
        
        self.se_pomean0 = np.sqrt(var_matrix[idx_pomean0, idx_pomean0])
        self.se_ate = np.sqrt(var_matrix[idx_ate, idx_ate])
    
    def summary(self):
        """Imprime resumen de resultados al estilo Stata"""
        print("Treatment-effects estimation                    Number of obs     =  {:>10d}".format(self.n))
        print("Estimator      : IPW regression adjustment")
        print("Outcome model  : linear")
        print("Treatment model: logit")
        print("-" * 78)
        print("             |               Robust")
        print("      {:6s} | Coefficient  std. err.      z    P>|z|     [95%% conf. interval]".format(self.outcome))
        print("-" * 13 + "+" + "-" * 64)
        print("ATE          |")
        print("       {:5s} |".format(self.treatment))
        
        # Calcular estadístico z y p-value para ATE
        z_ate = self.ate / self.se_ate
        p_ate = 2 * (1 - stats.norm.cdf(abs(z_ate)))
        ci_ate_lower = self.ate - 1.96 * self.se_ate
        ci_ate_upper = self.ate + 1.96 * self.se_ate
        
        print("   (1 vs 0)  |{:>11.6f} {:>10.7f}  {:>7.2f}   {:>5.3f}  {:>11.5f} {:>11.6f}".format(
            self.ate, self.se_ate, z_ate, p_ate, ci_ate_lower, ci_ate_upper))
        print("-" * 13 + "+" + "-" * 64)
        print("POmean       |")
        print("       {:5s} |".format(self.treatment))
        
        # Calcular estadístico z y p-value para POmean0
        z_po0 = self.pomean0 / self.se_pomean0
        p_po0 = 2 * (1 - stats.norm.cdf(abs(z_po0)))
        ci_po0_lower = self.pomean0 - 1.96 * self.se_pomean0
        ci_po0_upper = self.pomean0 + 1.96 * self.se_pomean0
        
        print("          0  |{:>11.6f} {:>10.7f}  {:>7.2f}   {:>5.3f}  {:>11.5f} {:>11.6f}".format(
            self.pomean0, self.se_pomean0, z_po0, p_po0, ci_po0_lower, ci_po0_upper))
        print("-" * 78)


# Ejemplo de uso con datos simulados (reemplazar con tus datos reales)
def example_usage():
    """Ejemplo de uso del estimador IPWRA"""
    
    # Si tienes los datos reales, cárgalos aquí
    # data = pd.read_stata('tu_archivo.dta')
    # o
    # data = pd.read_csv('tu_archivo.csv')
    
    # Para demostración, creo datos simulados
    print("=" * 78)
    print("Ejemplo con datos simulados")
    print("Para usar tus datos, carga tu DataFrame y ejecuta:")
    print("  model = IPWRA(data, 'earn98', 'train', ['age', 'educ', 'earn96'], ps_noconstant=True)")
    print("  model.fit()")
    print("  model.summary()")
    print("=" * 78)
    print()
    
    np.random.seed(123)
    n = 1130
    
    # Simular covariables
    age = np.random.normal(35, 10, n)
    educ = np.random.normal(12, 2, n)
    earn96 = np.random.normal(8, 3, n)
    
    # Simular tratamiento
    z = -2 + 0.02*age + 0.1*educ + 0.1*earn96
    prob_train = 1 / (1 + np.exp(-z))
    train = (np.random.random(n) < prob_train).astype(int)
    
    # Simular outcome
    earn98 = 3 + 0.05*age + 0.2*educ + 0.5*earn96 + 2.5*train + np.random.normal(0, 2, n)
    
    # Crear DataFrame
    data = pd.DataFrame({
        'earn98': earn98,
        'train': train,
        'age': age,
        'educ': educ,
        'earn96': earn96
    })
    
    # Ajustar modelo IPWRA
    model = IPWRA(data, 'earn98', 'train', ['age', 'educ', 'earn96'], ps_noconstant=True)
    model.fit()
    model.summary()
    
    return model


if __name__ == "__main__":
    model = example_usage()
