import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class NearestNeighborMatching:
    """
    Implementación del estimador de matching por vecino más cercano
    con distancia de Mahalanobis y errores estándares robustos AI (Abadie-Imbens)
    """
    
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.ate = None
        self.se_robust = None
        self.z_stat = None
        self.p_value = None
        self.ci_lower = None
        self.ci_upper = None
        
    def mahalanobis_distance(self, X_treated, X_control, cov_inv):
        """
        Calcula la distancia de Mahalanobis entre tratados y controles
        """
        distances = np.zeros((X_treated.shape[0], X_control.shape[0]))
        for i in range(X_treated.shape[0]):
            diff = X_control - X_treated[i]
            distances[i] = np.sqrt(np.sum((diff @ cov_inv) * diff, axis=1))
        return distances
    
    def find_matches(self, X_treated, X_control, cov_inv):
        """
        Encuentra los k vecinos más cercanos para cada unidad tratada
        """
        distances = self.mahalanobis_distance(X_treated, X_control, cov_inv)
        matches = []
        match_counts = np.zeros(len(X_control))  # Contador de veces que cada control es usado
        
        for i in range(len(X_treated)):
            # Encontrar los k vecinos más cercanos
            nearest_indices = np.argpartition(distances[i], self.n_neighbors)[:self.n_neighbors]
            matches.append(nearest_indices)
            # Actualizar contador de matches
            for idx in nearest_indices:
                match_counts[idx] += 1
                
        return matches, match_counts
    
    def calculate_ate_with_robust_se(self, y_treated, y_control, X_treated, X_control, 
                                     matches, match_counts):
        """
        Calcula el ATE y los errores estándares robustos de Abadie-Imbens
        """
        n = len(y_treated) + len(y_control)
        n_treated = len(y_treated)
        n_control = len(y_control)
        
        # Calcular el efecto del tratamiento para cada unidad tratada
        treatment_effects = []
        for i, match_indices in enumerate(matches):
            # Promedio de los outcomes de los controles emparejados
            y_matched = y_control[match_indices].mean()
            treatment_effects.append(y_treated[i] - y_matched)
        
        # ATE como promedio de los efectos individuales
        ate = np.mean(treatment_effects)
        
        # Calcular varianza robusta Abadie-Imbens
        # Componente 1: Varianza del efecto del tratamiento
        var_treated = np.var(treatment_effects, ddof=1) / n_treated
        
        # Componente 2: Ajuste por reutilización de controles (matching con reemplazo)
        # Este es el ajuste específico de AI para errores estándares robustos
        var_adjustment = 0
        for i, match_indices in enumerate(matches):
            for j in match_indices:
                # Peso del control j cuando es emparejado con tratado i
                weight = 1 / self.n_neighbors
                # Contribución a la varianza por reutilización
                km_j = match_counts[j]  # Número de veces que control j es usado
                if km_j > 0:
                    var_adjustment += (weight ** 2) * km_j * (y_control[j] - y_control[match_indices].mean()) ** 2
        
        var_adjustment = var_adjustment / (n_treated ** 2)
        
        # Varianza total robusta
        var_robust = var_treated + var_adjustment
        
        # Error estándar robusto
        se_robust = np.sqrt(var_robust)
        
        return ate, se_robust
    
    def fit(self, X, y, treatment):
        """
        Estima el efecto del tratamiento usando matching
        
        Parameters:
        -----------
        X : array-like, covariables para el matching
        y : array-like, variable de resultado
        treatment : array-like, indicador de tratamiento (1=tratado, 0=control)
        """
        # Separar tratados y controles
        treated_mask = treatment == 1
        control_mask = treatment == 0
        
        X_treated = X[treated_mask]
        X_control = X[control_mask]
        y_treated = y[treated_mask]
        y_control = y[control_mask]
        
        # Calcular matriz de covarianza inversa para distancia de Mahalanobis
        # Usar la covarianza pooled (combinada) de tratados y controles
        X_pooled = np.vstack([X_treated, X_control])
        cov_matrix = np.cov(X_pooled.T)
        cov_inv = np.linalg.inv(cov_matrix)
        
        # Encontrar matches
        matches, match_counts = self.find_matches(X_treated, X_control, cov_inv)
        
        # Calcular ATE y errores estándares robustos
        self.ate, self.se_robust = self.calculate_ate_with_robust_se(
            y_treated, y_control, X_treated, X_control, matches, match_counts
        )
        
        # Calcular estadísticos de inferencia
        self.z_stat = self.ate / self.se_robust
        self.p_value = 2 * (1 - stats.norm.cdf(abs(self.z_stat)))
        
        # Intervalo de confianza al 95%
        z_critical = 1.96
        self.ci_lower = self.ate - z_critical * self.se_robust
        self.ci_upper = self.ate + z_critical * self.se_robust
        
        # Información adicional
        self.n_obs = len(treatment)
        self.n_treated = len(y_treated)
        self.n_control = len(y_control)
        
        return self
    
    def summary(self):
        """
        Imprime un resumen de los resultados similar a Stata
        """
        print("="*78)
        print(f"Treatment-effects estimation                   Number of obs      = {self.n_obs:10,}")
        print(f"Estimator      : nearest-neighbor matching     Matches: requested = {self.n_neighbors:10}")
        print("Outcome model  : matching")
        print("Distance metric: Mahalanobis")
        print("-"*78)
        print("             |              AI robust")
        print("      earn98 | Coefficient  std. err.      z    P>|z|     [95% conf. interval]")
        print("-------------+----------------------------------------------------------------")
        print("ATE          |")
        print("       train |")
        print(f"   (1 vs 0)  | {self.ate:11.6f} {self.se_robust:11.7f} {self.z_stat:8.2f} {self.p_value:7.3f}   "
              f"{self.ci_lower:11.6f} {self.ci_upper:11.6f}")
        print("="*78)


# Función principal para replicar el análisis
def replicate_stata_analysis():
    """
    Función para replicar el análisis de Stata
    """
    
    # IMPORTANTE: Necesitarás cargar tus datos aquí
    # Como el archivo .dta no está disponible, creo datos de ejemplo
    # Reemplaza esta sección con la carga real de tus datos:
    
    print("NOTA: Usando datos simulados para demostración.")
    print("Para usar tus datos reales, reemplaza la sección de carga de datos.\n")
    
    # Simulación de datos (REEMPLAZAR CON TUS DATOS REALES)
    np.random.seed(42)
    n = 1130  # Número de observaciones como en tu output
    
    # Simular variables
    train = np.random.binomial(1, 0.3, n)  # Variable de tratamiento
    age = np.random.normal(35, 10, n)
    educ = np.random.normal(12, 3, n)
    earn96 = np.random.normal(15000, 5000, n)
    
    # Simular outcome con efecto de tratamiento
    earn98 = earn96 + 1000 + 1.2 * train * 1000 + np.random.normal(0, 3000, n)
    
    # Para cargar tus datos reales de Stata, descomenta y usa:
    """
    # Opción 1: Si tienes el archivo .dta
    import pandas as pd
    df = pd.read_stata("ruta/a/tu/archivo/jobtraining.dta")
    
    # Extraer las variables necesarias
    train = df['train'].values
    earn98 = df['earn98'].values
    earn96 = df['earn96'].values
    age = df['age'].values
    educ = df['educ'].values
    """
    
    # Preparar los datos para el matching
    # Covariables para el matching (earn96, age, educ)
    X = np.column_stack([earn96, age, educ])
    
    # Variable de resultado
    y = earn98
    
    # Variable de tratamiento
    treatment = train
    
    # Realizar el matching
    print("Estimando efecto del tratamiento con Nearest Neighbor Matching...")
    print("Covariables: earn96, age, educ")
    print("Variable de resultado: earn98")
    print("Variable de tratamiento: train\n")
    
    # Crear y ajustar el modelo
    nn_matcher = NearestNeighborMatching(n_neighbors=1)
    nn_matcher.fit(X, y, treatment)
    
    # Mostrar resultados
    nn_matcher.summary()
    
    return nn_matcher


# Ejecutar el análisis
if __name__ == "__main__":
    # Replicar el análisis de Stata
    results = replicate_stata_analysis()
    
    print("\n" + "="*78)
    print("INSTRUCCIONES PARA USAR CON TUS DATOS:")
    print("-"*78)
    print("1. Instala las librerías necesarias:")
    print("   pip install numpy pandas scipy")
    print("\n2. Para cargar tu archivo .dta de Stata:")
    print("   df = pd.read_stata('ruta/a/jobtraining.dta')")
    print("\n3. Extrae las variables y ejecuta el análisis:")
    print("   X = df[['earn96', 'age', 'educ']].values")
    print("   y = df['earn98'].values")
    print("   treatment = df['train'].values")
    print("   nn_matcher = NearestNeighborMatching(n_neighbors=1)")
    print("   nn_matcher.fit(X, y, treatment)")
    print("   nn_matcher.summary()")
    print("="*78)
