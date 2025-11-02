"""
Script para replicar el análisis de matching de Stata con tus datos reales
Equivalente a: teffects nnmatch (earn98 earn96 age educ) (train), nneighbor(1)
"""

import numpy as np
import pandas as pd
from matching_analysis import NearestNeighborMatching

# ===================================================================
# PASO 1: CARGAR LOS DATOS DE STATA
# ===================================================================

# Cargar el archivo .dta
# IMPORTANTE: Actualiza esta ruta con la ubicación correcta de tu archivo
data_path = r"C:\Users\HP\OneDrive\Escritorio\David Guzzi\DiTella\MEC\Materias\2025\2025 2T\[MT08-MT13] Microeconometría II\Clases\Stata\jobtraining.dta"

try:
    # Cargar datos
    df = pd.read_stata(data_path)
    print("Datos cargados exitosamente")
    print(f"Dimensiones del dataset: {df.shape}")
    print(f"Variables disponibles: {list(df.columns)}")
    print("\n")
    
    # Verificar que las variables necesarias existen
    required_vars = ['train', 'earn98', 'earn96', 'age', 'educ']
    missing_vars = [v for v in required_vars if v not in df.columns]
    
    if missing_vars:
        print(f"ADVERTENCIA: Las siguientes variables no se encontraron: {missing_vars}")
        print("Verifica los nombres de las variables en tu dataset")
    else:
        print("Todas las variables requeridas están presentes")
    
    # ===================================================================
    # PASO 2: PREPARAR LOS DATOS PARA EL ANÁLISIS
    # ===================================================================
    
    # Eliminar valores faltantes
    df_clean = df[required_vars].dropna()
    print(f"\nObservaciones después de eliminar NAs: {len(df_clean)}")
    
    # Extraer las variables
    X = df_clean[['earn96', 'age', 'educ']].values  # Covariables para matching
    y = df_clean['earn98'].values                    # Variable de resultado
    treatment = df_clean['train'].values             # Variable de tratamiento
    
    # Información descriptiva
    print(f"Número de tratados: {sum(treatment == 1)}")
    print(f"Número de controles: {sum(treatment == 0)}")
    
    # ===================================================================
    # PASO 3: REALIZAR EL ANÁLISIS DE MATCHING
    # ===================================================================
    
    print("\n" + "="*78)
    print("REPLICACIÓN DEL ANÁLISIS DE STATA")
    print("="*78 + "\n")
    
    # Crear el modelo de matching con 1 vecino más cercano
    nn_matcher = NearestNeighborMatching(n_neighbors=1)
    
    # Ajustar el modelo
    nn_matcher.fit(X, y, treatment)
    
    # Mostrar resultados en formato Stata
    nn_matcher.summary()
    
    # ===================================================================
    # PASO 4: RESULTADOS ADICIONALES
    # ===================================================================
    
    print("\n" + "="*78)
    print("RESULTADOS ADICIONALES")
    print("="*78)
    print(f"\nEfecto Promedio del Tratamiento (ATE): {nn_matcher.ate:.4f}")
    print(f"Error Estándar Robusto AI: {nn_matcher.se_robust:.4f}")
    print(f"Estadístico z: {nn_matcher.z_stat:.4f}")
    print(f"Valor p: {nn_matcher.p_value:.4f}")
    print(f"Intervalo de Confianza 95%: [{nn_matcher.ci_lower:.4f}, {nn_matcher.ci_upper:.4f}]")
    
    # Interpretación
    print("\n" + "-"*78)
    print("INTERPRETACIÓN:")
    print("-"*78)
    if nn_matcher.p_value < 0.05:
        print(f"✓ El efecto del tratamiento es estadísticamente significativo al 5%")
        if nn_matcher.ate > 0:
            print(f"✓ El programa de entrenamiento aumenta los ingresos en promedio ${nn_matcher.ate:.2f}")
        else:
            print(f"✓ El programa de entrenamiento reduce los ingresos en promedio ${abs(nn_matcher.ate):.2f}")
    else:
        print("✗ El efecto del tratamiento NO es estadísticamente significativo al 5%")
    
    # ===================================================================
    # PASO 5: ANÁLISIS DE SENSIBILIDAD (OPCIONAL)
    # ===================================================================
    
    print("\n" + "="*78)
    print("ANÁLISIS DE SENSIBILIDAD: Variando el número de vecinos")
    print("="*78)
    
    for k in [1, 3, 5]:
        matcher_k = NearestNeighborMatching(n_neighbors=k)
        matcher_k.fit(X, y, treatment)
        print(f"\nk={k} vecinos: ATE={matcher_k.ate:.4f}, SE={matcher_k.se_robust:.4f}, p-valor={matcher_k.p_value:.4f}")
    
    print("\n" + "="*78)
    print("Análisis completado exitosamente")
    print("="*78)
    
except FileNotFoundError:
    print("ERROR: No se pudo encontrar el archivo de datos")
    print(f"Ruta buscada: {data_path}")
    print("\nPor favor:")
    print("1. Verifica que la ruta al archivo sea correcta")
    print("2. Asegúrate de que el archivo jobtraining.dta existe en esa ubicación")
    print("3. Si el archivo está en otra ubicación, actualiza la variable 'data_path'")
    
except Exception as e:
    print(f"ERROR: {str(e)}")
    print("\nSi el error persiste, verifica que tienes instaladas las librerías necesarias:")
    print("pip install pandas numpy scipy")
