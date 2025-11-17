# MECMT08 - Microeconometría II

Repositorio con materiales del curso de Microeconometría II, Universidad Di Tella, 2025.

## Contenido del Repositorio

### 📁 Directorio `Examen/`

Proyecto de examen que implementa simulaciones Monte Carlo para evaluar las propiedades en muestra finita de distintos estimadores de efectos causales.

**Entregable principal:**
- **`guzzi_03649.pdf`**: Respuestas al examen
  
**Notebook principal:**
- **`guzzi_03649.ipynb`**: Solución completa del examen con 5 ejercicios de simulación Monte Carlo

**Módulo `utils/`:**

El directorio `Examen/utils/` contiene funciones modulares organizadas en 4 archivos Python:

| Archivo | Descripción |
|---------|-------------|
| `estimators.py` | Implementa estimadores econométricos: diferencia de medias, propensity score matching (PSM), variables instrumentales (2SLS), y diferencias en diferencias (DiD) |
| `metrics.py` | Funciones para calcular métricas de evaluación Monte Carlo: sesgo, varianza, MSE, y cobertura de intervalos de confianza |
| `visualization.py` | Genera tablas de resultados y gráficos de distribuciones, convergencia y trade-offs sesgo-varianza |
| `data_generation.py` | Procesos generadores de datos (DGPs) para cada uno de los 5 ejercicios del examen |

## Requisitos

- **Python**: 3.9 o superior
- **Dependencias**: Ver [requirements.txt](requirements.txt)

### Dependencias principales:
- `numpy`: Operaciones con arrays
- `pandas`: Manipulación de datos y lectura de archivos Stata
- `scipy`: Computación científica (estadística, optimización)
- `statsmodels`: Modelos econométricos (OLS, Logit, 2SLS)
- `matplotlib` y `seaborn`: Visualización de datos
- `jupyter`: Ejecución de notebooks (opcional)

## Instalación

### 1. Crear entorno virtual

**En Windows:**
```bash
python -m venv venv
```

**En macOS/Linux:**
```bash
python3 -m venv venv
```

### 2. Activar el entorno virtual

**En Windows:**
```bash
venv\Scripts\activate
```

**En macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

## Cómo Ejecutar los Notebooks

Una vez instaladas las dependencias en el entorno virtual:

### Opción 1: Usando Jupyter Notebook
```bash
# Asegurarse de que el entorno virtual esté activado
jupyter notebook
```

Esto abrirá Jupyter en tu navegador. Navega hasta el notebook que desees ejecutar (por ejemplo, `Examen/guzzi_03649.ipynb`).

### Opción 2: Usando JupyterLab
```bash
# Instalar JupyterLab si no está instalado
pip install jupyterlab

# Lanzar JupyterLab
jupyter lab
```

### Opción 3: Usando VS Code
1. Abrir VS Code en el directorio del proyecto
2. Instalar la extensión "Jupyter" de Microsoft
3. Seleccionar el kernel de Python del entorno virtual (Ctrl+Shift+P > "Python: Select Interpreter" > elegir `venv`)
4. Abrir el notebook y ejecutar las celdas

## Estructura de Archivos

```
MECMT08/
├── Examen/
|   ├── guzzi_03649.pdf            # Respuestas del examen
│   ├── guzzi_03649.ipynb          # Solución del examen
│   └── utils/
│       ├── estimators.py          # Estimadores econométricos
│       ├── metrics.py             # Métricas Monte Carlo
│       ├── visualization.py       # Visualizaciones
│       └── data_generation.py     # Generación de datos
├── requirements.txt               # Dependencias
└── README.md                      # Este archivo
```

## Autor

David Guzzi - Universidad Di Tella, 2025.