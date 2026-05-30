# Instalación

Para ejecutar este proyecto en tu entorno local, asegúrate de tener **Python 3.8 o superior** instalado y sigue estos pasos:

### 1. Clonar el repositorio
Abre tu terminal y descarga el código fuente:
```bash
git clone [https://github.com/jpimpatapim/hand-tracking-nbody.git](https://github.com/jpimpatapim/hand-tracking-nbody.git)
cd hand-tracking-nbody
```

### 2. Instalar dependencias
Instala las librerías requeridas (OpenCV, MediaPipe y NumPy) usando el gestor de paquetes de Python:
```bash
pip install opencv-python mediapipe numpy
```
### 3. Ejecutar el simulador
Inicia el programa con el siguiente comando:
```bash
python main.py
```
Nota: La primera vez que ejecutes el programa, este descargará automáticamente el modelo de MediaPipe (hand_landmarker.task) de aproximadamente 3MB. Asegúrate de tener conexión a internet.
