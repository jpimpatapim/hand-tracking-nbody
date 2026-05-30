import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import math
import time

# ==============================================================================
# --- CONFIGURACIÓN VISUAL ---
# ==============================================================================
GRID_RES = 45

COLOR_HUD       = (0, 255, 255)
COLOR_GRID_BASE = (255, 255, 0)
COLOR_GRAVITY   = (0, 0, 255)
COLOR_FIJADA    = (0, 255, 0)
COLOR_BORRADO   = (0, 0, 255)     
COLOR_DIVISOR   = (100, 100, 100) 
COLOR_CAMARA    = (255, 0, 255)   # Magenta
COLOR_COOLDOWN  = (150, 150, 150) 
# ==============================================================================

os.environ["QT_QPA_PLATFORM"] = "xcb"

# 1. IA Modelo
ruta_modelo = "hand_landmarker.task"
if not os.path.exists(ruta_modelo):
    print("Descargando el modelo de IA...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task", ruta_modelo)

opciones = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=ruta_modelo),
    running_mode=mp.tasks.vision.RunningMode.VIDEO, 
    num_hands=2, min_hand_detection_confidence=0.5
)

def interpolar_color(color1, color2, factor):
    b = int(color1[0] + (color2[0] - color1[0]) * factor)
    g = int(color1[1] + (color2[1] - color1[1]) * factor)
    r = int(color1[2] + (color2[2] - color1[2]) * factor)
    return (b, g, r)

def renderizar_espacio_multimasa(masas_fijas, masa_activa, W, H, rot_x, rot_z, zoom):
    x = np.linspace(-1.5, 1.5, GRID_RES)
    y = np.linspace(-1.5, 1.5, GRID_RES)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    
    todas_las_masas = masas_fijas.copy()
    if masa_activa is not None:
        todas_las_masas.append(masa_activa)
    
    for masa in todas_las_masas:
        (cx, cy), tamano, densidad = masa
        dist_sq = (X - cx)**2 + (Y - cy)**2
        Z += -densidad * np.exp(-dist_sq / max(0.01, tamano))
    
    puntos_3d = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    
    sx, cx_a = np.sin(rot_x), np.cos(rot_x)
    Mx = np.array([[1, 0, 0], [0, cx_a, -sx], [0, sx, cx_a]])
    sz, cz = np.sin(rot_z), np.cos(rot_z)
    Mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    M_rot = Mx @ Mz
    puntos_rotados = puntos_3d @ M_rot.T
    
    distancia_camara = 3.0
    factor_perspectiva = distancia_camara / (distancia_camara - puntos_rotados[:, 2])
    px = puntos_rotados[:, 0] * factor_perspectiva
    py = puntos_rotados[:, 1] * factor_perspectiva
    
    escala_visual = H * 0.5 * zoom
    
    proyectados_px = np.stack([
        (W / 2 + px * escala_visual).astype(np.int32),
        (H / 2 + py * escala_visual).astype(np.int32)
    ], axis=-1)
    
    proyectados_grid = proyectados_px.reshape(GRID_RES, GRID_RES, 2)
    Z_grid = Z.reshape(GRID_RES, GRID_RES)
    
    lienzo = np.zeros((H, W, 3), dtype=np.uint8)
    
    max_z = np.min(Z_grid)
    max_z = min(max_z, -0.1) 
    
    for i in range(GRID_RES):
        for j in range(GRID_RES):
            pt_orig = proyectados_grid[i, j]
            if j < GRID_RES - 1:
                pt_dest = proyectados_grid[i, j+1]
                factor = min(1.0, Z_grid[i, j] / max_z) if max_z != 0 else 0
                c = interpolar_color(COLOR_GRID_BASE, COLOR_GRAVITY, factor)
                cv2.line(lienzo, tuple(pt_orig), tuple(pt_dest), c, 1)
            
            if i < GRID_RES - 1:
                pt_dest = proyectados_grid[i+1, j]
                factor = min(1.0, Z_grid[i, j] / max_z) if max_z != 0 else 0
                c = interpolar_color(COLOR_GRID_BASE, COLOR_GRAVITY, factor)
                cv2.line(lienzo, tuple(pt_orig), tuple(pt_dest), c, 1)
                
    return lienzo

with mp.tasks.vision.HandLandmarker.create_from_options(opciones) as detector:
    camara = cv2.VideoCapture(0) 
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tiempo_ms = 0
    
    masas_fijas = []
    tiempo_ultimo_drop = 0
    tiempo_ultimo_borrado = 0 
    
    origen_gravedad = (0.0, 0.0)
    tamano_masa = 0.5
    densidad_masa = 0.5
    
    modo_camara = False
    inicio_mano_abierta = None
    tiempo_ultimo_toggle = 0
    
    angulo_camara_x = math.pi * 0.35 
    angulo_camara_z = math.pi * 0.25 
    zoom_camara = 1.0 
    
    print("¡Sistema listo!")
    
    while camara.isOpened():
        exito, frame_real = camara.read()
        if not exito: break
            
        frame_real = cv2.flip(frame_real, 1)
        H, W, _ = frame_real.shape
        frame_rgb = cv2.cvtColor(frame_real, cv2.COLOR_BGR2RGB)
        tiempo_ms += 30 
        resultados = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), tiempo_ms)
        
        pos_der_detectada = False
        num_manos_detectadas = len(resultados.hand_landmarks) if resultados.hand_landmarks else 0

        color_zona = COLOR_CAMARA if modo_camara else COLOR_DIVISOR
        cv2.line(frame_real, (W//2, 0), (W//2, H), color_zona, 2)
        
        texto_der = "CAMARA: Mover=Rotar | Dedos=Zoom" if modo_camara else "MOVIMIENTO (Mano Der)"
        cv2.putText(frame_real, "PROPIEDADES (Mano Izq)", (20, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_zona, 2)
        cv2.putText(frame_real, texto_der, (W//2 + 20, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_zona, 2)

        if resultados.hand_landmarks:
            for indice, puntos_mano in enumerate(resultados.hand_landmarks):
                etiqueta = resultados.handedness[indice][0].category_name
                puntos_px = [(int(p.x * W), int(p.y * H)) for p in puntos_mano]
                
                # --- MANO IZQUIERDA ---
                if etiqueta == "Right":
                    dist_8_12  = math.hypot(puntos_mano[8].x - puntos_mano[12].x, puntos_mano[8].y - puntos_mano[12].y)
                    dist_12_16 = math.hypot(puntos_mano[12].x - puntos_mano[16].x, puntos_mano[12].y - puntos_mano[16].y)
                    dist_16_20 = math.hypot(puntos_mano[16].x - puntos_mano[20].x, puntos_mano[16].y - puntos_mano[20].y)
                    dist_8_0   = math.hypot(puntos_mano[8].x - puntos_mano[0].x, puntos_mano[8].y - puntos_mano[0].y)
                    
                    mano_bien_abierta = (dist_8_12 > 0.04 and dist_12_16 > 0.04 and dist_16_20 > 0.04 and dist_8_0 > 0.15)
                    
                    tiempo_objetivo = 1.0 
                    
                    cumple_regla_manos = True
                    if not modo_camara and num_manos_detectadas > 1:
                        cumple_regla_manos = False

                    if mano_bien_abierta:
                        if cumple_regla_manos:
                            if inicio_mano_abierta is None:
                                inicio_mano_abierta = time.time()
                            else:
                                tiempo_abierta = time.time() - inicio_mano_abierta
                                progreso = min(100, int((tiempo_abierta / tiempo_objetivo) * 100))
                                
                                mensaje = "SALIENDO: " if modo_camara else "ENTRANDO A CAMARA: "
                                cv2.putText(frame_real, f"{mensaje}{progreso}%", (puntos_px[9][0] - 80, puntos_px[9][1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CAMARA, 2)
                                
                                if tiempo_abierta > tiempo_objetivo and (time.time() - tiempo_ultimo_toggle) > 1.5:
                                    modo_camara = not modo_camara 
                                    tiempo_ultimo_toggle = time.time()
                                    inicio_mano_abierta = None
                        else:
                            inicio_mano_abierta = None
                            cv2.putText(frame_real, "BAJA LA MANO DERECHA PARA CAMBIAR", (puntos_px[9][0] - 140, puntos_px[9][1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        inicio_mano_abierta = None 
                    
                    if not modo_camara:
                        dist_indice = math.hypot(puntos_mano[8].x - puntos_mano[4].x, puntos_mano[8].y - puntos_mano[4].y)
                        tamano_masa = max(0.05, min(dist_indice * 2.0, 1.5)) 
                        
                        dist_menique = math.hypot(puntos_mano[20].x - puntos_mano[4].x, puntos_mano[20].y - puntos_mano[4].y)
                        densidad_masa = max(0.0, min(dist_menique * 3.0, 2.5)) 
                        
                        cv2.line(frame_real, puntos_px[4], puntos_px[8], (0, 255, 0), 2)
                        cv2.line(frame_real, puntos_px[4], puntos_px[20], (0, 0, 255), 2)
                    else:
                        cv2.circle(frame_real, puntos_px[9], 15, COLOR_CAMARA, 2)
                        cv2.putText(frame_real, "PAUSADO", (puntos_px[9][0] - 40, puntos_px[9][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CAMARA, 2)
                    
                # --- MANO DERECHA ---
                elif etiqueta == "Left":
                    pos_der_detectada = True
                    
                    if modo_camara:
                        dx = (puntos_mano[9].x - 0.75) * 3.0 
                        dy = (puntos_mano[9].y - 0.5) * 2.5  
                        
                        angulo_camara_z = math.pi * 0.25 + (dx * math.pi) 
                        angulo_camara_x = max(0.1, min(math.pi * 0.5, math.pi * 0.35 + (dy * math.pi * 0.5))) 
                        
                        dist_zoom = math.hypot(puntos_mano[8].x - puntos_mano[4].x, puntos_mano[8].y - puntos_mano[4].y)
                        zoom_camara = max(0.4, min(dist_zoom * 10.0, 3.0))
                        
                        cv2.circle(frame_real, puntos_px[9], 10, COLOR_CAMARA, -1)
                        cv2.line(frame_real, puntos_px[4], puntos_px[8], COLOR_CAMARA, 3)
                        cv2.putText(frame_real, f"ZOOM: {zoom_camara:.1f}x", (puntos_px[8][0] + 15, puntos_px[8][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CAMARA, 2)
                        
                    else:
                        x_malla = max(-1.5, min((puntos_mano[9].x - 0.75) * 6.0, 1.5))
                        y_malla = max(-1.5, min((puntos_mano[9].y - 0.5) * 3.5, 1.5))
                        origen_gravedad = (x_malla, y_malla)
                        
                        dist_dedos = math.hypot(puntos_mano[8].x - puntos_mano[4].x, puntos_mano[8].y - puntos_mano[4].y)
                        
                        en_cooldown = (time.time() - tiempo_ultimo_toggle) < 1.5 
                        
                        if en_cooldown:
                            color_pinza = COLOR_COOLDOWN
                            cv2.putText(frame_real, "ESPERA...", (puntos_px[8][0] + 15, puntos_px[8][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_COOLDOWN, 2)
                        else:
                            color_pinza = (255, 150, 0) 
                            if dist_dedos < 0.05:
                                color_pinza = COLOR_FIJADA
                                if (time.time() - tiempo_ultimo_drop) > 1.0:
                                    masas_fijas.append((origen_gravedad, tamano_masa, densidad_masa))
                                    tiempo_ultimo_drop = time.time()
                                    if len(masas_fijas) > 10: masas_fijas.pop(0)
                                    cv2.circle(frame_real, puntos_px[8], 30, COLOR_FIJADA, -1)
                                    
                            elif dist_dedos > 0.20: 
                                color_pinza = COLOR_BORRADO
                                if (time.time() - tiempo_ultimo_borrado) > 1.0:
                                    if len(masas_fijas) > 0:
                                        masas_fijas.pop() 
                                        cv2.putText(frame_real, "DESHECHO!", (puntos_px[8][0] - 50, puntos_px[8][1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_BORRADO, 3)
                                    tiempo_ultimo_borrado = time.time()
                                
                        cv2.line(frame_real, puntos_px[4], puntos_px[8], color_pinza, 3)
                        cv2.circle(frame_real, puntos_px[9], 8, (255, 255, 255), -1)

        if pos_der_detectada and not modo_camara:
            masa_activa = (origen_gravedad, tamano_masa, densidad_masa)
        else:
            masa_activa = None
        
        lienzo_holograma = renderizar_espacio_multimasa(masas_fijas, masa_activa, W, H, angulo_camara_x, angulo_camara_z, zoom_camara)
        
        # --- NUEVO: BORDE MAGENTA EN LA VENTANA 3D ---
        if modo_camara:
            grosor_borde = 12
            cv2.rectangle(lienzo_holograma, (0, 0), (W-1, H-1), COLOR_CAMARA, grosor_borde)
        
        estado = "MODO CAMARA (Rotando y Zoom)" if modo_camara else "MODO EDICION (Esculpiendo)"
        color_hud = COLOR_CAMARA if modo_camara else COLOR_FIJADA
        cv2.putText(frame_real, estado, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_hud, 2)
        cv2.putText(frame_real, "Izquierda: Mano Abierta 1s (Solo 1 mano) = Cambiar Modo", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Panel de Control Multiverso", frame_real)
        cv2.imshow("Gravedad N-Body TD(FOSS)", lienzo_holograma)
        
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'): break
        elif tecla == ord('c'): masas_fijas = []

    camara.release()
    cv2.destroyAllWindows()
