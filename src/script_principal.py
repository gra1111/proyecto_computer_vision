import cv2
import numpy as np
import time


def classify_shape(contour):
    """Clasifica un contorno en 'SQUARE', 'CIRCLE', 'TRIANGLE', 'STAR' o None"""
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    vertices = len(approx)
    area = cv2.contourArea(contour)
    if area == 0:
        return None
    # circularity formula: 4π * Area / Perimeter**2 ->si es cercano a 1 la figura tiene forma circular
    # usado para diferenciar la estrella y el circulo
    circularity = 4 * np.pi * area / (perimeter * perimeter)

    if vertices == 3:
        return "TRIANGLE"

    if vertices == 4:
        return "SQUARE"

    if vertices >= 8:
        if circularity > 0.8:
            return "CIRCLE"
        else:
            return "STAR"

    return None


def detect_colored_shapes(roi, display, offset_x, offset_y):
    """
    detecta figuras que cumplan color + forma
    - CIRCLE rojo
    - TRIANGLE negro
    - SQUARE verde
    - STAR magenta
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    color_defs = [
        {
            "shape_id": "CIRCLE",
            "label": "Circulo rojo",
            "lower": np.array([155, 70, 60]),
            "upper": np.array([190, 255, 255]),
            "bgr":   (0, 0, 255),
        },
        {
            "shape_id": "SQUARE",
            "label": "Cuadrado verde",
            "lower": np.array([70, 100, 80]),
            "upper": np.array([95, 255, 255]),
            "bgr":   (0, 255, 0),
        },
        {
            "shape_id": "TRIANGLE",
            "label": "Triangulo negro",
            "lower": np.array([95, 120, 140]),
            "upper": np.array([120, 255, 255]),
            "bgr":   (0, 255, 255),
        },
        {
            "shape_id": "STAR",
            "label": "Estrella magenta",
            "lower": np.array([130, 100, 80]),
            "upper": np.array([150, 255, 255]),
            "bgr":   (255, 0, 255),
        },
    ]

    kernel = np.ones((3, 3), np.uint8)
    detected_shapes = set()

    for cd in color_defs:
        mask = cv2.inRange(hsv, cd["lower"], cd["upper"])
        # morphologyEx aplica apertura morfologica (erosion y luego dilatacion),
        # para quitarnos pequeños puntos blancos por iluminacion o textura
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # buscamos contorno para este color y nos quedamos el mas grande
        # (se asume que la figura es el contorno mas grande del color)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 500:
            continue

        # vemos si cuadra el shape que buscabamos para dicho color y si es asi nos lo quedamos
        shape = classify_shape(c)
        if shape != cd["shape_id"]:
            continue

        detected_shapes.add(cd["shape_id"])

        # display para qeu se vea que se esta detectando
        x, y, w_box, h_box = cv2.boundingRect(c)
        cv2.rectangle(roi, (x, y), (x + w_box, y + h_box), cd["bgr"], 2)
        cv2.rectangle(display,
                      (offset_x + x, offset_y + y),
                      (offset_x + x + w_box, offset_y + y + h_box),
                      cd["bgr"], 2)
        cv2.putText(display, cd["label"],
                    (offset_x + x, offset_y + y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cd["bgr"], 2)

    return detected_shapes


def detectar_pelota_rosa(frame):
    """
    Detecta la pelota rosa usando unicamente el color
    Devuelve una bbox (x, y, w, h) o None si no la encuentar
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # rango aproximado para rosa ajusatado con el color de nuestra pelota
    lower_pink = np.array([140, 70, 70])
    upper_pink = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 40:
        return None

    x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h)


def punto_en_rect(cx, cy, rect):
    if rect is None:
        return False
    x, y, w, h = rect
    return x <= cx <= x + w and y <= cy <= y + h


def main(camera_index=0, width=1280, height=720):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    window_name = "Live Camera - press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # fase 1 : contraseña
    state_order = ["CIRCLE", "TRIANGLE", "SQUARE", "STAR"]
    state_index = 0
    previous_detected = "STAR"
    correct_password = False

    while not correct_password:
        ret, frame = cap.read()
        if not ret:
            print("No frame received (the camera may have been disconnected).")
            cap.release()
            cv2.destroyAllWindows()
            return

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        h, w = display.shape[:2]

        # ROI en esquina arriba izquierda
        # hacemos que el ROI depende del tamaño del frame y no sea uno fijo
        roi_size = int(min(w, h) * 0.4)
        x1 = 20
        y1 = 20
        # para que no se salga de la imagen:
        roi_size = min(roi_size, w - x1, h - y1)
        x2 = x1 + roi_size
        y2 = y1 + roi_size

        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(display, "Coloca el movil aqui",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)

        roi = frame[y1:y2, x1:x2]

        # detectar formas color+forma en la ROI
        detected = detect_colored_shapes(roi, display, x1, y1)

        # maquina de estados para la contraseña
        expected_shape = state_order[state_index]
        if detected and previous_detected != detected:
            if expected_shape in detected:
                state_index += 1
                if state_index >= len(state_order):
                    correct_password = True
                    print("CONTRASENA CORRECTA. MODO TIROS ACTIVADO")
            else:
                if state_index != 0:
                    print("Figura incorrecta. Volviendo al inicio.")
                state_index = 0
            previous_detected = detected

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        # si se presiona la q o ESC salimos del programa
        # 27 equivale al ESC
        if key == ord('q') or key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    basket_rect = None    # zona de la papelera (x, y, w, h)
    intial_shooting_rect = None
    made_shots = 0

    # fase dos, pelota cnasata y kalman
    # Create the Kalman filter object
    kf = cv2.KalmanFilter(4, 2)
    # Initialize the state of the Kalman filter
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0]], dtype=np.float32)  # Measurement matrix np.array of shape (2, 4) and type np.float32
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)  # Transition matrix np.array of shape (4, 4) and type np.float32
    # Process noise covariance np.array of shape (4, 4) and type np.float32
    kf.processNoiseCov = np.eye(4, dtype=np.float32)

    measurement = np.array((2, 1), np.float32)
    prediction = np.zeros((2, 1), np.float32)

    # Show the frames to select the initial position of the object

    kalman_initialized = False
    # usams el tamaño del ultimo bounding box
    last_bbox = None
    basket_rect = None

    shot_in_progress = False
    scored_this_shot = False
    total_shots = 0
    missed_shots = 0
    made_shots = 0
    # nº de frames seguidos dentro para que sea canasta (15 decidido en pruebas en el lab)
    inside_frames_time = 15
    inside_counter = 0
    inside_tiro_prev = False

    # para caluclar los fps
    prev_time = time.time()
    fps = 0

    while True:
        # calculo de fps
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        if dt > 0:
            fps = 1.0 / dt

        ret, frame = cap.read()
        if not ret:
            print("No frame received (the camera may have been disconnected)")
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = display.shape[:2]

        cv2.putText(
            display,
            "MODO TIROS (pelota rosa + papelera + Kalman)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # mostrar papelera si ya esta definida y sino seleccionarla
        if basket_rect is None:
            cv2.putText(
                display,
                "Pulsa 'b' y selecciona la papelera con el raton, luego pulsa la t y selecciona en punto de incio del tiro",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        else:
            x_p, y_p, w_p, h_p = basket_rect
            cv2.rectangle(display, (x_p, y_p), (x_p + w_p, y_p + h_p),
                          (0, 0, 255), 2)
            cv2.putText(
                display,
                "PAPELERA",
                (x_p, y_p - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        if intial_shooting_rect is not None:
            x_t, y_t, w_t, h_t = intial_shooting_rect
            cv2.rectangle(display, (x_t, y_t), (x_t + w_t, y_t + h_t),
                          (255, 0, 0), 2)
            cv2.putText(
                display,
                "PUNTO DE TIRO",
                (x_t, y_t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        bbox = detectar_pelota_rosa(frame)

        if bbox is not None:
            last_bbox = bbox
            x_b, y_b, w_b, h_b = bbox
            c_x = x_b + w_b / 2.0
            c_y = y_b + h_b / 2.0

            if not kalman_initialized:
                # equivalente a cuando en el ejemplo se pulsa 's' y se inicializa
                kf.statePost = np.array([[c_x],
                                         [c_y],
                                         [0.],
                                         [0.]], np.float32)
                kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.01
                kalman_initialized = True

            measurement = np.array([[np.float32(c_x)],
                                    [np.float32(c_y)]], np.float32)
        else:
            measurement = None  # distinto: si no hay detección, solo predicción

        if kalman_initialized:
            prediction = kf.predict()
            p_x = float(prediction[0][0])
            p_y = float(prediction[1][0])

            if measurement is not None:
                kf.correct(measurement)

            # dibujamos un cuadrado que siga la pelota
            if last_bbox is not None:
                _, _, w_b, h_b = last_bbox
                x_pred = int(p_x - w_b / 2.0)
                y_pred = int(p_y - h_b / 2.0)
                cv2.rectangle(display,
                              (x_pred, y_pred),
                              (x_pred + int(w_b), y_pred + int(h_b)),
                              (0, 255, 0), 2)
                cv2.circle(display, (int(p_x), int(p_y)), 4, (0, 255, 0), -1)

                # logica de canastas y fallos
                inside_basket = basket_rect is not None and punto_en_rect(
                    p_x, p_y, basket_rect)
                inside_tiro = intial_shooting_rect is not None and punto_en_rect(
                    p_x, p_y, intial_shooting_rect)

                # 1) si hay tiro en curso comprobamos si hay una cansta
                if shot_in_progress:
                    if inside_basket:
                        inside_counter += 1
                        if inside_counter >= inside_frames_time and not scored_this_shot:
                            made_shots += 1
                            scored_this_shot = True
                            print("TIRO ANOTADO")
                    else:
                        inside_counter = 0

                # 2) inicio de tiro (sale del punto de tiro)
                if intial_shooting_rect is not None:
                    if inside_tiro_prev and not inside_tiro:
                        shot_in_progress = True
                        scored_this_shot = False
                        inside_counter = 0
                        total_shots += 1
                        print("NUEVO TIRO. Intentos totales:", total_shots)

                    # 3) final del tiro (vuelve al punto de tiro)
                    if not inside_tiro_prev and inside_tiro and shot_in_progress:
                        if not scored_this_shot:
                            missed_shots += 1
                            print("TIRO FALLADO. Fallos:", missed_shots)
                        shot_in_progress = False
                        scored_this_shot = False
                        inside_counter = 0

                    inside_tiro_prev = inside_tiro
            else:
                inside_counter = 0
                inside_tiro_prev = False

        # mostramos fallos canastas e intentos
        cv2.putText(
            display,
            f"Canastas: {made_shots}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        cv2.putText(
            display,
            f"Intentos: {total_shots}  Fallos: {missed_shots}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        # dibujo de los fps
        cv2.putText(display,
                    f"FPS: {fps:.1f}",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        # seleccionar la papelera con 'b'
        if key == ord('b'):
            roi = cv2.selectROI(window_name, frame,
                                showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(window_name)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            if roi[2] > 0 and roi[3] > 0:
                basket_rect = roi

        if key == ord('r'):
            # la r para resetear el contador
            total_shots = 0
            missed_shots = 0
            made_shots = 0
        # seleccionar el punto de tiro con la 't'
        if key == ord('t'):
            roi = cv2.selectROI(window_name, frame,
                                showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(window_name)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            if roi[2] > 0 and roi[3] > 0:
                intial_shooting_rect = roi

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
