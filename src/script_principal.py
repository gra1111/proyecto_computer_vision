import cv2
import numpy as np

# ---------- Clasificación geométrica ----------


def classify_shape(contour):
    """Clasifica un contorno en 'SQUARE', 'CIRCLE', 'TRIANGLE', 'STAR' o None."""
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return None

    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    vertices = len(approx)
    area = cv2.contourArea(contour)
    if area == 0:
        return None

    circularity = 4 * np.pi * area / (peri * peri)

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
    Dentro de la ROI detecta figuras que cumplan color + forma:
      - CIRCLE rojo
      - TRIANGLE amarillo
      - SQUARE verde
      - STAR magenta
    Devuelve un conjunto con los identificadores de forma detectados.
    Puede devolver set() si no detecta nada.
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # (177,255,168) rojo círculo
    # (81,181,156)  verde cuadrado
    # (36,153,153)  amarillo triángulo
    # (138,204,144) magenta estrella
    color_defs = [
        {
            "shape_id": "CIRCLE",
            "label": "Circulo rojo",
            "lower": np.array([165, 70, 60]),
            "upper": np.array([179, 255, 255]),
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
            "label": "Triangulo amarillo",
            "lower": np.array([25, 100, 80]),
            "upper": np.array([47, 255, 255]),
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
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 500:
            continue

        shape = classify_shape(c)
        if shape != cd["shape_id"]:
            continue

        detected_shapes.add(cd["shape_id"])

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


def main(camera_index=0, width=1280, height=720):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    window_name = "Live Camera - press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    state_order = ["CIRCLE", "TRIANGLE", "SQUARE", "STAR"]
    state_index = 0
    previous_detected = "STAR"
    correct_password = False
    while not correct_password:
        ret, frame = cap.read()
        if not ret:
            print("No frame received (the camera may have been disconnected).")
            break
        # Optional: flip horizontally (mirror), comment out if not desired
        frame = cv2.flip(frame, 1)

        # Display the frame
        cv2.imshow(window_name, frame)

        display = frame.copy()

        h, w = display.shape[:2]

        # ROI en esquina superior izquierda
        roi_size = int(min(w, h) * 0.4)
        x1 = 20
        y1 = 20
        roi_size = min(roi_size, w - x1, h - y1)
        x2 = x1 + roi_size
        y2 = y1 + roi_size

        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(display, "Coloca el movil aqui",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)

        roi = frame[y1:y2, x1:x2]

        # Detectar formas color+forma en la ROI
        detected = detect_colored_shapes(roi, display, x1, y1)

        # Mostrar estado actual
        expected_shape = state_order[state_index]
        if detected and previous_detected != detected:
            if expected_shape in detected:
                state_index += 1
                if state_index >= len(state_order):
                    correct_password = True
            else:
                if state_index != 0:
                    print(
                        "Figura incorrecta. Volviendo al inicio de la maquina de estados.")
                state_index = 0
            previous_detected = detected
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
