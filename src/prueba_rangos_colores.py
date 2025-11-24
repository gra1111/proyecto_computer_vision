import cv2
import numpy as np

# Variables globales para el frame actual
current_frame = None


def on_mouse(event, x, y, flags, param):
    global current_frame
    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        # Ojo: x = columnas, y = filas
        bgr = current_frame[y, x]  # B, G, R
        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])

        # Convertir ese único píxel a HSV
        pixel_bgr = np.uint8([[[b, g, r]]])      # shape (1,1,3)
        pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = pixel_hsv[0, 0]

        print(f"Pos ({x},{y})  BGR=({b},{g},{r})  HSV=({h},{s},{v})")


def main():
    global current_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return

    window_name = "Video - click para ver color (q/Esc para salir)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        current_frame = frame

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
