import cv2

#capturar img camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir a camera")
    exit()

ret, frame = cap.read()
if ret:
    print(frame.shape)

    cv2.imshow("cap", frame)
    cv2.waitKey(0)
    cv2.imwrite('img_png.png', frame)
    cv2.imwrite('img_jpg.jpg', frame)

cap.release()