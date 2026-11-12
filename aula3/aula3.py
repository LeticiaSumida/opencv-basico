import cv2

#capturar img camera e brincando com cores
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

    frame2 = frame.copy()
    for y in range(100, frame2.shape[0]):
        for x in range(200, frame2.shape[1]):
            frame2[y:y+5, x:x+5] = (255, 0, 0)
    cv2.imshow("imagem alterada", frame2)
    cv2.waitKey()

cap.release()