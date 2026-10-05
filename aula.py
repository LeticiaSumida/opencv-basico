import cv2

# comando pra ler imagem
# no python podemos ler 3 tipos de diferentes com o mesmo comando
def dados_img(img):
    print("Largura", end=' ')
    print(img.shape[1])
    print("Altura", end=' ')
    print(img.shape[0])
    print("Canais", end=' ')
    print(img.shape[2])

png = cv2.imread('desenho1.png')
jpg = cv2.imread('desenho2.jpg')
bmp = cv2.imread('desenho3.bmp')

dados_img(png)
dados_img(jpg)
dados_img(bmp)

cv2.imshow("png", png)
cv2.waitKey(0)
cv2.destroyWindows("png")

cv2.imshow("jpg", jpg)
cv2.waitKey(0)
cv2.destroyWindows("jpg")

cv2.imshow("bmp", bmp)
cv2.waitKey(0)
cv2.destroyWindows("bmp")

cv2.destroyAllWindows()