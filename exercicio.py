import cv2

NOME = "Leticia Akemi Sumida"
RA = "24008474"

vermelho = (0, 0, 255)
verde = (0, 255, 0)
azul = (255, 0, 0)
branco = (255, 255, 255)

def dados_img(img):
    print("Largura", end=' ')
    print(img.shape[1])
    print("Altura", end=' ')
    print(img.shape[0])
    print("Canais", end=' ')
    print(img.shape[2])


# Captura da selfie
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir a camera")
    exit()

ret, imagem = cap.read()
cap.release()
if not ret:
    print("Erro ao capturar a imagem")
    exit()

print(imagem.shape)

dados_img(imagem)
altura = imagem.shape[0]
largura = imagem.shape[1]

# Colocando nome e RA, e posicionando de acordo com o RA
fonte = cv2.FONT_HERSHEY_SIMPLEX
escala = 0.7
espessura = 2
margem = 15
texto = NOME + " - " + RA

(larg_txt, alt_txt), base = cv2.getTextSize(texto, fonte, escala, espessura)
y_txt = margem + alt_txt

if int(RA[-1]) % 2 == 1:
    x_txt = largura - larg_txt - margem
else:
    x_txt = margem

# arrumando o texto
cv2.putText(imagem, texto, (x_txt, y_txt), fonte, escala, (0, 0, 0), espessura + 3, cv2.LINE_AA)
cv2.putText(imagem, texto, (x_txt, y_txt), fonte, escala, branco, espessura, cv2.LINE_AA)

# os dois circulos
centro = (largura // 2, altura // 2)
cv2.circle(imagem, centro, 60, vermelho, 3)

posicao_escolhida = (largura // 4, altura // 4)   # troque se quiser
cv2.circle(imagem, posicao_escolhida, 40, verde, 3)

# Retângulo na posição Y dada pelos dois últimos dígitos do RA
y_ret = int(RA[-2:])
cv2.rectangle(imagem, (0, y_ret), (largura - 1, y_ret + 30), azul, 3)

# mostrando e salvando
cv2.imshow("Atividade", imagem)
cv2.imwrite("selfie_editada.png", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()