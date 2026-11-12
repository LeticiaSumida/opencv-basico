# Atividade - Detecção de bordas (VCIOT / SLMM 2026)

import os
import sys
import cv2
import numpy as np

NOME = "Letícia Akemi Sumida"
RA = "000000000"

BASE = os.path.dirname(os.path.abspath(__file__))
PASTA_IMAGEM = os.path.join(BASE, "imagem")
PASTA_FOTOS = os.path.join(BASE, "fotos")
PASTA_RESULTADOS = os.path.join(BASE, "resultados")
for p in (PASTA_IMAGEM, PASTA_FOTOS, PASTA_RESULTADOS):
    os.makedirs(p, exist_ok=True)

BLUR_PADRAO, C1_PADRAO, C2_PADRAO = 7, 70, 150
TESTES_BLUR = [3, 15, 31]
TESTES_CANNY = [(10, 50), (150, 300)]


def escreve(img, texto, cor=(255, 0, 0), embaixo=False):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    pos = (10, img.shape[0] - 12) if embaixo else (10, 20)
    cv2.putText(img, texto, pos, fonte, 0.5, cor, 1, cv2.LINE_AA)


def identificar(img):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    y = 30
    for linha in (NOME, "RA: " + RA):
        cv2.putText(img, linha, (10, y), fonte, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, linha, (10, y), fonte, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y += 32


def tirar_selfie():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Não foi possível abrir a câmera.")
        return None
    foto = None
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        previa = frame.copy()
        cv2.putText(previa, "ESPACO = tirar foto | ESC = sair", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Camera - selfie", previa)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 32:
            foto = frame
            break
        if tecla == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
    return foto


def carregar_selfie(usar_camera):
    caminho = os.path.join(PASTA_FOTOS, "selfie.jpg")
    if usar_camera:
        foto = tirar_selfie()
        if foto is None:
            return None
        foto = cv2.resize(foto, (640, 480))
        identificar(foto)
        cv2.imwrite(caminho, foto)
        print("Selfie salva em", caminho)
        return foto
    return cv2.imread(caminho)


def carregar_objetos():
    for arq in sorted(os.listdir(PASTA_IMAGEM)):
        if arq.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            img = cv2.imread(os.path.join(PASTA_IMAGEM, arq))
            if img is not None:
                print("Foto dos objetos:", arq)
                return cv2.resize(img, (640, 480))
    return None


def detectar(imgColorida, blur, c1, c2):
    # Passo 1: tons de cinza
    img = cv2.cvtColor(imgColorida, cv2.COLOR_BGR2GRAY)
    # Passo 2: blur
    suave = cv2.blur(img, (blur, blur))
    # Passo 3: binarização com Otsu (no slide é com mahotas; aqui usamos o Otsu do OpenCV,
    # que faz a mesma coisa: pixels acima do limiar T viram 255, o resto 0)
    T, bin = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin = cv2.bitwise_not(bin)
    # Passo 4: Canny
    bordas = cv2.Canny(bin, c1, c2)
    # Passo 5: contornos externos
    objetos, lx = cv2.findContours(bordas.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return img, suave, bin, bordas, objetos


def processar(nome, imgColorida, blur, c1, c2):
    img, suave, bin, bordas, objetos = detectar(imgColorida, blur, c1, c2)

    img, suave = img.copy(), suave.copy()
    escreve(img, "Imagem em tons de cinza", 0)
    escreve(suave, "Suavizacao com Blur (%dx%d)" % (blur, blur), 0)
    escreve(bin, "Binarizacao com Metodo Otsu", 255)
    escreve(bordas, "Canny (%d, %d)" % (c1, c2), 255)
    etapas = np.vstack([np.hstack([img, suave]), np.hstack([bin, bordas])])

    resultado = imgColorida.copy()
    cv2.drawContours(resultado, objetos, -1, (255, 0, 0), 2)
    
    escreve(resultado, "%d objetos encontrados!" % len(objetos), (0, 255, 255), embaixo=True)

    sufixo = "blur%d_canny%d_%d" % (blur, c1, c2)
    cv2.imwrite(os.path.join(PASTA_RESULTADOS, "%s_etapas_%s.jpg" % (nome, sufixo)), etapas)
    cv2.imwrite(os.path.join(PASTA_RESULTADOS, "%s_contornos_%s.jpg" % (nome, sufixo)), resultado)
    print("%-8s blur=%2d canny=(%4d,%4d) -> %d objetos" % (nome, blur, c1, c2, len(objetos)))


def rodar(nome, imgColorida):
    print("\n=== %s ===" % nome)
    processar(nome, imgColorida, BLUR_PADRAO, C1_PADRAO, C2_PADRAO)
    for b in TESTES_BLUR:
        processar(nome, imgColorida, b, C1_PADRAO, C2_PADRAO)
    for c1, c2 in TESTES_CANNY:
        processar(nome, imgColorida, BLUR_PADRAO, c1, c2)


if __name__ == "__main__":
    usar_camera = "--sem-camera" not in sys.argv

    selfie = carregar_selfie(usar_camera)
    if selfie is None:
        print("Selfie não disponível (tire a foto ou use fotos/selfie.jpg).")
    else:
        rodar("selfie", selfie)

    objetos = carregar_objetos()
    if objetos is None:
        print("Coloque a foto dos objetos na pasta 'imagem'.")
    else:
        rodar("objetos", objetos)