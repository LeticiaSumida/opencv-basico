import argparse
import os
import time
from datetime import datetime

import cv2

PASTA = os.path.dirname(os.path.abspath(__file__))

# Utilitários
def carregar_cascade(nome):
    caminho = os.path.join(cv2.data.haarcascades, nome)
    clf = cv2.CascadeClassifier(caminho)
    if clf.empty():
        raise IOError(f"Não foi possível carregar o Haar Cascade: {caminho}")
    return clf


def escreve(img, texto, pos=(10, 25), cor=(0, 255, 0), escala=0.6):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, pos, fonte, escala, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, texto, pos, fonte, escala, cor, 1, cv2.LINE_AA)


def preprocessar(frame):
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(cinza)

# Detecção
class DetectorFaces:
    def __init__(self, scale=1.1, vizinhos=7, min_size=60, detectar_olhos=True, exigir_olhos=True):
        self.faces_clf = carregar_cascade("haarcascade_frontalface_default.xml")
        self.olhos_clf = carregar_cascade("haarcascade_eye.xml") if detectar_olhos else None
        # Validação cruzada: uma face só vale se tiver >=1 olho (elimina falsos positivos)
        self.exigir_olhos = exigir_olhos and detectar_olhos
        self.scale = scale
        self.vizinhos = vizinhos
        self.min_size = (min_size, min_size)

    def detectar(self, frame):
        cinza = preprocessar(frame)
        faces = self.faces_clf.detectMultiScale(
            cinza,
            scaleFactor=self.scale,
            minNeighbors=self.vizinhos,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        resultado = []
        for (x, y, w, h) in faces:
            olhos_abs = []
            if self.olhos_clf is not None:
                # Olhos só são procurados na metade superior do rosto
                # (reduz falsos positivos e acelera a busca).
                roi = cinza[y:y + int(h * 0.6), x:x + w]
                olhos = self.olhos_clf.detectMultiScale(
                    roi, scaleFactor=1.1, minNeighbors=8,
                    minSize=(w // 8, w // 8))
                olhos_abs = [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in olhos]
            if self.exigir_olhos and not olhos_abs:
                continue
            resultado.append(((x, y, w, h), olhos_abs))
        return resultado

    def desenhar(self, frame, deteccoes):
        for i, ((x, y, w, h), olhos) in enumerate(deteccoes, 1):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
            escreve(frame, f"Face {i}", (x, max(y - 8, 15)), (0, 255, 255), 0.55)
            for (ex, ey, ew, eh) in olhos:
                cx, cy = ex + ew // 2, ey + eh // 2
                cv2.circle(frame, (cx, cy), ew // 2, (255, 128, 0), 2)
        escreve(frame, f"{len(deteccoes)} face(s) encontrada(s)", (10, frame.shape[0] - 15))
        return frame


def nome_saida(prefixo, ext):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return os.path.join(PASTA, f"{prefixo}_{ts}.{ext}")

# Modos
def modo_imagem(det, caminho, mostrar):
    img = cv2.imread(caminho)
    if img is None:
        raise IOError(f"Não consegui abrir a imagem: {caminho}")
    t0 = time.time()
    deteccoes = det.detectar(img)
    dt = (time.time() - t0) * 1000
    det.desenhar(img, deteccoes)
    saida = nome_saida("rosto", "jpg")
    cv2.imwrite(saida, img)
    print(f"{len(deteccoes)} face(s) em {dt:.1f} ms -> {saida}")
    if mostrar:
        cv2.imshow("Resultado", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def modo_stream(det, fonte, mostrar, salvar_video=True):
    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir a fonte de vídeo: {fonte}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    larg = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if salvar_video:
        saida = nome_saida("video_faces", "mp4")
        writer = cv2.VideoWriter(saida, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (larg, alt))

    n, total_faces, t_ant, fps = 0, 0, time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        deteccoes = det.detectar(frame)
        det.desenhar(frame, deteccoes)

        n += 1
        total_faces += len(deteccoes)
        agora = time.time()
        fps = 0.9 * fps + 0.1 / max(agora - t_ant, 1e-6)
        t_ant = agora
        escreve(frame, f"FPS: {fps:.1f}", (10, 50), (0, 200, 255))

        if writer:
            writer.write(frame)
        if mostrar:
            cv2.imshow("Haar - q: sair | s: salvar frame", frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("s"):
                cv2.imwrite(nome_saida("frame", "jpg"), frame)

    cap.release()
    if writer:
        writer.release()
        print(f"Vídeo salvo em: {saida}")
    if mostrar:
        cv2.destroyAllWindows()
    print(f"{n} frames processados | média {total_faces / max(n, 1):.2f} faces/frame")


def main():
    ap = argparse.ArgumentParser(description="Detector de faces Haar Cascade")
    ap.add_argument("--imagem", help="caminho de uma imagem")
    ap.add_argument("--video", help="caminho de um vídeo")
    ap.add_argument("--camera", type=int, default=0, help="índice da webcam")
    ap.add_argument("--scale", type=float, default=1.1, help="scaleFactor")
    ap.add_argument("--vizinhos", type=int, default=7, help="minNeighbors")
    ap.add_argument("--min-size", type=int, default=60, help="tamanho mínimo da face (px)")
    ap.add_argument("--sem-olhos", action="store_true", help="desativa detecção de olhos")
    ap.add_argument("--aceitar-sem-olhos", action="store_true", help="aceita faces sem olhos detectados (mais sensível, mais falsos positivos)")
    ap.add_argument("--sem-janela", action="store_true", help="não abrir janelas")
    args = ap.parse_args()

    print("OpenCV", cv2.__version__)
    det = DetectorFaces(args.scale, args.vizinhos, args.min_size, not args.sem_olhos,
                        not args.aceitar_sem_olhos)
    mostrar = not args.sem_janela

    if args.imagem:
        modo_imagem(det, args.imagem, mostrar)
    elif args.video:
        modo_stream(det, args.video, mostrar)
    else:
        modo_stream(det, args.camera, mostrar, salvar_video=False)


if __name__ == "__main__":
    main()