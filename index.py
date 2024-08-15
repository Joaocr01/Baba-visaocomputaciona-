from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import threading
import time
import pygame

app = Flask(__name__)

modelo = YOLO('best.pt')

# Inicialize o pygame mixer
pygame.mixer.init()

# Carregue o arquivo de som que você deseja usar como alarme
alarme_som = pygame.mixer.Sound('C:/Users/jc018/Desktop/tube2/videoplayback.wav')

# Abrir o vídeo
video = cv2.VideoCapture('ex3.mp4')

# Definir a área
area = [430, 240, 930, 280]

# Variáveis de controle do alarme e de pausa
alarmeAtivado = False
paused = False
alarmeDesativado = True
alarmeThread = None
alarmeLock = threading.Lock()

# FPS
fps = video.get(cv2.CAP_PROP_FPS)

# quantidade de frames pulados
frames_pular = 10

def alarme_continuo():
    global alarmeDesativado
    while True:
        with alarmeLock:
            if alarmeDesativado:
                pygame.mixer.stop()  # Para o som quando o alarme for desativado
                break
        if not paused:
            if not pygame.mixer.get_busy():  # Toca o som apenas se não estiver tocando
                alarme_som.play()
            time.sleep(0.5)  # Pequena pausa entre os toques do alarme
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/toggle_pause')
def toggle_pause():
    global paused
    paused = not paused
    return jsonify({"status": "paused" if paused else "playing"})

@app.route('/toggle_alarme')
def toggle_alarme():
    global alarmeAtivado, alarmeDesativado, alarmeThread

    with alarmeLock:
        if alarmeAtivado:
            alarmeDesativado = True
            if alarmeThread:
                alarmeThread.join()
            alarmeAtivado = False
            return jsonify({"alarmeAtivado": alarmeAtivado})

        else:
            alarmeDesativado = False
            alarmeAtivado = True
            alarmeThread = threading.Thread(target=alarme_continuo, daemon=True)
            alarmeThread.start()
            return jsonify({"alarmeAtivado": alarmeAtivado})

@app.route('/restart_video')
def restart_video():
    global video, alarmeAtivado, alarmeDesativado, alarmeThread, paused

    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Reiniciar o alarme se estiver ativado
    with alarmeLock:
        if alarmeAtivado:
            alarmeDesativado = True
            if alarmeThread:
                alarmeThread.join()
            alarmeAtivado = False

    # Garantir que o vídeo não esteja pausado ao reiniciar
    paused = False

    return jsonify({"status": "video restarted", "alarmeAtivado": alarmeAtivado})

def gerar_frames():
    global alarmeAtivado, paused, alarmeDesativado

    while True:
        if paused:
            time.sleep(0.1)
            continue

        check, img = video.read()
        if not check:
            # Reiniciar o vídeo
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Pular frames para acelerar o vídeo
        for _ in range(frames_pular):
            video.grab()

        # Redimensionar a imagem
        img = cv2.resize(img, (1270, 720))

        # Criar sobreposição
        img2 = img.copy()

        # Desenhar área de interesse na imagem copiada
        cv2.rectangle(img2, (area[0], area[1]), (area[2], area[3]), (0, 255, 0), -1)

        # Detectar objetos
        resultados = modelo(img)

        bebeDentroDaArea = False

        for resultado in resultados:
            boxes = resultado.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0].item())
                label = modelo.names[cls]

                # Verificar se a classe detectada é 'bebe'
                if label == 'bebe':
                    # Desenhar a caixa delimitadora
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)

                    # Calcular o centro da caixa delimitadora
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # Verificar se o centro do objeto está na área de interesse
                    if area[0] <= cx <= area[2] and area[1] <= cy <= area[3]:
                        bebeDentroDaArea = True

        # Ativar o alarme contínuo se o bebê for detectado e alterar a exibição
        if bebeDentroDaArea and not paused:
            if not alarmeAtivado:
                with alarmeLock:
                    alarmeDesativado = False
                    alarmeAtivado = True
                    alarmeThread = threading.Thread(target=alarme_continuo, daemon=True)
                    alarmeThread.start()

        # Deixar o alarme em evidência
        if alarmeAtivado:
            cv2.rectangle(img2, (area[0], area[1]), (area[2], area[3]), (0, 0, 255), -1)
            cv2.rectangle(img, (100, 30), (470, 80), (0, 0, 255), -1)
            cv2.putText(img, 'BEBE EM PERIGO', (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        # Combinar as imagens com sobreposição
        imgFinal = cv2.addWeighted(img2, 0.5, img, 0.5, 0)

        # Codificar como JPEG
        ret, jpeg = cv2.imencode('.jpg', imgFinal)
        if not ret:
            continue

        # Converter para bytes e gerar o frame
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gerar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
