from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import threading
import winsound

app = Flask(__name__)

modelo = YOLO('best.pt')

# Abrir o vídeo
video = cv2.VideoCapture('ex3.mp4')

# Definir a área de interesse
area = [430, 240, 930, 280]

# Variáveis de controle do alarme e de pausa
alarmeAtivado = False
paused = False
alarmeDesativado = False
alarmeThread = None

def alarme_continuo():
    global alarmeDesativado
    while not alarmeDesativado:
        winsound.Beep(2500, 1000)

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

    if alarmeAtivado:
        alarmeDesativado = True
        alarmeAtivado = False
        if alarmeThread:
            alarmeThread.join()  # Aguarda a thread do alarme parar
    else:
        alarmeDesativado = False

    return jsonify({"alarmeAtivado": alarmeAtivado})

def gerar_frames():
    global alarmeAtivado, paused, alarmeDesativado, alarmeThread

    while True:
        if paused:
            if alarmeAtivado:
                alarmeDesativado = True  # Desativa o alarme se o vídeo estiver pausado
                if alarmeThread:
                    alarmeThread.join()
            continue

        check, img = video.read()
        if not check:
            # Reiniciar o vídeo
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Redimensionar a imagem para o tamanho desejado
        img = cv2.resize(img, (1270, 720))

        # Criar uma cópia da imagem para sobreposição
        img2 = img.copy()

        # Desenhar a área de interesse na imagem copiada
        cv2.rectangle(img2, (area[0], area[1]), (area[2], area[3]), (0, 255, 0), -1)

        # Detectar objetos na imagem
        resultados = modelo(img)

        bebeDentroDaArea = False

        for resultado in resultados:
            boxes = resultado.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0].item())
                label = modelo.names[cls]

                # Verificar se a classe detectada é "bebe"
                if label == 'bebe':
                    # Desenhar a caixa delimitadora no frame original
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)

                    # Calcular o centro da caixa delimitadora
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # Verificar se o centro do objeto está dentro da área de interesse
                    if area[0] <= cx <= area[2] and area[1] <= cy <= area[3]:
                        bebeDentroDaArea = True

        # Se o bebê foi detectado dentro da área, iniciar o alarme contínuo e alterar a exibição
        if bebeDentroDaArea and not paused and not alarmeDesativado:
            if not alarmeAtivado:
                alarmeAtivado = True
                alarmeThread = threading.Thread(target=alarme_continuo, daemon=True)
                alarmeThread.start()

        # Se o alarme estiver ativado, exibir os elementos em vermelho e a mensagem
        if alarmeAtivado:
            cv2.rectangle(img2, (area[0], area[1]), (area[2], area[3]), (0, 0, 255), -1)
            cv2.rectangle(img, (100, 30), (470, 80), (0, 0, 255), -1)
            cv2.putText(img, 'BEBE EM PERIGO', (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        # Combinar as imagens com sobreposição
        imgFinal = cv2.addWeighted(img2, 0.5, img, 0.5, 0)

        # Codificar o frame como JPEG
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
