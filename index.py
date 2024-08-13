import cv2
from ultralytics import YOLO
import winsound
import threading

# Carregar o modelo treinado
modelo = YOLO('best.pt')

# Abrir o vídeo
video = cv2.VideoCapture('ex3.mp4')

# Definir a área de interesse
area = [430, 240, 930, 250]

# Variável de controle do alarme
alarmeAtivado = False

def alarme_continuo():
    while True:
        winsound.Beep(2500, 1000)

while True:
    check, img = video.read()
    if not check:
        break

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
    if bebeDentroDaArea:
        if not alarmeAtivado:
            alarmeAtivado = True
            threading.Thread(target=alarme_continuo, daemon=True).start()

    # Se o alarme estiver ativado, exibir os elementos em vermelho e a mensagem
    if alarmeAtivado:
        cv2.rectangle(img2, (area[0], area[1]), (area[2], area[3]), (0, 0, 255), -1)
        cv2.rectangle(img, (100, 30), (470, 80), (0, 0, 255), -1)
        cv2.putText(img, 'BEBE EM PERIGO', (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # Combinar as imagens com sobreposição
    imgFinal = cv2.addWeighted(img2, 0.5, img, 0.5, 0)

    # Exibir o frame final com as detecções
    cv2.imshow('Detecções', imgFinal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
