import cv2
from matplotlib import pyplot as plt


def mostrar(img, titulo):
    imagem_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(imagem_rgb)
    plt.axis('off')
    plt.title(titulo)
    plt.show()


imagem = cv2.imread(
    'Images/car.jpg')
# mostrar(imagem, "imagem original")

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
# mostrar(imagem_cinza, "imagem escala de cinza")

detector_carro = cv2.CascadeClassifier('Cascades/cars.xml')
detecoes_carro = detector_carro.detectMultiScale(
    imagem_cinza, scaleFactor=1.04, minNeighbors=3)

for (x, y, w, h) in detecoes_carro:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 2)

# mostrar(imagem, "detecções de carros")

################################################################################################

imagem = cv2.imread('Images/clock.jpg')
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

detector_relgoio = cv2.CascadeClassifier('Cascades/clocks.xml')
deteccoes_relogio = detector_relgoio.detectMultiScale(
    imagem_cinza, scaleFactor=1.01, maxSize=(110, 110), minNeighbors=3)


for (x, y, w, h) in deteccoes_relogio:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 2)

# mostrar(imagem, "detecções relogio")

################################################################################################

imagem = cv2.imread('Images/people3.jpg')
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

detector_corpo = cv2.CascadeClassifier('Cascades/fullbody.xml')
deteccoes_corpo = detector_corpo.detectMultiScale(
    imagem_cinza, scaleFactor=1.01, minSize=(70, 100))


i = 1
for (x, y, w, h) in deteccoes_corpo:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(imagem, str(i), (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    i += 1

mostrar(imagem, "detecções corpo")
