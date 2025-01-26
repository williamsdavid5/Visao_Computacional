from PIL import Image
import cv2
import numpy as np
import os

# preparação as imagens para o treinamento


def get_image_data():
    paths = [os.path.join('yalefaces\\train', f)
             for f in os.listdir('yalefaces\\train')]
    # print(paths)
    faces = []
    ids = []
    for path in paths:
        # print(path)
        imagem = Image.open(path).convert('L')
        imagem_np = np.array(imagem, 'uint8')
        # print(type(imagem_np))
        id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
        ids.append(id)
        faces.append(imagem_np)

    return np.array(ids), faces


ids, faces = get_image_data()

# treinamento do classificador LBPH
lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.train(faces, ids)
lbph_classifier.write('2- Reconhecimento facial//lbph_classifier.yml')


# reconhecimento
lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read('2- Reconhecimento facial\\lbph_classifier.yml')

imagem_teste = 'yalefaces\\test\\subject07.happy.gif'
imagem = Image.open(imagem_teste).convert('L')
imagem_np = np.array(imagem, 'uint8')

previsao = lbph_face_classifier.predict(imagem_np)
saida_esperada = int(os.path.split(imagem_teste)[
                     1].split('.')[0].replace('subject', ''))

print(f"Pessoa ID: {previsao[0]}\nPessoa Esperada ID: {
      saida_esperada}\nConfiança: {previsao[1]:.2f}")

y_start = 30  # Posição inicial no eixo Y
line_height = 20  # Espaço entre as linhas

cv2.putText(imagem_np, f"Pred: {
            previsao[0]}", (10, y_start), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
cv2.putText(imagem_np, f"Exp: {saida_esperada}", (
    10, y_start + line_height), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)


cv2.imshow(f"Confidence: {previsao[1]:.2f}", imagem_np)
cv2.waitKey(0)
