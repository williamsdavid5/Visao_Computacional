from PIL import Image
import cv2
import numpy as np
import os
from sklearn.metrics import accuracy_score

# preparação as imagens para o treinamento


def get_image_data():  # a função percorre a pasta com as imagen de treino e resgata o id de cada imagem de seus pixels
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
# o classificador cria um arquivo que irá conter os dados para fazer o reconhecimento das pessoas
lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.train(faces, ids)
lbph_classifier.write('2- Reconhecimento facial//lbph_classifier.yml')


# reconhecimento
# se baseando no arquivo com os dados de cada face, o algoriitmo irá fazer o reconhecimento em uma imagem
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

# resultado do reconhecimento em uma imagem
# cv2.putText(imagem_np, f"Pred: {
#             previsao[0]}", (10, y_start), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
# cv2.putText(imagem_np, f"Exp: {saida_esperada}", (
#     10, y_start + line_height), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
# cv2.imshow(f"Confidence: {previsao[1]:.2f}", imagem_np)
# cv2.waitKey(0)

# avaliação do classificador
# separa os ids existentes na pasta e faz as previsões para cada imagem
# assim poderiamos comparar os resultados e medir a acuracia
paths = [os.path.join('yalefaces\\test', f)
         for f in os.listdir('yalefaces\\test')]
previsoes = []
saidas_esperadas = []

for path in paths:
    imagem = Image.open(path).convert('L')
    imagem_np = np.array(imagem, 'uint8')
    previsao, _ = lbph_face_classifier.predict(imagem_np)
    saida_esperada = int(os.path.split(path)[1].split('.')[
                         0].replace('subject', ''))
    # print(saida_esperada)

    previsoes.append(previsao)
    saidas_esperadas.append(saida_esperada)

previsoes = np.array(previsoes)
saidas_esperadas = np.array(saidas_esperadas)

# print(previsoes)
# print(saidas_esperadas)
print(f'Acuracia: {100*accuracy_score(saidas_esperadas, previsoes)} %')
