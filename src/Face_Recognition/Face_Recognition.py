import numpy as np
import cv2
import skimage


from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt

from os import listdir

import tensorflow as tf

from sklearn.neighbors import BallTree

from skimage.filters import gaussian

class Face_Recognition(): 
  '''
    O objetivo da class Face_Recognition é abstrair o processo de reconhecimento facial com o modelo CNN Vgg Faces.
    Todos as informações do objeto são inicializados e deixados vazios até ser adicionado informações. Nescessario 
    que o banco de dados esteja dividido da seguinte maneira: 
    pasta_banco_de_dados/individuo_fransisco/foto_1_de_fransisco, pasta_banco_de_dados/individuo_fransisco/foto_2_de_fransisco, ...
  '''

  def __init__(self):
    pass

  #recxorte de rostos

  #Sem Ruido
  def preprocess_image(self,image_path):
    '''
      Para pre-processar as imagens, facilitando o processo de extração de caracteristicas.

      :parametro image_path: string que informa o diretorio da imagem que vai ser processada.
      :retorna a imagem processada
    '''
    img = cv2.imread(image_path)


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #inserindo ruido

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #Preprocessando ruido

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  #recorte da face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    #filtro Gaussiano
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray,cv2.COLOR_BGR2RGB)
    smooth = cv2.GaussianBlur(gray,((gray.shape[0]//7)|1,(gray.shape[1]//7)|1),0)
    gray = cv2.divide(gray, smooth, scale=255)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
      img = img[y:y+h,x:x+w]
      break


    img=cv2.resize(img,(224, 224))

    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.show()


    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img


  def loadVggFaceModel(self,diretorio_do_modelo):
    '''
      Para inicializar o modeno CNN VGG faces, para posteriormente extrair as caracteristicas da foto.

      :retorna o modelo pronto paara extrarir as caracteristicas.
    '''
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("physical_devices-------------", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    self.diretorio_do_modelo = diretorio_do_modelo


    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    #you can download pretrained weights from https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
    from keras.models import model_from_json
    
    try:
      model.load_weights(self.diretorio_do_modelo)
      vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    
      return vgg_face_descriptor
    except:
      print('Não foi possivel encontrar o modelo treinado no diretorio informado !!!')
      print('Caso não tenha esse arquivo, siga as instruções do arquivo READEME desssa biblioteca !!!')
      return None


  def criando_modelo(self,diretorio_do_modelo):
    '''
      Para servir de interface onde é chamada a função que cria o modelo e armazena esse modelo nas informações do objeto.
    '''
    self.model = self.loadVggFaceModel(diretorio_do_modelo)

  def lista_individuo_treino(self,diretorio,qunatidade_de_trieno):
    '''
      Para retornar uma lista com strings que informa o diretorio das imagens de treino.

      :parametro diretorio: string que informa o diretorio das imagens que irão ser processadas e armazenadas no banco de treino.
      :parametro qunatidade_de_trieno: interio que informa a quantidade de imagens que serão armazenadas no banco de treino.
      :retorna uma lista com o strings que informam o diretorio das imagens de treino
    '''
    lista_diretorio_treino = []
    diretorio_pastas = listdir(diretorio)

    for conte in range(0,qunatidade_de_trieno):
      lista_diretorio_treino.append(diretorio+diretorio_pastas[conte])
    
    #print('\n')
    return lista_diretorio_treino 

  def lista_individuo_teste(self,diretorio,qunatidade_de_trieno,quantidade_testes):
    '''
      Para retornar uma lista com strings que informa o diretorio das imagens de teste.

      :parametro diretorio: string que informa o diretorio das imagens que irão ser processadas e armazenadas no banco de teste.
      :parametro qunatidade_de_trieno: interio que informa a quantidade de imagens que serão armazenadas no banco de treino.
      :parametro quantidade_testes: interio que informa a quantidade de imagens que serão armazenadas no banco de teste.
      :retorna uma lista com o strings que informam o diretorio das imagens de teste
    '''

    lista_diretorio_teste = []
    diretorio_pastas = listdir(diretorio)

    for conte in range(qunatidade_de_trieno,quantidade_testes):
      lista_diretorio_teste.append(diretorio+diretorio_pastas[conte])
    
    #print('\n')
    return lista_diretorio_teste 

  def dicionario_diretorio_treino_teste(self,diretorio_da_base,qunatidade_de_trieno,quantidade_teste):
    '''
      Para retornar um dicionarios contendo todas os diretorios de iagens para treiono e tes, organizadas da seguinte forma: diretorio_treino_teste[identificador do individuo]=[lista com diretorios das imagens de treino,lista com diretorios das imagens de teste].

      :parametro diretorio_da_base: diretorio das pastas que contem outras pastas com fotos do individuo.
      :parametro qunatidade_de_trieno: interio que informa a quantidade de imagens que serão armazenadas no banco de treino.
      :retorna um dicionario com os diretorios das fotos de treino e teste.
    '''
    diretorio_treino_teste = dict()

    for pasta in listdir(diretorio_da_base):
      #diretorio_treino_teste[n_individuo]:[[treino],[teste]]
      treino = self.lista_individuo_treino(diretorio_da_base + pasta + '/',qunatidade_de_trieno)
      #teste = self.lista_individuo_teste(diretorio_da_base + pasta + '/',qunatidade_de_trieno,qunatidade_de_trieno)
      teste = self.lista_individuo_teste(diretorio_da_base + pasta + '/',qunatidade_de_trieno,qunatidade_de_trieno+quantidade_teste)
      diretorio_treino_teste[pasta]=[treino,teste]
    
    return diretorio_treino_teste

  def extracao_de_caracteristicas_1_foto(self,diretorio_foto_individuo):
    '''
      Para chamar a função que extrai as caracteristicas de uma foto e retorna essas caracteristicas.

      :parametro diretorio_foto_individuo: string que informa o diretorio da foto que deseja extrar as caracteristicas.
      :retorna as caracteristicas extraidas da foto.
    '''
    return self.model.predict(self.preprocess_image('{}'.format(diretorio_foto_individuo)))[0,:]

  def extracao_de_caracteristicas_diretorio_treino(self,diretorio_treino_teste):
    '''
      Para servir de interface para a extração de caracteristicas de todas as fotos selecionadas para treino.

      :parametro diretorio_treino_teste: dicionario com o diretorio das fotos de treino e teste.
      :retorna uma lista fromatada da seguinte maneira [lista com identificador dos individuos,lista com as caracteristicas do individuo , lista com o diretorio da foto].
    '''
    #------------------------

    #coloque as fotos de seus funcionários neste caminho como name_of_employee.jpg 
    #diretorio_treino = "/content/drive/MyDrive/Reconhecimento Facial/TREINO 2/"
    
    caracteristicas_individuo_treino = []
    numero_individuo_treino = []
    diretorio_individuo_treino = []

    for individuo in diretorio_treino_teste:
      for diretorio_foto_individuo in diretorio_treino_teste[individuo][0]:
        caracteristicas_individuo_treino.append(self.extracao_de_caracteristicas_1_foto(diretorio_foto_individuo))
        numero_individuo_treino.append(individuo)
        diretorio_individuo_treino.append(diretorio_foto_individuo)

    print("TREINO foi realizado com sucesso ")

    base_treino=[numero_individuo_treino,caracteristicas_individuo_treino,diretorio_individuo_treino]

    return base_treino


  def extracao_de_caracteristicas_diretorio_teste(self,diretorio_treino_teste):
    '''
      Para servir de interface para a extração de caracteristicas de todas as fotos selecionadas para teste.

      :parametro diretorio_treino_teste: dicionario com o diretorio das fotos de treino e teste.
      :retorna uma lista formatada da seguinte maneira [lista com identificador dos individuos,lista com as caracteristicas do individuo , lista com o diretorio da foto].
    '''
        #coloque as fotos de seus funcionários neste caminho como name_of_employee.jpg 
    #diretorio_treino = "/content/drive/MyDrive/Reconhecimento Facial/TREINO 2/"
    
    caracteristicas_individuo_teste = []
    numero_individuo_teste = []
    diretorio_individuo_teste = []

    for individuo in diretorio_treino_teste:
      for diretorio_foto_individuo in diretorio_treino_teste[individuo][1]:

        caracteristicas_individuo_teste.append(self.extracao_de_caracteristicas_1_foto(diretorio_foto_individuo))
        numero_individuo_teste.append(individuo)
        diretorio_individuo_teste.append(diretorio_foto_individuo)

    print("TESTE foi realizado com sucesso ")

    base_teste = [numero_individuo_teste,caracteristicas_individuo_teste,diretorio_individuo_teste]

    return base_teste

  def findCosineSimilarity(self,source_representation, test_representation):
    '''
      Para retorna o coeficiente de similaridade entre duas fotos com caracteristicas extraidas.

      :parametro source_representation: caracteristicas da foto 1.
      :parametro test_representation: caracteristicas da foto 2.
      :retorna o coeficiente ade similaridade entre essas duas fotos.
    '''
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

  def classificacao_individuo(self,individuo_teste,base_treino):
    '''
      Para retorna o identificador com maior similaridade do individuo de teste.

      :parametro individuo_teste: caracteristicas do indivifuo de teste que iremos identificar.
      :parametro base_treino: base com as caracteristicas dos individuos de treino.
      :retorna o indentificador da foto com maior similaridade com a foto do individuo de teste.
    '''
    found = 0
    manor_similaridade = 0.50
    passagem_pelo_treino = 0
    label_name = None

    for conte in range(0,len(base_treino[0])):
      name = base_treino[0][conte]

      representation = base_treino[1][conte]
      similarity = self.findCosineSimilarity(representation, individuo_teste)

      if((manor_similaridade>similarity)):

        label_name = name
        manor_similaridade = similarity			
        found = 1
        
    if(found == 0): #if found image is not in employee database
      print('-menor similaridade >> {}'.format(similarity))
      print('-classe da menor similaridade >> {}'.format(label_name))
      #print('Similaridade não encontrada')
      print('')

    return label_name

  def index_em_nome(self,base_treino,ind):
    '''
      Para recebe uma lista com a numeração do index da base de teste e retorna uma lista com os nomes dos respectivos posições.

      :parametro base_treino: lista com caracteristicas, identificador e diretorio dos indivifuo de treino.
      :parametro ind: lista com os numeros de index dos identificadores.
      :retorna uma lista com os nomes dos respectivos posições.
    '''
    lista_nomes = []
    for index in ind:
      lista_nomes.append(base_treino[0][int(index)])

    return lista_nomes

  def moda_nomes(self,lista_nomes):
    '''
      Para a operação de moda na lista de nomes e retorna ou o nome que mais ocorre ou a primeira ocorrencia.

      :parametro lista_nomes: lista com o nomes dos identicadores podem classificar o imagem de teste.
      :retorna ou o nome que mais ocorre ou a primeira ocorrencia.
    '''
    quantidade_de_vezes_que_aparece = []
    for nome in lista_nomes:
      quantidade_de_vezes_que_aparece.append(0)
      for nome_2 in lista_nomes:
        if nome == nome_2:
          quantidade_de_vezes_que_aparece[-1] += 1
      quantidade_de_vezes_que_aparece[-1] -= 1

    maior_quantidade = 1
    posicao_maior = -1
    flag = 0
    for cont in range(0,len(quantidade_de_vezes_que_aparece)):
      if(quantidade_de_vezes_que_aparece[cont] > maior_quantidade):
        maior_quantidade = quantidade_de_vezes_que_aparece[cont]
        posicao_maior = cont
        flag = 1

    if(posicao_maior == -1):
      #print(lista_nomes)
      #print(lista_nomes[0])
      #print(posicao_maior)
      #print('')
      return lista_nomes[0]
    else:
      #print(lista_nomes)
      #print(lista_nomes[posicao_maior])
      #print(posicao_maior)
      #print('')
      return lista_nomes[posicao_maior]


  def BallTree_classificacao_individual(self,individuo_teste,base_treino):
    '''
      Para identificar o individuo teste por meio da comparação com os individuos do banco de treino.

      :parametro individuo_teste: caracteristicas do individuo que queremos classificar.
      :parametro base_treino: base com os individuos de treino.
      :retorna o primeiro individuo na lista de identificação feita pelo BallTree.
    '''
    novo_treino = np.array(base_treino[1])
    novo_individuo = [np.array(individuo_teste)]

    tree = BallTree(novo_treino, leaf_size=2)              # doctest: +SKIP
    dist, ind = tree.query(novo_individuo, k=10)

    lista_nomes = self.index_em_nome(base_treino,ind[0])
    moda_lista = self.moda_nomes(lista_nomes)

    #return moda_lista
    return lista_nomes[0]

  def resultados_do_reconhecimento_de_uma_foto(self,base_treino,image_path):

    individuo_teste = self.extracao_de_caracteristicas_1_foto(image_path)
    label_name = self.BallTree_classificacao_individual(individuo_teste,base_treino)
    return label_name

  def resultados_da_classificacao(self,base_treino,base_teste):
    '''
      Para realizar a identificação da base de teste com atravez na base de treino.

      :parametro base_treino: base com identificador, caracteristicas e diretorio dos individuos utilizados para treino.
      :parametro base_teste: base com identificador, caracteristicas e diretorio dos individuos utilizados para teste.
      :retorna os resultados da classificação feita.
    '''

    #base_teste = [numero_individuo_teste,caracteristicas_individuo_teste]

    classificacao = []

    for individuo_teste in base_teste[1]:
      label_name = self.BallTree_classificacao_individual(individuo_teste,base_treino)
      #label_name = self.classificacao_individuo(individuo_teste,base_treino)
      classificacao.append(label_name)

    return classificacao

  def resultado_dos_teste_percentual_acertos(self,classificacao,base_teste):
    '''
      Para realizar o calculo do percentual de acerto comparando os identificadores feitos pelo classificador e os identificadores 
      fornecidos pela base de teste.

      :parametro classificacao: lista com os identificadores classificados pelo classificado BallTree.
      :parametro base_teste: base com identificador, caracteristicas e diretorio dos individuos utilizados para teste.
      :retorna o percentual de acerto comparando os identificadores classificados e os identificadores esperados.
    '''
    acertos = 0
    erros = 0
    print('predicao >> correto')
    for conte in range(0,len(base_teste[0])):
      if(classificacao[conte] == base_teste[0][conte]):
        acertos += 1
      else:
        print('{} >> {}'.format(classificacao[conte], base_teste[0][conte]))
        erros += 1
        #print('Foto de teste:')
        #self.mostrar_uma_imagem(base_teste[2][conte])
        #print('Classificação feita:')
        #print('{}'.format(classificacao[conte]))
        print('')
    print('acertos : {}'.format(acertos))
    print('erros : {}'.format(erros))
    return (acertos/(acertos+erros))

  def mostrar_uma_imagem(self,diretorio):
    '''
      Para realizar o calculo do percentual de acerto comparando os identificadores feitos pelo classificador e os identificadores 
      fornecidos pela base de teste.

      :parametro classificacao: lista com os identificadores classificados pelo classificado BallTree.
      :parametro base_teste: base com identificador, caracteristicas e diretorio dos individuos utilizados para teste.
      :retorna o percentual de acerto comparando os identificadores classificados e os identificadores esperados.
    '''
    image = cv2.imread(diretorio)
    plt.imshow(image)
    plt.show()
    return True

  def adicionando_individuos_a_base_treino(self,base_treino,individuo_path):
    '''
      Para realizar a adiçãod e mais um individuo no banco de dados.

      :parametro base_treino: base de dados.
      :parametro individuo_path: diretorio para o novo individuo adicionado.
    '''
    #------------------------

    #coloque as fotos de seus funcionários neste caminho como name_of_employee.jpg 
    #diretorio_individuo = "/content/drive/MyDrive/Reconhecimento Facial/individuo/"

    individuo = individuo_path.split('/')[-2]

    for foto in listdir(individuo_path):
        base_treino[1].append(self.extracao_de_caracteristicas_1_foto(individuo_path + '/' + foto))
        base_treino[0].append(individuo)
        base_treino[2].append(individuo_path + '/' + foto)

  def remover_individuos_a_base_treino(self,base_treino,identificador_do_individuo_indesejado):
    '''
      Para realizar a remoção do individuo identificado no parametro identificador_do_individuo_indesejado.

      :parametro base_treino: base de dados.
      :parametro identificador_do_individuo_indesejado: identificador do individuo que deseja remover.
    '''
    #------------------------

    #coloque as fotos de seus funcionários neste caminho como name_of_employee.jpg 
    #diretorio_individuo = "/content/drive/MyDrive/Reconhecimento Facial/individuo/"
    flag = 0
    while flag == 0 :
      flag = 1
      for index in range(0,len(base_treino[1])):
        if(base_treino[0][index] == identificador_do_individuo_indesejado):
          base_treino[0].pop(index)
          base_treino[1].pop(index)
          base_treino[2].pop(index)
          flag = 0
          break

