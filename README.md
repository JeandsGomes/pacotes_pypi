Reconhecimento Facial com VGG faces e BallTree
=============

#### Esse é que abstrai o processo de reconhecimento facial com o modelo VGG faces e o classificador BallTree.

## Instalação:

    '
    pip install filterRomuere
    '

## Uso:

### Importando a biblioteca

    '''
    import Face_Recognition
    '''

### Inicializando o modelo
#### é nescessario o caminho para o arquivo vgg_face_weights.h3
#### você pode baixar o arquivo nesse link [Google](https://drive.google.com/file/d/1yMLtU3tcfp7hVwoMt6JIPnG0MvXVwTEs/view?usp=sharing)

    '''
    r_f = Face_Recognition('.../Reconhecimento_Facial/vgg_face_weights.h5')
    r_f.criando_modelo()
    '''

### Formatar o caminho do diretorio para importar o banco de dados
#### A função dicionario_diretorio_treino_teste ira formatar os caminhos para as fotos presentes no banco de dados.
#### Os parametros são os seguintes: 
1. diretorio : caminho para o banco de dados formatado da seguinte forma (.../base_de_dados/) 
2. quantidade_de_fotos_para_treino : quantidade de fotos armazenadas no banoco de treino.
3. quantidade_de_fotos_para_teste : quantidade de fotos armazenadas no banoco de teste.
    
    '''
    diretorio = '.../base_de_dados/'
    dicionario_treino_teste = r_f.dicionario_diretorio_treino_teste(diretorio,quantidade_de_fotos_para_treino,quantidade_de_fotos_para_teste)
    '''

### Formatar o banco de dados
#### A função extracao_de_caracteristicas_diretorio_treino vai retornar uma lista com as caracteristicas das fotos do banco de treino.
#### Os parametros são os seguintes: 
1. dicionario_treino_teste : resultado retornado da função dicionario_diretorio_treino_teste

    '''
    base_treino = r_f.extracao_de_caracteristicas_diretorio_treino(dicionario_treino_teste)
    '''

#### A função extracao_de_caracteristicas_diretorio_teste vai retornar uma lista com as caracteristicas das fotos do banco de teste.
#### Os parametros são os seguintes: 
1. dicionario_treino_teste : resultado retornado da função dicionario_diretorio_treino_teste

    '''
    base_teste = r_f.extracao_de_caracteristicas_diretorio_teste(dicionario_treino_teste)
    '''

### Classificar o banco de dados teste
#### A função resultados_da_classificacao realiza a classificação do banco de dados teste.
#### Os parametros são os seguintes: 
1. base_teste : base de treino.
2. base_teste : base de teste.

    '''
    classificacao = r_f.resultados_da_classificacao(base_treino,base_teste)
    '''

### Reultados da classificação da base de dados teste
#### A função resultado_dos_teste_percentual_acertos mostra resultados da classificação do banco teste.
#### Os parametros são os seguintes: 
1. classificacao : resultados da função resultados_da_classificaca.
2. base_teste : base de teste.

    '''
    r_f.resultado_dos_teste_percentual_acertos(classificacao,base_teste)
    '''

### Reconhecimneto de uma foto
#### A função resultados_do_reconhecimento_de_uma_foto realiza o reconheicmento de uma foto apenas informando o diretorio dela e fornecendo a base de dados de treino.
#### Os parametros são os seguintes: 
1. base_treino : base de teste.
2. image_path : caminho para o diretorio da imagem que deseja identificar.

    '''
    image_path = '.../5.pgm'
    r_f.resultados_do_reconhecimento_de_uma_foto(base_treino,image_path)
    '''

### Adicionar individuo a base de dados
#### A função adicionando_individuos_a_base_treino que vai adicionar um novo individuo na base de dados.
#### Os parametros são os seguintes: 
1. base_de_dados : base de dadis resultante das funções de estrações de caracteristicas.
2. diretorio_individuo_para_adicionar : caminho para o diretorio do individuo que deseja adicionar.

    '''
    individuo_para_adicionar = '.../Jeanderson/'
    r_f.adicionando_individuos_a_base_treino(base_de_dados,diretorio_individuo_para_adicionar)
    '''

### Remover individuo a base de dados
#### A função remover_individuos_a_base_treino que vai remover o individuo informado.
#### Os parametros são os seguintes: 
1. base_de_dados : base de dadis resultante das funções de estrações de caracteristicas.
2. identificador_do_individuo : identificador usado na base de dados.

    '''
    r_f.remover_individuos_a_base_treino(base_treino,identificador_do_individuo)
    '''
