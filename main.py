# Passo a Passo
    # Passo 0: Entender o desafio da empresa
    # Passo 1: Importar a Base de Dados e corrigir os erros
    # Passo 2: Preparar a base de dados para a IA
    # Passo 3: Criar um modelo de IA -> Score de Credito: Ruim, Médio, Bom
    # Passo 4: Escolher o melhor modelo
    # Passo 5: Usar a IA para fazer as novas previsões

# Bibliotecas usadas
    # Pandas -> pip install pandas numpy openpyxl
    # Scikit-learn -> pip install scikit-learn

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

baseDeDados = pd.read_csv('clientes.csv')

print(baseDeDados)
print('')

print(baseDeDados.info())
print('')

# Precisamos alterar os tipos de dados das colunas para numeros para que a IA consiga fazer a leitura
    # profissao
        # LabelEncoder - Codificando um texto, exemplo:
            # engenheiro = 1
            # cientista = 2
            # advogado = 3
            # ator = 4
            # mecanico = 5
    # mix_credito
    # comportamento_pagamento

# Porque não alteramos a coluna score_credito? Ela é o nosso gabarito, é ela que queremos prever com a IA

codificador = LabelEncoder() # Troca os textos por numeros

baseDeDados['profissao'] = codificador.fit_transform(baseDeDados['profissao'])
baseDeDados['mix_credito'] = codificador.fit_transform(baseDeDados['mix_credito'])
baseDeDados['comportamento_pagamento'] = codificador.fit_transform(baseDeDados['comportamento_pagamento'])

print(baseDeDados.info())
print('')

# Aprendizado de Máquina (Machine Learning)

# A IA precisa de:
    # Dados de Treino
    # Dados de Teste
    # y é a coluna que eu quero prever
    # x são as colunas que vou usar para fazer a previsão
        # Não vou usar a coluna id_cliente porque ela é algo irrelevante

y = baseDeDados['score_credito']
x = baseDeDados.drop(columns=['score_credito', 'id_cliente'])

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)
# train_test_split é uma função que faz a separação da base de dados em dados de teste e dados de treino

# Criar IA
    # Modelos:
        # Árvore de Decisão - RandomForest
            # A IA vai fazendo perguntas relacionadas a base de dados para montar uma previsão do perfil do cliente por exemplo
        # KNN - Vizinhos Proximos - Kneighbors
    # Treinar os modelos
    # Testar os modelos e compara os modelos


# Criando as IA's

modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

# Treinar as IA's

modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)

# Testar os Modelos

previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste.to_numpy())

# Analise a Acurácia - significa quantos porcentos a IA acertou

from sklearn.metrics import accuracy_score

print('Percentual de Testes')
print(f'Árvore de Decisão: {accuracy_score(y_teste, previsao_arvoredecisao):.1%}')
print(f'KNN: {accuracy_score(y_teste, previsao_knn):.1%}')
print('')

# Melhor Modelo:
    # A árvore de decisão é o melhor modelo, com 82.9% de acerto e o KNN com 74.7%

# Fazer novas previsões:
    # Importar os novo cliente (nova Base de Dados)
    # Codificar os clientes
    # Fazer as Previsões

baseDeDados_novosClientes = pd.read_csv('novos_clientes.csv')

print(baseDeDados_novosClientes)
print('')

print(baseDeDados_novosClientes.info())
print('')

baseDeDados_novosClientes['profissao'] = codificador.fit_transform(baseDeDados_novosClientes['profissao'])
baseDeDados_novosClientes['mix_credito'] = codificador.fit_transform(baseDeDados_novosClientes['mix_credito'])
baseDeDados_novosClientes['comportamento_pagamento'] = codificador.fit_transform(baseDeDados_novosClientes['comportamento_pagamento'])

print(baseDeDados_novosClientes.info())
print('')

# Nova previsão para os clientes
novas_previsoes = modelo_arvoredecisao.predict(baseDeDados_novosClientes)
print(f'Previsão para os novos cliente: {novas_previsoes}')