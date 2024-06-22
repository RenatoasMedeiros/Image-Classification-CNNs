# FOLDER STRUTURE

# dataset
Datasets usados para teste, treino e validação dos modelos. 

## Webapp
Construímos uma aplicação web em flask, para testar os modelos com imagens diferentes das usadas para treino, validação e teste.

## Models
Pasta onde estão guardados os modelos treinados durante o desenvolver do projeto.

## Logs
Ficheiros csv criados durante o treino que indicam os resultados por epoch na validação (NÃO TESTE).

## Plots
Graficos gerados no fim de cada modelo ser treinado com sucesso.

## Features & labels
Features e labels dos modelos desenvolvidos usando transfer learning sem data augmentation

## PDFs
Contém todos os pdfs dos ficheiros .ipynb com markdown

## old_tests
Contém uma série de testes que foram realizados no desenvolver deste projeto

## plots
Contém plots que foram gerados no treino dos modelos

NOTA:
Existem vários ficheiros ".py" pois facilitava o treino de vários modelos seguidamente sem interação nossa, isso atravez do script "run_all_scripts.py" que criamos, esse script encontra-se no folder "scripts"
