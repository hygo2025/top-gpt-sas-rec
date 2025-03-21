
# Relatório do Projeto: Sistema de Recomendação com SASRec

Este projeto implementa um sistema de recomendação baseado no modelo **SASRec** (Self-Attentive Sequential Recommendation), que utiliza mecanismos de atenção para capturar padrões sequenciais no comportamento do usuário. O modelo foi adaptado a partir da implementação disponível em [SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch).

# Índice

1. [Visão Geral](#visão-geral)
   - [Objetivo do Projeto](#objetivo-do-projeto)
   - [Fluxo de Trabalho](#fluxo-de-trabalho)
2. [Como Executar o Código](#como-executar-o-código)
   - [Rodar via Google Colab](#rodar-via-google-colab)
   - [Rodar localmente](#rodar-localmente)
3. [Estrutura do Projeto](#estrutura-do-projeto)
   - [dataset.py](#datasetpy)
   - [main.py](#mainpy)
   - [evaluate.py](#evaluatepy)
   - [load_data.py](#load_datapy)
   - [model.py](#modelpy)
     - [SASRec Neural Network Architecture](#sasrec-neural-network-architecture)
   - [point_wise_feed_foward.py](#point_wise_feed_fowardpy)
4. [Métricas de Avaliação](#métricas-de-avaliação)
   - [nDCG@K](#ndcgk-normalized-discounted-cumulative-gain)
   - [HR@K](#hrk-hit-rate)
5. [Resultados](#resultados)
   - [Desempenho no MovieLens 1M](#desempenho-no-movielens-1m)
   - [Desempenho no MovieLens 20M](#desempenho-no-movielens-20m)
6. [Comparação entre Modelos](#comparação-entre-modelos)
7. [Conclusão](#conclusão)

---

## Visão Geral

### Objetivo do Projeto
O objetivo do projeto é criar um sistema de recomendação que sugere itens (filmes, no caso do dataset MovieLens) com base no histórico de interações do usuário. O modelo SASRec é particularmente eficaz para capturar dependências de longo prazo em sequências de interações, utilizando camadas de atenção multi-head e feed-forward.


### Fluxo de Trabalho

1. **Carregamento dos Dados**: Os dados são carregados a partir do dataset MovieLens, que contém interações de usuários com filmes (avaliações).
2. **Pré-processamento**: Os dados são organizados em sequências temporais para cada usuário.
3. **Treinamento do Modelo**: O modelo SASRec é treinado para prever o próximo item na sequência de interações do usuário.
4. **Avaliação**: O modelo é avaliado usando métricas como `nDCG@K` e `HR@K`.
5. **Geração de Recomendações**: Após o treinamento, o modelo pode ser usado para gerar recomendações personalizadas para cada usuário.

## Como Executar o Código
O código foi desenvolvido em Python 3.9 e torch 2.8.0. Para executar o projeto, siga as etapas abaixo:


### Rodar via Google Colab
- Foram realizados dois testes o primeiro com o dataset MovieLens 1M e o segundo com o dataset MovieLens 20M.
  - Para executar o teste com o dataset MovieLens 1M, acesse o seguinte link: [MovieLens 1M - SASRec](https://colab.research.google.com/github/hygo2025/top-gpt-sas-rec/blob/main/trabalho_1m.ipynb) e execute as células sequencialmente.
  - Para executar o teste com o dataset MovieLens 32M, acesse o seguinte link: [MovieLens 20M - SASRec](https://colab.research.google.com/github/hygo2025/top-gpt-sas-rec/blob/main/trabalho_20m.ipynb) e execute as células sequencialmente.
  - Os resultados obtidos estão disponíveis no final de cada notebook.

### Rodar localmente
1. Clone o repositório:
   ```bash
   git clone https://github.com/hygo2025/top-gpt-sas-rec.git
    ```
2. Instale as dependências:
   ```bash
   make install
   ```
3. Ative o ambiente virtual:
   ```bash
   . .local/bin/activate
   ```
4. Execute o script principal:
   ```bash
   python3 src/main.py
   ```

---

## Estrutura do Projeto

O projeto é organizado em vários arquivos Python, cada um com uma responsabilidade específica. Abaixo está uma descrição detalhada de cada arquivo:

### `dataset.py`

Este arquivo define a classe `MovielensDataset`, que é responsável por carregar e preparar os dados para o treinamento do modelo.

#### Detalhes:
- **Classe `MovielensDataset`**:
  - Herda de `torch.utils.data.Dataset`.
  - Recebe os dados de treinamento, o comprimento da sequência (`sequence_length`), o número de usuários (`usernum`) e o número de itens (`itemnum`).
  - O método `__getitem__` gera sequências de itens, positivos e negativos para treinamento.
  - O método `random_neg` gera itens negativos (não interagidos) para cada usuário.

#### Exemplo de Uso:
```python
train_dataset = MovielensDataset(train_data, sequence_length=50, usernum=user_num, itemnum=item_num)
train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=os.cpu_count())
```

---

### `main.py`

Este é o ponto de entrada do projeto. Ele orquestra o carregamento dos dados, o treinamento do modelo e a avaliação.

#### Detalhes:
- **Função `train_and_evaluate`**:
  - Carrega os dados do MovieLens.
  - Cria o dataset e o DataLoader.
  - Instancia o modelo SASRec.
  - Define a função de perda (`BCEWithLogitsLoss`) e o otimizador (`Adam`).
  - Executa o treinamento e avaliação em loop.
  - Salva métricas e gráficos de desempenho.

#### Exemplo de Uso:
```python
if __name__ == "__main__":
    train_and_evaluate(dataset=MovieLensDataset.ML_1M)
```

---

### `evaluate.py`

Este arquivo contém a lógica para avaliar o modelo durante o treinamento.

#### Detalhes:
- **Função `evaluate`**:
  - Calcula as métricas `nDCG@K` e `HR@K` para o conjunto de treino ou validação.
  - Utiliza batches para processar os usuários de forma eficiente.
  - Retorna as métricas médias para o batch atual.

#### Exemplo de Uso:
```python
avg_ndcg, avg_hr = evaluate(model, [train_data, valid_data, test_data, user_num, item_num], sequence_length=50)
```

---

### `load_data.py`

Este arquivo é responsável por carregar e pré-processar os dados do MovieLens.

#### Detalhes:
- **Função `load_data_from_df`**:
  - Recebe um DataFrame do MovieLens e divide os dados em conjuntos de treino, validação e teste.
  - Para cada usuário, a sequência de interações é dividida em:
    - Treino: Todos os itens, exceto os dois últimos.
    - Validação: O penúltimo item.
    - Teste: O último item.

#### Exemplo de Uso:
```python
train_data, valid_data, test_data, user_num, item_num = load_data_from_df(ratings_df)
```

---

### `model.py`

Este arquivo define a arquitetura do modelo SASRec.

#### Detalhes:
- **Classe `SASRec`**:
  - **Camadas**:
    - Embeddings de itens e posições.
    - Blocos de atenção multi-head e feed-forward.
    - Dropout para regularização.
  - **Métodos**:
    - `encode_sequence`: Codifica a sequência de itens usando embeddings e camadas de atenção.
    - `forward`: Realiza o forward pass durante o treinamento.
    - `predict`: Gera previsões para inferência.
    - `get_num_params`: Retorna o número de parâmetros do modelo.

#### Exemplo de Uso:
```python
model = SASRec(
    user_num=user_num,
    item_num=item_num,
    hidden_units=64,
    dropout_rate=0.1,
    sequence_length=50,
    num_of_blocks=2,
    num_of_heads=2
)
```

#### SASRec Neural Network Architecture

----------------------------------
          
    ┌─────────────────────────────┐
    │ Input Sequence              │
    ├─────────────────────────────┤
    │ Item Embedding (3953,128)   │  <- Embedding Layer
    │ Position Embedding (50,128) │
    └─────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────┐
    │ LayerNorm (128)             │  <- Layer Normalization for Attention
    └─────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────┐
    │ Multi-Head Attention (x2)   │  <- Self-Attention Layers
    │ Linear Projection (128)     │
    └─────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────┐
    │ LayerNorm (128)             │  <- Layer Normalization for FFN
    └─────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────┐
    │ PointWise FeedForward (x2)  │  <- Feed Forward Layers
    │ Conv1D (128 → 128)          │
    │ ReLU Activation             │
    │ Dropout (p=0.2)             │
    │ Conv1D (128 → 128)          │
    │ Dropout (p=0.2)             │
    └─────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────┐
    │ Dropout (p=0.2)             │  <- Final Regularization Layer
    └─────────────────────────────┘
                │
                ▼
          Output Prediction

---

### `point_wise_feed_foward.py`

Este arquivo define a camada feed-forward usada no modelo SASRec.

#### Detalhes:
- **Classe `PointWiseFeedForward`**:
  - Implementa uma camada feed-forward com convoluções 1D e ativação ReLU.
  - Utiliza dropout para regularização.

#### Exemplo de Uso:
```python
ffn_layer = PointWiseFeedForward(hidden_units=64, dropout_rate=0.1)
```

---

## Métricas de Avaliação

O modelo é avaliado usando as seguintes métricas:

#### **nDCG@K (Normalized Discounted Cumulative Gain)**:
   - Mede a qualidade da recomendação, considerando a posição dos itens relevantes na lista de recomendações.
   - Quanto maior, melhor.

#### **HR@K (Hit Rate)**:
   - Mede a proporção de usuários para os quais pelo menos um item relevante está na lista de recomendações.
   - Quanto maior, melhor.

---
## Resultados

O desempenho do modelo SASRec foi avaliado no dataset **MovieLens 1M** com 710 mil parâmetros, utilizando as métricas **HR@K** e **nDCG@K**, tanto no conjunto de **treinamento** quanto no de **validação**. Além disso, a função de perda foi monitorada ao longo das épocas para avaliar a convergência do modelo.

### Desempenho no MovieLens 1M
#### 1. Evolução das Métricas ao Longo do Treinamento

Os gráficos a seguir mostram a evolução das métricas **HR@K** e **nDCG@K** durante o treinamento e validação:

- **HR@K (Hit Rate)**:
  - No conjunto de **treino**, o modelo começou com um **HR@K ≈ 0.47** e atingiu **~0.80** após 320 épocas.
  - No conjunto de **validação**, o desempenho seguiu uma trajetória semelhante, alcançando um **HR@K de ~0.81**.

  ![HR@K (Treinamento)](/results_1m/HR_train.png)
  ![HR@K (Validação)](/results_1m/HR_val.png)

- **nDCG@K (Normalized Discounted Cumulative Gain)**:
  - O modelo começou com um **nDCG@K ≈ 0.25** e convergiu para **~0.58** no treino.
  - No conjunto de **validação**, os valores finais foram semelhantes, atingindo **~0.60**.

  ![nDCG@K (Treinamento)](/results_1m/nDCG_train.png)
  ![nDCG@K (Validação)](/results_1m/nDCG_val.png)

### 2. Análise da Função de Perda

A função de perda foi monitorada ao longo do treinamento. Inicialmente, a perda estava **acima de 1.0**, mas foi reduzida de forma consistente até estabilizar em torno de **0.25 - 0.30**.

Isso indica que o modelo convergiu adequadamente, sem sinais evidentes de **overfitting**, já que as métricas no conjunto de validação mantiveram um comportamento estável próximo às do treino.

### 3. Convergência do Modelo

Observamos que o modelo apresentou uma **rápida melhora nas primeiras 50 épocas**, com um aumento significativo nas métricas de desempenho. A partir da **época 150**, o crescimento se tornou mais estável, com pequenas oscilações naturais.

Os valores de **HR@K e nDCG@K estabilizaram entre as épocas 250 e 320**, sugerindo que mais treinamento não traria ganhos significativos.

### 4. Conclusão Parcial (MovieLens 1M)

- O **SASRec conseguiu capturar os padrões de interação dos usuários de forma eficiente**, com desempenho satisfatório tanto em HR@K quanto em nDCG@K.
- As métricas de **treino e validação foram consistentes**, sem indícios significativos de overfitting.
- O modelo convergiu após **~250 épocas**, tornando esse um ponto adequado para interrupção do treinamento caso seja necessário economizar tempo computacional.

Na próxima seção, realizaremos os mesmos testes com o dataset **MovieLens 20M** para avaliar a escalabilidade do modelo em um conjunto de dados maior.


## Desempenho no MovieLens 20M

### Evolução das Métricas ao Longo do Treinamento

Os gráficos a seguir mostram a evolução das métricas **HR@K** e **nDCG@K** durante o treinamento e validação:

- **HR@K (Hit Rate)**:
  - No conjunto de **treino**, o modelo começou com um **HR@K ≈ 0.9915** e atingiu **~0.9970** após 50 épocas.
  - No conjunto de **validação**, o desempenho foi semelhante, alcançando **HR@K ~0.9971**.

  ![HR@K (Treinamento)](/results_20m/HR_train.png)
  ![HR@K (Validação)](/results_20m/HR_val.png)

- **nDCG@K (Normalized Discounted Cumulative Gain)**:
  - O modelo começou com um **nDCG@K ≈ 0.8185** e convergiu para **~0.9182** no treino.
  - No conjunto de **validação**, os valores finais foram **~0.9179**.

  ![nDCG@K (Treinamento)](/results_20m/nDCG_train.png)
  ![nDCG@K (Validação)](/results_20m/nDCG_val.png)

### Análise da Função de Perda

A função de perda foi monitorada ao longo do treinamento. Inicialmente, a perda estava **em 0.1545**, reduzindo consistentemente e estabilizando em torno de **0.0652** após 50 épocas.

Isso indica que o modelo **convergiu rapidamente** e manteve um desempenho estável, sem indícios significativos de overfitting.

### Convergência do Modelo

Observamos que o modelo apresentou uma **melhora significativa nas primeiras 10 épocas**, com um aumento expressivo nas métricas de desempenho. A partir da **época 20**, o crescimento se tornou mais estável, com pequenas oscilações naturais.

Os valores de **HR@K e nDCG@K estabilizaram entre as épocas 40 e 50**, sugerindo que mais treinamento não traria ganhos significativos.

## Comparação entre Modelos

| Dataset       | HR@K (Final) | nDCG@K (Final) | Loss (Final)  |
|---------------|--------------|----------------|---------------|
| MovieLens 1M  | **0.81**     | **0.60**       | **0.25-0.30** |
| MovieLens 20M | **0.9970**   | **0.9182**     | **0.0652**    |

O desempenho no **MovieLens 20M** foi significativamente superior ao **MovieLens 1M**, o que sugere que o modelo **aprendeu melhor os padrões dos usuários** em um dataset maior, possivelmente devido à maior diversidade e riqueza de interações.

## Conclusão

- O **SASRec apresentou uma melhora expressiva no MovieLens 20M**, atingindo valores muito altos de **HR@K** e **nDCG@K**.
- O modelo convergiu **rapidamente (em ~50 épocas)**, sendo um indicativo de que não há necessidade de treinamentos longos.
- O desempenho superior pode ser atribuído ao maior volume de dados, permitindo uma **melhor generalização** das interações dos usuários.

Com esses resultados, o modelo pode ser considerado **altamente eficaz** para recomendações em datasets grandes e complexos.


