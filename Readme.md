
# Relatório do Projeto: Sistema de Recomendação com SASRec

Este projeto implementa um sistema de recomendação baseado no modelo **SASRec** (Self-Attentive Sequential Recommendation), que utiliza mecanismos de atenção para capturar padrões sequenciais no comportamento do usuário. O modelo foi adaptado a partir da implementação disponível em [SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch).

## Visão Geral

O objetivo do projeto é criar um sistema de recomendação que sugere itens (filmes, no caso do dataset MovieLens) com base no histórico de interações do usuário. O modelo SASRec é particularmente eficaz para capturar dependências de longo prazo em sequências de interações, utilizando camadas de atenção multi-head e feed-forward.

### Fluxo de Trabalho

1. **Carregamento dos Dados**: Os dados são carregados a partir do dataset MovieLens, que contém interações de usuários com filmes (avaliações).
2. **Pré-processamento**: Os dados são organizados em sequências temporais para cada usuário.
3. **Treinamento do Modelo**: O modelo SASRec é treinado para prever o próximo item na sequência de interações do usuário.
4. **Avaliação**: O modelo é avaliado usando métricas como `nDCG@K` e `HR@K`.
5. **Geração de Recomendações**: Após o treinamento, o modelo pode ser usado para gerar recomendações personalizadas para cada usuário.

---

## Estrutura do Projeto

O projeto é organizado em vários arquivos Python, cada um com uma responsabilidade específica. Abaixo está uma descrição detalhada de cada arquivo:

### 1. `dataset.py`

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

### 2. `main.py`

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

### 3. `evaluate.py`

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

### 4. `load_data.py`

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

### 5. `model.py`

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

---

### 6. `point_wise_feed_foward.py`

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

1. **nDCG@K (Normalized Discounted Cumulative Gain)**:
   - Mede a qualidade da recomendação, considerando a posição dos itens relevantes na lista de recomendações.
   - Quanto maior, melhor.

2. **HR@K (Hit Rate)**:
   - Mede a proporção de usuários para os quais pelo menos um item relevante está na lista de recomendações.
   - Quanto maior, melhor.

---

## Conclusão

Este projeto implementa um sistema de recomendação baseado no modelo SASRec, que utiliza mecanismos de atenção para capturar padrões sequenciais no comportamento do usuário. O código é modular e bem organizado, facilitando a extensão e manutenção. As métricas de avaliação indicam que o modelo é eficaz na geração de recomendações personalizadas.


