import numpy as np
from typing import Tuple, List, Any
from tqdm import tqdm
import numpy as np
import torch
from typing import Tuple, List, Any
from tqdm import tqdm

def evaluate(
    model: Any,
    dataset: List[Any],
    sequence_length: int,
    k: int = 10,
    isvalid: bool = False
) -> Tuple[float, float, float, float, float]:
    """
    Avalia o modelo de recomendação utilizando diversas métricas:
    - nDCG@K
    - Hit Rate (HR@K)
    - MAP@K (Mean Average Precision)
    - Precision@K
    - Recall@K

    O conjunto de dados deve ser uma lista no formato:
        [train, valid, test, usernum, itemnum]
    onde:
        - train, valid, test: dicionários mapeando o ID do usuário para uma lista de interações (itens)
        - usernum: número total de usuários
        - itemnum: número total de itens

    Args:
        model: Modelo de recomendação que possui o método predict.
        dataset (List[Any]): Lista contendo os dados [train, valid, test, usernum, itemnum].
        sequence_length (int): Comprimento fixo da sequência de entrada.
        k (int, optional): Valor de corte para as métricas (default: 10).
        isvalid (bool, optional): Se True, utiliza o conjunto de validação; caso contrário, o conjunto de teste.

    Returns:
        Tuple[float, float, float, float, float]:
            (avg_nDCG@K, avg_HR@K, avg_MAP@K, avg_Precision@K, avg_Recall@K)
    """
    train, valid, test, usernum, itemnum = dataset

    total_ndcg = 0.0
    total_hr = 0.0
    total_map = 0.0
    total_precision = 0.0
    total_recall = 0.0
    evaluated_users = 0

    # Itera sobre todos os usuários com tqdm para monitoramento
    for u in tqdm(range(1, usernum + 1), desc="Evaluating Users", leave=False):
        # Verifica se o usuário possui dados suficientes para avaliação
        if isvalid:
            if len(train[u]) < 1 or len(valid[u]) < 1:
                continue
        else:
            if len(train[u]) < 1 or len(test[u]) < 1:
                continue

        # Inicializa a sequência com zeros (padding)
        seq = np.zeros(sequence_length, dtype=np.int32)
        idx = sequence_length - 1

        # Para fase de teste, usa o primeiro item do conjunto de validação como último elemento
        if not isvalid:
            seq[idx] = valid[u][0]
            idx -= 1

        # Preenche a sequência com as interações de treino em ordem reversa
        for item in reversed(train[u]):
            seq[idx] = item
            idx -= 1
            if idx < 0:
                break

        # Seleciona o item alvo (ground truth)
        ground_truth = valid[u][0] if isvalid else test[u][0]
        candidate_items = [ground_truth]

        # Cria um conjunto com os itens já vistos no treino para evitar repetições
        rated = set(train[u])
        # Em vez de usar um laço while, utiliza np.random.choice para amostrar 100 negativos
        all_items = set(range(1, itemnum + 1))
        negatives = list(all_items - rated)
        if len(negatives) < 100:
            # Se não houver negativos suficientes, pula esse usuário
            continue
        sampled_negatives = np.random.choice(negatives, size=100, replace=False).tolist()
        candidate_items.extend(sampled_negatives)

        # Realiza a predição; o modelo espera inputs em formato batch
        predictions = - model.predict(np.array([seq]), np.array(candidate_items))
        predictions = predictions[0]

        # Mapeia os scores para ranks
        rank = predictions.argsort().argsort()[0].item()

        # Calcula as métricas para o usuário considerando o cutoff k.
        if rank < k:
            ndcg = 1.0 / np.log2(rank + 2)  # rank é zero-indexado
            hr = 1.0
            average_precision = 1.0 / (rank + 1)
            precision = 1.0 / k
            recall = 1.0  # Como há um único item relevante
        else:
            ndcg = 0.0
            hr = 0.0
            average_precision = 0.0
            precision = 0.0
            recall = 0.0

        total_ndcg += ndcg
        total_hr += hr
        total_map += average_precision
        total_precision += precision
        total_recall += recall
        evaluated_users += 1

    # Se nenhum usuário foi avaliado, retorna zeros para todas as métricas
    if evaluated_users == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    avg_ndcg = total_ndcg / evaluated_users
    avg_hr = total_hr / evaluated_users
    avg_map = total_map / evaluated_users
    avg_precision = total_precision / evaluated_users
    avg_recall = total_recall / evaluated_users

    return avg_ndcg, avg_hr, avg_map, avg_precision, avg_recall
