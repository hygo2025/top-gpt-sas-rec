import numpy as np
import torch
from typing import Tuple, List, Any
from tqdm import tqdm

from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalMAP, RetrievalPrecision, RetrievalRecall

def evaluate_torch(
    model: Any,
    dataset: List[Any],
    sequence_length: int,
    k: int = 10,
    isvalid: bool = False,
    batch_size: int = 128
) -> Tuple[float, float, float, float, float]:
    """
    Avalia o modelo de recomendação utilizando métricas de recuperação:
      - nDCG@K
      - Hit Rate (HR@K)
      - MAP@K (Mean Average Precision)
      - Precision@K
      - Recall@K

    Essa função é vetorizada para operar em batches na GPU e utiliza as classes do TorchMetrics.

    Args:
        model: Modelo de recomendação com método predict.
        dataset (List[Any]): [train, valid, test, usernum, itemnum]
        sequence_length (int): Comprimento fixo da sequência.
        k (int, optional): Cutoff para as métricas (default: 10).
        isvalid (bool, optional): Se True, utiliza o conjunto de validação; caso contrário, o de teste.
        batch_size (int, optional): Tamanho do batch para predição.

    Returns:
        Tuple[float, float, float, float, float]:
            (avg_nDCG@K, avg_HR@K, avg_MAP@K, avg_Precision@K, avg_Recall@K)
    """
    train, valid, test, usernum, itemnum = dataset

    seq_list = []
    cand_list = []

    # Constrói as sequências e candidatos para todos os usuários elegíveis
    for u in tqdm(range(1, usernum + 1), desc="Building evaluation batches", leave=False):
        # Verifica se o usuário possui dados suficientes
        if isvalid:
            if len(train[u]) < 1 or len(valid[u]) < 1:
                continue
        else:
            if len(train[u]) < 1 or len(test[u]) < 1:
                continue

        seq = np.zeros(sequence_length, dtype=np.int32)
        idx = sequence_length - 1

        # Para a fase de teste, usa o primeiro item do conjunto de validação como último elemento
        if not isvalid:
            seq[idx] = valid[u][0]
            idx -= 1

        # Preenche a sequência com as interações de treino (mais recentes primeiro)
        for item in reversed(train[u]):
            seq[idx] = item
            idx -= 1
            if idx < 0:
                break

        ground_truth = valid[u][0] if isvalid else test[u][0]
        candidate_items = [ground_truth]

        # Gera 100 negativos de forma vetorizada usando np.random.choice
        rated = set(train[u])
        all_items = set(range(1, itemnum + 1))
        negatives = list(all_items - rated)
        if len(negatives) < 100:
            continue
        sampled_negatives = np.random.choice(negatives, size=100, replace=False).tolist()
        candidate_items.extend(sampled_negatives)

        seq_list.append(seq)
        cand_list.append(candidate_items)

    if len(seq_list) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Converte as listas para arrays e, depois, para tensores no dispositivo do modelo
    sequences_np = np.stack(seq_list, axis=0)      # (N, sequence_length)
    candidates_np = np.stack(cand_list, axis=0)      # (N, 101)
    device = model.item2emb.weight.device
    sequences_tensor = torch.tensor(sequences_np, dtype=torch.long, device=device)
    candidates_tensor = torch.tensor(candidates_np, dtype=torch.long, device=device)

    # Realiza as predições em batches
    all_preds = []
    num_samples = sequences_tensor.size(0)
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Predicting batches", leave=False):
            batch_seq = sequences_tensor[i : i + batch_size]
            batch_cand = candidates_tensor[i : i + batch_size]
            batch_preds = - model.predict(batch_seq, batch_cand)  # (B, 101)
            all_preds.append(batch_preds)
    preds_tensor = torch.cat(all_preds, dim=0)  # (N, 101)

    # Cria o tensor target: 1 para o ground truth (primeiro candidato) e 0 para os demais
    target = torch.zeros_like(preds_tensor, dtype=torch.int)
    target[:, 0] = 1

    # Instancia as métricas do TorchMetrics com cutoff k
    ndcg_metric = RetrievalNormalizedDCG(top_k=k)
    map_metric = RetrievalMAP(top_k=k)
    precision_metric = RetrievalPrecision(top_k=k)
    recall_metric = RetrievalRecall(top_k=k)

    indexes = torch.arange(preds_tensor.size(0), device=preds_tensor.device).unsqueeze(1).expand(-1,
                                                                                                 preds_tensor.size(1))

    # Calcula as métricas utilizando os tensores (supondo que os tensores estejam no mesmo dispositivo e no formato esperado)
    avg_ndcg = ndcg_metric(preds_tensor, target, indexes)
    avg_map = map_metric(preds_tensor, target, indexes)
    avg_precision = precision_metric(preds_tensor, target, indexes)
    avg_recall = recall_metric(preds_tensor, target, indexes)

    # Calcula o Hit Rate (HR@K): se o ground truth estiver entre os top-k preditos
    sorted_preds = torch.argsort(preds_tensor, dim=1, descending=True)
    hits = (sorted_preds[:, :k] == 0).any(dim=1).float()
    avg_hr = hits.mean()

    return avg_ndcg.item(), avg_hr.item(), avg_map.item(), avg_precision.item(), avg_recall.item()



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


def evaluate(model, dataset, sequence_length, isvalid=False):
    [train, valid, test, usernum, itemnum] = dataset

    NDCG = 0.0
    HT = 0.0
    evaluate_usernum = 0.0
    users = range(1, usernum + 1)

    for u in users:
        if isvalid and (len(train[u]) < 1 or len(valid[u]) < 1):
            continue
        elif not isvalid and (len(train[u]) < 1 or len(test[u]) < 1):
            continue

        seq = np.zeros([sequence_length], dtype=np.int32)
        idx = sequence_length - 1 #取出最后一个位置

        if not isvalid:
            seq[idx] = valid[u][0]
            idx -= 1

        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        if isvalid:
            item_idx = [valid[u][0]]
        else:
            item_idx = [test[u][0]]

        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = - model.predict(np.array([seq]), np.array(item_idx))
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0].item()

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        evaluate_usernum += 1
    return NDCG / evaluate_usernum, HT / evaluate_usernum
