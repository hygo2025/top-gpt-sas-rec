from typing import Tuple, List, Any

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm


def evaluate(
        model: Any,
        dataset: List[Any],
        sequence_length: int,
        batch_size: int = 512,  # Tamanho do batch
        isvalid: bool = False
) -> Tuple[float, float]:
    train, valid, test, usernum, itemnum = dataset

    total_ndcg = 0.0
    total_hr = 0.0
    evaluated_users = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Lista de usuários elegíveis para avaliação
    eligible_users = []
    for u in range(1, usernum + 1):
        if isvalid:
            if len(train[u]) >= 1 and len(valid[u]) >= 1:
                eligible_users.append(u)
        else:
            if len(train[u]) >= 1 and len(test[u]) >= 1:
                eligible_users.append(u)

    # Processa os usuários em batches
    for batch_start in tqdm(range(0, len(eligible_users), batch_size), desc="Evaluating Users Simple", leave=False):
        batch_end = min(batch_start + batch_size, len(eligible_users))
        batch_users = eligible_users[batch_start:batch_end]

        # Listas para armazenar dados do batch
        batch_seqs = []
        batch_candidate_items = []
        batch_ground_truths = []

        for u in batch_users:
            # Constrói a sequência de entrada
            seq = np.zeros(sequence_length, dtype=np.int32)
            idx = sequence_length - 1

            if not isvalid:
                seq[idx] = valid[u][0]
                idx -= 1

            for item in reversed(train[u]):
                seq[idx] = item
                idx -= 1
                if idx < 0:
                    break

            # Define o ground truth e os candidatos
            ground_truth = valid[u][0] if isvalid else test[u][0]
            candidate_items = [ground_truth]

            rated = set(train[u])
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                candidate_items.append(t)

            # Adiciona ao batch
            batch_seqs.append(seq)
            batch_candidate_items.append(candidate_items)
            batch_ground_truths.append(ground_truth)

        # Converte listas para tensores e move para a GPU
        batch_seqs_tensor = torch.tensor(np.array(batch_seqs), dtype=torch.int32).to(device)
        batch_candidate_items_tensor = torch.tensor(np.array(batch_candidate_items), dtype=torch.int32).to(device)

        # Realiza a predição em batch
        with torch.no_grad():
            with autocast():  # Ativa mixed precision
                batch_predictions = - model.predict(batch_seqs_tensor, batch_candidate_items_tensor)

        # Calcula as métricas para cada usuário no batch
        for i, predictions in enumerate(batch_predictions):
            rank = predictions.argsort().argsort()[0].item()  # Rank do ground truth
            if rank < 10:
                ndcg = 1.0 / np.log2(rank + 2)
                hr = 1.0
            else:
                ndcg = 0.0
                hr = 0.0

            total_ndcg += ndcg
            total_hr += hr
            evaluated_users += 1

    # Retorna as métricas médias
    if evaluated_users == 0:
        return 0.0, 0.0

    avg_ndcg = total_ndcg / evaluated_users
    avg_hr = total_hr / evaluated_users

    return avg_ndcg, avg_hr

