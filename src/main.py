import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MovielensDataset
from src.model.evaluate import evaluate
from src.model.load_data import load_data_from_df
from src.model.model import SASRec
from src.loader.loader import Loader
from src.utils.enums import MovieLensDataset, MovieLensType
import src.utils.defaults as d
from src.utils.logger import Logger

logger = Logger.get_logger("SASRec")


def plot_and_save(x: list, y: list, title: str, filename: str) -> None:
    """
    Plota e salva um gráfico com os dados fornecidos.
    """
    plt.plot(x, y)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()


def train_and_evaluate() -> None:
    """
    Executa o treinamento e a avaliação do modelo SASRec.
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = os.cpu_count()

    # Carrega os dados
    ratings_df = Loader().load_pandas(dataset=MovieLensDataset.ML_100K, ml_type=MovieLensType.RATINGS)
    train_data, valid_data, test_data, user_num, item_num = load_data_from_df(ratings_df)

    # Cria o dataset e o DataLoader
    train_dataset = MovielensDataset(train_data, d.sequence_length, user_num, item_num)
    train_dataloader = DataLoader(train_dataset, batch_size=d.batch_size, num_workers=num_workers)

    # Instancia o modelo e move para o dispositivo adequado
    model = SASRec(
        user_num=user_num,
        item_num=item_num,
        hidden_units=d.hidden_units,
        dropout_rate=d.dropout_rate,
        sequence_length=d.sequence_length,
        num_of_blocks=d.num_of_blocks,
        num_of_heads=d.num_heads,
    ).to(device)

    # Define a função de perda e o otimizador
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=d.lr, betas=(0.9, 0.98))

    model.train()

    epoch_list, ndcg_list, hr_list = [], [], []

    os.makedirs("./saved_results", exist_ok=True)
    with open(f"./saved_results/result0.txt", "w") as results_file:
        for epoch in range(1, d.num_epochs + 1):
            for userid, seq, pos, neg in train_dataloader:
                # Move os tensores para o dispositivo
                seq = seq.to(device)
                pos = pos.to(device)
                neg = neg.to(device)

                # Forward pass
                pos_preds, neg_preds = model(seq, pos, neg)
                pos_labels = torch.ones_like(pos_preds)
                neg_labels = torch.zeros_like(neg_preds)

                optimizer.zero_grad()

                # Cria máscara para itens de padding e calcula a loss
                real_labels_mask = (pos != 0)
                loss = criterion(pos_preds[real_labels_mask], pos_labels[real_labels_mask])
                loss += criterion(neg_preds[real_labels_mask], neg_labels[real_labels_mask])
                loss.backward()
                optimizer.step()

                logger.info(f"Epoch: {epoch} - Loss: {loss.item():.4f}")

            # Avaliação a cada 10 épocas
            if epoch % 10 == 0:
                model.eval()
                epoch_list.append(epoch)
                ndcg, hr = evaluate(model, [train_data, valid_data, test_data, user_num, item_num], d.sequence_length)
                ndcg_list.append(ndcg)
                hr_list.append(hr)
                logger.info(f"Epoch: {epoch}, NDCG: {ndcg}, HR: {hr}")
                results_file.write(f"Epoch: {epoch}, NDCG: {ndcg}, HR: {hr}, Loss: {loss.item()}\n")
                results_file.flush()

                # Avaliação adicional (opcional)
                ndcg_val, hr_val = evaluate(model, [train_data, valid_data, test_data, user_num, item_num], d.sequence_length, True)
                logger.info(f"(Validate) Epoch: {epoch}, NDCG: {ndcg_val}, HR: {hr_val}")
                model.train()

    plot_and_save(epoch_list, ndcg_list, "NDCG", "./saved_results/NDCG.png")
    plot_and_save(epoch_list, hr_list, "HR", "./saved_results/HR.png")
    logger.info("Finished....")
    plt.close()

if __name__ == "__main__":
    train_and_evaluate()
