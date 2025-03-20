import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MovielensDataset
from src.model.evaluate import evaluate, evaluate_simple
from src.model.load_data import load_data_from_df
from src.model.model import SASRec
from src.loader.loader import Loader
from src.utils.enums import MovieLensDataset, MovieLensType
import src.utils.defaults as d
from tqdm import tqdm
from src.utils.logger import Logger

logger = Logger.get_logger("SASRec")


def plot_and_save(x: list, y: list, title: str, filename: str) -> None:
    plt.plot(x, y)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()


def train_and_evaluate() -> None:
    """
    Executa o treinamento e a avaliação do modelo SASRec,
    calculando as métricas nDCG@K, HR@K, MAP@K, Precision@K e Recall@K.
    Registra os valores no TensorBoard e via logger.
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = os.cpu_count()
    evaluate_type = "simple"

    # Configura o TensorBoard
    writer = SummaryWriter(log_dir="../.tensorboard_logs")

    # Carrega os dados
    ratings_df = Loader().load_pandas(dataset=MovieLensDataset.ML_20M, ml_type=MovieLensType.RATINGS)
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

    # Listas para armazenar as métricas ao longo das épocas
    epoch_list = []
    ndcg_list = []
    hr_list = []

    global_step = 0
    os.makedirs("./saved_results", exist_ok=True)
    with open(f"./saved_results/result0.txt", "w") as results_file:

        for epoch in range(1, d.num_epochs + 1):
            epoch_loss = 0.0
            num_batches = 0
            # Cria uma barra de progresso para a época
            for userid, seq, pos, neg in tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False):
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

                epoch_loss += loss.item()
                num_batches += 1
                writer.add_scalar("Loss/train", loss.item(), global_step)
                global_step += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"Epoch: {epoch} - Avg Loss: {avg_loss:.4f}")

            if epoch % 1 == 0:
                model.eval()
                epoch_list.append(epoch)
                avg_ndcg, avg_hr = evaluate_simple(
                    model, [train_data, valid_data, test_data, user_num, item_num], d.sequence_length
                )
                ndcg_list.append(avg_ndcg)
                hr_list.append(avg_hr)
                logger.info(
                    f"(Train)    Epoch: {epoch}, nDCG@K: {avg_ndcg:.4f}, HR@K: {avg_hr:.4f}, Loss: {loss.item():.4f}\n"
                )
                results_file.write(
                    f"(Train)    Epoch: {epoch}, nDCG@K: {avg_ndcg:.4f}, HR@K: {avg_hr:.4f}, Loss: {loss.item():.4f}\n"
                )
                results_file.flush()

                writer.add_scalar("nDCG/val", avg_ndcg, epoch)
                writer.add_scalar("HR/val", avg_hr, epoch)

                # Avaliação adicional (opcional) utilizando o conjunto de validação
                val_ndcg, val_hr = evaluate_simple(
                    model,
                    [train_data, valid_data, test_data, user_num, item_num],
                    d.sequence_length,
                    isvalid=True
                )
                logger.info(
                    f"(Validate) Epoch: {epoch}, nDCG@K: {val_ndcg:.4f}, HR@K: {val_hr:.4f}, Loss: {loss.item():.4f}\n"
                )
                model.train()

    # Salva os gráficos para cada métrica
    plot_and_save(epoch_list, ndcg_list, "nDCG@K", "./saved_results/nDCG.png")
    plot_and_save(epoch_list, hr_list, "HR@K", "./saved_results/HR.png")
    logger.info("Finished....")
    writer.close()
    plt.close()


if __name__ == "__main__":
    train_and_evaluate()
