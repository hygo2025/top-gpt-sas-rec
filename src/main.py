import os
import time
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import src.utils.defaults as d
from src.dataset import MovielensDataset
from src.loader.loader import Loader
from src.model.evaluate import evaluate
from src.model.load_data import load_data_from_df
from src.model.model import SASRec
from src.utils.enums import MovieLensDataset, MovieLensType
from src.utils.logger import Logger

logger = Logger.get_logger("SASRec")

def plot_and_save(x: List[int], y: List[float], title: str, filename: str) -> None:
    """
    Plota e salva um gráfico das métricas ao longo das épocas.

    Args:
        x (List[int]): Lista de épocas.
        y (List[float]): Lista de valores da métrica.
        title (str): Título do gráfico.
        filename (str): Nome do arquivo para salvar o gráfico.
    """
    plt.figure()
    plt.plot(x, y, label=title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def train_and_evaluate(dataset: MovieLensDataset = MovieLensDataset.ML_1M) -> None:
    """
    Executa o treinamento e a avaliação do modelo SASRec,
    calculando as métricas nDCG@K, HR@K, MAP@K, Precision@K e Recall@K.
    Registra os valores no TensorBoard e via logger.
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = os.cpu_count()
    evaluate_type = "simple"

    # Configura o TensorBoard
    writer = SummaryWriter(log_dir="../tensorboard_logs")

    # Carrega os dados
    ratings_df = Loader().load_pandas(dataset=dataset, ml_type=MovieLensType.RATINGS)
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

    logger.info("number of parameters: %.2fM" % (model.get_num_params() / 1e6,))
    logger.info("model architecture:")
    logger.info(model)

    # Define a função de perda e o otimizador
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=d.lr, betas=(0.9, 0.98))

    model.train()

    # Listas para armazenar as métricas ao longo das épocas
    epoch_list = []
    ndcg_list = []
    hr_list = []
    val_ndcg_list = []
    val_hr_list = []

    global_step = 0
    os.makedirs("./saved_results", exist_ok=True)

    # Inicia o cronômetro para o tempo total de treinamento
    start_time = time.time()

    with open(f"./saved_results/result0.txt", "w") as results_file:
        for epoch in range(1, d.num_epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            # Inicia o cronômetro para o tempo da época
            epoch_start_time = time.time()

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

            # Calcula o tempo gasto na época
            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"Epoch: {epoch} - Avg Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")

            if epoch % 1 == 0:
                model.eval()
                epoch_list.append(epoch)

                # Avaliação no conjunto de treino
                avg_ndcg, avg_hr = evaluate(
                    model, [train_data, valid_data, test_data, user_num, item_num], d.sequence_length, batch_size=1024
                )
                ndcg_list.append(avg_ndcg)
                hr_list.append(avg_hr)
                logger.info(
                    f"(Train)    Epoch: {epoch}, nDCG@K: {avg_ndcg:.4f}, HR@K: {avg_hr:.4f}, Loss: {loss.item():.4f}"
                )
                results_file.write(
                    f"(Train)    Epoch: {epoch}, nDCG@K: {avg_ndcg:.4f}, HR@K: {avg_hr:.4f}, Loss: {loss.item():.4f}\n"
                )
                results_file.flush()

                writer.add_scalar("nDCG/train", avg_ndcg, epoch)
                writer.add_scalar("HR/train", avg_hr, epoch)

                # Avaliação no conjunto de validação
                val_ndcg, val_hr = evaluate(
                    model,
                    [train_data, valid_data, test_data, user_num, item_num],
                    d.sequence_length,
                    batch_size=1024,
                    isvalid=True
                )
                val_ndcg_list.append(val_ndcg)
                val_hr_list.append(val_hr)
                logger.info(
                    f"(Validate) Epoch: {epoch}, nDCG@K: {val_ndcg:.4f}, HR@K: {val_hr:.4f}, Loss: {loss.item():.4f}\n"
                )
                writer.add_scalar("nDCG/val", val_ndcg, epoch)
                writer.add_scalar("HR/val", val_hr, epoch)

                model.train()

    # Calcula o tempo total de treinamento
    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time:.2f}s")

    # Salva os gráficos para cada métrica
    plot_and_save(epoch_list, ndcg_list, "nDCG@K (Train)", "./saved_results/nDCG_train.png")
    plot_and_save(epoch_list, hr_list, "HR@K (Train)", "./saved_results/HR_train.png")
    plot_and_save(epoch_list, val_ndcg_list, "nDCG@K (Validation)", "./saved_results/nDCG_val.png")
    plot_and_save(epoch_list, val_hr_list, "HR@K (Validation)", "./saved_results/HR_val.png")

    logger.info("Finished....")
    writer.close()
    plt.close()

if __name__ == "__main__":
    train_and_evaluate()
