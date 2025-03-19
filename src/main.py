import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import model_selection, preprocessing
from torch.utils.data import DataLoader

from dataset import MovielensDataset
from src.model.evaluate import evaluate
from src.model.load_data import load_data_from_df
from src.model.model import SASRec
from src.loader.loader import Loader
from src.utils.enums import MovieLensDataset, MovieLensType
import src.utils.defaults as d

# Função para plotar e salvar os gráficos de NDCG e HR
def plot_and_save(x, y, title, filename):
    plt.plot(x, y)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = os.cpu_count()

    ratings_df = Loader().load_pandas(dataset=MovieLensDataset.ML_100K, ml_type=MovieLensType.RATINGS)
    [train_data, valid_data, test_data, user_num, item_num] = load_data_from_df(ratings_df)

    train_dataset = MovielensDataset(train_data, d.sequence_length, user_num, item_num)
    train_dataloader = DataLoader(train_dataset, batch_size=d.batch_size, num_workers=num_workers)

    model = SASRec(
        user_num=user_num,
        item_num=item_num,
        hidden_units=d.hidden_units,
        dropout_rate=d.dropout_rate,
        sequence_length=d.sequence_length,
        num_of_blocks=d.num_of_blocks,
        num_of_heads=d.num_heads,
    )
    model = model.to(device)
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=d.lr, betas=(0.9, 0.98))

    model.train()
    t0 = time.time()

    epoch_list = []
    NDCG_list = []
    HR_list = []

    num = 0
    os.makedirs("./saved_results", exist_ok=True)
    f = open(f"./saved_results/result{num}.txt", "w")

    for epoch in range(1, d.num_epochs + 1):
        print("Epoch:", epoch)
        for userid, seq, pos, neg in train_dataloader:
            # Move os tensores para o dispositivo definido em CFG.device
            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pos_predictions, neg_predictions = model(seq, pos, neg)
            pos_labels = torch.ones_like(pos_predictions)
            neg_labels = torch.zeros_like(neg_predictions)

            adam_optimizer.zero_grad()
            # Cria uma máscara para filtrar os itens de padding
            real_labels_mask = (pos != 0)
            loss = bce_criterion(pos_predictions[real_labels_mask], pos_labels[real_labels_mask])
            loss += bce_criterion(neg_predictions[real_labels_mask], neg_labels[real_labels_mask])
            loss.backward()
            adam_optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            epoch_list.append(epoch)
            NDCG, HR = evaluate(model, [train_data, valid_data, test_data, user_num, item_num], d.sequence_length)
            NDCG_list.append(NDCG)
            HR_list.append(HR)
            print("Epoch:", epoch, "NDCG:", NDCG, "HR:", HR)
            f.write(f"Epoch: {epoch}, NDCG: {NDCG}, HR: {HR}, loss: {loss.item()}\n")
            f.flush()
            # Avaliação adicional (opcional)
            NDCG, HR = evaluate(model, [train_data, valid_data, test_data, user_num, item_num], d.sequence_length, True)
            print("(Validate) Epoch:", epoch, "NDCG:", NDCG, "HR:", HR)
            model.train()
    f.close()

    plot_and_save(epoch_list, NDCG_list, "NDCG", "./saved_results/NDCG.png")
    plot_and_save(epoch_list, HR_list, "HR", "./saved_results/HR.png")
    print("Finished....")
    plt.close()
