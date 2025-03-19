import time

import numpy as np
import torch
import torch.nn as nn

from src.model.point_wise_feed_foward import PointWiseFeedForward


class SASRec(nn.Module):
    def __init__(self, user_num, item_num, hidden_units, dropout_rate, sequence_length, num_of_blocks, num_of_heads):
        super(SASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.item2emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.position2emb = nn.Embedding(sequence_length, hidden_units)
        self.sequence_length = sequence_length

        # Construído de acordo com o artigo original: bloco.
        # Como o número de blocos é variável, aqui foi utilizada uma ModuleList.
        # A seguir, cada uma corresponde ao layerNorm, dropout e camada do mecanismo de atenção multi-cabeça e do FFN.
        self.layerNorm_of_multi_head_attention_layers = nn.ModuleList()
        self.multi_head_attention_layers = nn.ModuleList()

        self.layerNorm_of_FFN_layers = nn.ModuleList()
        self.FFN_layers = nn.ModuleList()

        self.drop = nn.Dropout(dropout_rate)
        # Construindo os blocos de acordo com a quantidade de blocos
        for _ in range(num_of_blocks):
            self.layerNorm_of_multi_head_attention_layers.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.multi_head_attention_layers.append(nn.MultiheadAttention(hidden_units, num_of_heads, dropout_rate))

            self.layerNorm_of_FFN_layers.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.FFN_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        self.initialize()

    def initialize(self):
        for name, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass # just ignore those failed init layers

    def getF(self, seqs):
        device = seqs.device  # Garante que usamos o mesmo dispositivo do input
        seqs_emb = self.item2emb(seqs)
        batch_size, seq_len = seqs.size(0), seqs.size(1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        seqs_emb += self.position2emb(positions)

        time_line_mask = seqs.eq(0)
        attention_mask = ~torch.tril(
            torch.ones(self.sequence_length, self.sequence_length, dtype=torch.bool, device=device))

        for i in range(len(self.multi_head_attention_layers)):
            seqs_emb *= ~time_line_mask.unsqueeze(-1)
            inputs = seqs_emb.transpose(0, 1)
            inputs_norm = self.layerNorm_of_multi_head_attention_layers[i](inputs)
            attention_output, _ = self.multi_head_attention_layers[i](inputs_norm, inputs, inputs,
                                                                      attn_mask=attention_mask)
            attention_output = self.drop(attention_output)
            attention_output += inputs

            inputs_norm = self.layerNorm_of_FFN_layers[i](attention_output)
            outputs = self.FFN_layers[i](inputs_norm)
            outputs = self.drop(outputs)
            outputs += attention_output

            seqs_emb = outputs.transpose(0, 1)
        return seqs_emb

    def forward(self, seqs, pos_samples, neg_samples):
        seqs_emb = self.getF(seqs)  # Obtém o bloco F conforme o artigo original

        pos_samples_emb = self.item2emb(pos_samples)
        neg_samples_emb = self.item2emb(neg_samples)

        # Prediz as amostras positivas e negativas de acordo com a fórmula de predição do artigo original
        pos_predictions = torch.sum(seqs_emb * pos_samples_emb, -1)
        neg_predictions = torch.sum(seqs_emb * neg_samples_emb, -1)

        return pos_predictions, neg_predictions

    def predict(self, log_seqs, item_indices):  # para inferência
        log_feats = self.getF(torch.LongTensor(log_seqs))  # Extrai o embedding correspondente à sequência
        # Para prever o próximo item, basta extrair o embedding do último passo da sequência,
        # conforme o artigo original
        # 1*50
        final_feat = log_feats[:, -1, :]

        # Extrai o embedding dos itens.
        # Note que aqui deve ser passada uma lista, em que os elementos são os IDs dos itens.
        # Há 101 itens no total, dos quais apenas um é uma amostra positiva e os demais são negativos.
        item_embs = self.item2emb(torch.LongTensor(item_indices))
        # Aqui, as dimensões são transpostas para a multiplicação de matrizes subsequente
        item_embs = item_embs.transpose(0, 1)

        predictions = final_feat.matmul(item_embs)
        return predictions
