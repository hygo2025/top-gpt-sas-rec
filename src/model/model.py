import time
import numpy as np
import torch
import torch.nn as nn

from src.model.point_wise_feed_foward import PointWiseFeedForward


class SASRec(nn.Module):
    """
    Implementação do modelo SASRec conforme o artigo original, utilizando mecanismos de atenção multi-cabeça
    e redes feed-forward ponto a ponto (PointWiseFeedForward).
    """
    def __init__(
        self,
        user_num: int,
        item_num: int,
        hidden_units: int,
        dropout_rate: float,
        sequence_length: int,
        num_of_blocks: int,
        num_of_heads: int
    ) -> None:
        super(SASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.sequence_length = sequence_length

        # Camadas de embedding para itens e posições.
        self.item2emb = nn.Embedding(num_embeddings=item_num + 1, embedding_dim=hidden_units, padding_idx=0)
        self.position2emb = nn.Embedding(num_embeddings=sequence_length, embedding_dim=hidden_units)

        # Listas de módulos para os blocos SASRec: cada bloco contém layerNorm, multi-head attention e FFN.
        self.layer_norm_attn = nn.ModuleList()
        self.multihead_attn = nn.ModuleList()
        self.layer_norm_ffn = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        for _ in range(num_of_blocks):
            self.layer_norm_attn.append(nn.LayerNorm(normalized_shape=hidden_units, eps=1e-8))
            self.multihead_attn.append(
                nn.MultiheadAttention(embed_dim=hidden_units, num_heads=num_of_heads, dropout=dropout_rate)
            )
            self.layer_norm_ffn.append(nn.LayerNorm(normalized_shape=hidden_units, eps=1e-8))
            self.ffn_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """
        Inicializa os parâmetros do modelo utilizando a inicialização Xavier uniform.
        """
        for name, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except Exception:
                pass  # Ignora falhas na inicialização de alguns parâmetros

    def encode_sequence(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Codifica as sequências de entrada adicionando os embeddings de item e posição e
        aplicando os blocos SASRec (atenção multi-cabeça e FFN).

        Args:
            sequences (torch.Tensor): Tensor de inteiros com shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Embeddings codificados com shape (batch_size, sequence_length, hidden_units).
        """
        device = sequences.device
        batch_size, seq_len = sequences.size()

        # Obtém os embeddings dos itens e soma os embeddings posicionais.
        seqs_emb = self.item2emb(sequences)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        seqs_emb += self.position2emb(positions)

        # Cria uma máscara para os itens de padding (valor 0).
        padding_mask = sequences.eq(0)
        # Cria a máscara triangular para garantir o fluxo de informação apenas do passado para o futuro.
        attn_mask = ~torch.tril(torch.ones((self.sequence_length, self.sequence_length), dtype=torch.bool, device=device))

        # Aplica os blocos SASRec
        for i in range(len(self.multihead_attn)):
            # Zera as posições de padding
            seqs_emb = seqs_emb * (~padding_mask).unsqueeze(-1)
            # Transpõe para o formato esperado pela multi-head attention: (sequence_length, batch_size, hidden_units)
            inputs = seqs_emb.transpose(0, 1)
            normed_inputs = self.layer_norm_attn[i](inputs)
            attn_output, _ = self.multihead_attn[i](normed_inputs, inputs, inputs, attn_mask=attn_mask)
            attn_output = self.dropout(attn_output)
            # Conexão residual com o input original
            attn_output += inputs

            normed_ffn_input = self.layer_norm_ffn[i](attn_output)
            ffn_output = self.ffn_layers[i](normed_ffn_input)
            ffn_output = self.dropout(ffn_output)
            # Conexão residual com a saída da atenção
            block_output = ffn_output + attn_output

            # Transpõe de volta para (batch_size, sequence_length, hidden_units)
            seqs_emb = block_output.transpose(0, 1)

        return seqs_emb

    def forward(
        self,
        sequences: torch.Tensor,
        pos_samples: torch.Tensor,
        neg_samples: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Realiza o forward pass durante o treinamento, retornando as predições para amostras positivas e negativas.

        Args:
            sequences (torch.Tensor): Sequências de entrada com shape (batch_size, sequence_length).
            pos_samples (torch.Tensor): Itens amostrais positivos.
            neg_samples (torch.Tensor): Itens amostrais negativos.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predições para amostras positivas e negativas.
        """
        seqs_emb = self.encode_sequence(sequences)
        pos_samples_emb = self.item2emb(pos_samples)
        neg_samples_emb = self.item2emb(neg_samples)

        # Predição via produto escalar entre os embeddings
        pos_predictions = torch.sum(seqs_emb * pos_samples_emb, dim=-1)
        neg_predictions = torch.sum(seqs_emb * neg_samples_emb, dim=-1)

        return pos_predictions, neg_predictions

    def predict(self, log_seqs, item_indices) -> torch.Tensor:
        """
        Realiza a predição para inferência, calculando as pontuações para os itens candidatos
        com base no último embedding da sequência histórica.

        Args:
            log_seqs: Sequência(s) histórica(s) de itens. Pode ser lista ou tensor.
            item_indices: Lista ou tensor com os índices dos itens candidatos.

        Returns:
            torch.Tensor: Tensor com as pontuações preditas para os itens candidatos.
        """
        device = self.item2emb.weight.device

        # Converte log_seqs para tensor se necessário e garante que esteja no dispositivo correto.
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs_tensor = torch.tensor(log_seqs, dtype=torch.long, device=device)
        else:
            log_seqs_tensor = log_seqs.to(device)

        encoded_seqs = self.encode_sequence(log_seqs_tensor)
        # Seleciona o embedding do último passo da sequência.
        final_feature = encoded_seqs[:, -1, :]

        # Converte item_indices para tensor se necessário.
        if not isinstance(item_indices, torch.Tensor):
            item_indices_tensor = torch.tensor(item_indices, dtype=torch.long, device=device)
        else:
            item_indices_tensor = item_indices.to(device)

        # Obtém os embeddings dos itens candidatos e ajusta as dimensões para multiplicação.
        candidate_embs = self.item2emb(item_indices_tensor).transpose(0, 1)
        predictions = final_feature.matmul(candidate_embs)

        return predictions
import time
import numpy as np
import torch
import torch.nn as nn

from src.model.point_wise_feed_foward import PointWiseFeedForward


class SASRec(nn.Module):
    """
    Implementação do modelo SASRec conforme o artigo original, utilizando mecanismos de atenção multi-cabeça
    e redes feed-forward ponto a ponto (PointWiseFeedForward).
    """
    def __init__(
        self,
        user_num: int,
        item_num: int,
        hidden_units: int,
        dropout_rate: float,
        sequence_length: int,
        num_of_blocks: int,
        num_of_heads: int
    ) -> None:
        super(SASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.sequence_length = sequence_length

        # Camadas de embedding para itens e posições.
        self.item2emb = nn.Embedding(num_embeddings=item_num + 1, embedding_dim=hidden_units, padding_idx=0)
        self.position2emb = nn.Embedding(num_embeddings=sequence_length, embedding_dim=hidden_units)

        # Listas de módulos para os blocos SASRec: cada bloco contém layerNorm, multi-head attention e FFN.
        self.layer_norm_attn = nn.ModuleList()
        self.multihead_attn = nn.ModuleList()
        self.layer_norm_ffn = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        for _ in range(num_of_blocks):
            self.layer_norm_attn.append(nn.LayerNorm(normalized_shape=hidden_units, eps=1e-8))
            self.multihead_attn.append(
                nn.MultiheadAttention(embed_dim=hidden_units, num_heads=num_of_heads, dropout=dropout_rate)
            )
            self.layer_norm_ffn.append(nn.LayerNorm(normalized_shape=hidden_units, eps=1e-8))
            self.ffn_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """
        Inicializa os parâmetros do modelo utilizando a inicialização Xavier uniform.
        """
        for name, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except Exception:
                pass  # Ignora falhas na inicialização de alguns parâmetros

    def encode_sequence(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Codifica as sequências de entrada adicionando os embeddings de item e posição e
        aplicando os blocos SASRec (atenção multi-cabeça e FFN).

        Args:
            sequences (torch.Tensor): Tensor de inteiros com shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Embeddings codificados com shape (batch_size, sequence_length, hidden_units).
        """
        device = sequences.device
        batch_size, seq_len = sequences.size()

        # Obtém os embeddings dos itens e soma os embeddings posicionais.
        seqs_emb = self.item2emb(sequences)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        seqs_emb += self.position2emb(positions)

        # Cria uma máscara para os itens de padding (valor 0).
        padding_mask = sequences.eq(0)
        # Cria a máscara triangular para garantir o fluxo de informação apenas do passado para o futuro.
        attn_mask = ~torch.tril(torch.ones((self.sequence_length, self.sequence_length), dtype=torch.bool, device=device))

        # Aplica os blocos SASRec
        for i in range(len(self.multihead_attn)):
            # Zera as posições de padding
            seqs_emb = seqs_emb * (~padding_mask).unsqueeze(-1)
            # Transpõe para o formato esperado pela multi-head attention: (sequence_length, batch_size, hidden_units)
            inputs = seqs_emb.transpose(0, 1)
            normed_inputs = self.layer_norm_attn[i](inputs)
            attn_output, _ = self.multihead_attn[i](normed_inputs, inputs, inputs, attn_mask=attn_mask)
            attn_output = self.dropout(attn_output)
            # Conexão residual com o input original
            attn_output += inputs

            normed_ffn_input = self.layer_norm_ffn[i](attn_output)
            ffn_output = self.ffn_layers[i](normed_ffn_input)
            ffn_output = self.dropout(ffn_output)
            # Conexão residual com a saída da atenção
            block_output = ffn_output + attn_output

            # Transpõe de volta para (batch_size, sequence_length, hidden_units)
            seqs_emb = block_output.transpose(0, 1)

        return seqs_emb

    def forward(
        self,
        sequences: torch.Tensor,
        pos_samples: torch.Tensor,
        neg_samples: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Realiza o forward pass durante o treinamento, retornando as predições para amostras positivas e negativas.

        Args:
            sequences (torch.Tensor): Sequências de entrada com shape (batch_size, sequence_length).
            pos_samples (torch.Tensor): Itens amostrais positivos.
            neg_samples (torch.Tensor): Itens amostrais negativos.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predições para amostras positivas e negativas.
        """
        seqs_emb = self.encode_sequence(sequences)
        pos_samples_emb = self.item2emb(pos_samples)
        neg_samples_emb = self.item2emb(neg_samples)

        # Predição via produto escalar entre os embeddings
        pos_predictions = torch.sum(seqs_emb * pos_samples_emb, dim=-1)
        neg_predictions = torch.sum(seqs_emb * neg_samples_emb, dim=-1)

        return pos_predictions, neg_predictions

    def predict(self, log_seqs, item_indices) -> torch.Tensor:
        """
        Realiza a predição para inferência, calculando as pontuações para os itens candidatos
        com base no último embedding da sequência histórica.

        Args:
            log_seqs: Sequência(s) histórica(s) de itens. Pode ser lista ou tensor.
            item_indices: Lista ou tensor com os índices dos itens candidatos.

        Returns:
            torch.Tensor: Tensor com as pontuações preditas para os itens candidatos.
        """
        device = self.item2emb.weight.device

        # Converte log_seqs para tensor se necessário e garante que esteja no dispositivo correto.
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs_tensor = torch.tensor(log_seqs, dtype=torch.long, device=device)
        else:
            log_seqs_tensor = log_seqs.to(device)

        encoded_seqs = self.encode_sequence(log_seqs_tensor)
        # Seleciona o embedding do último passo da sequência.
        final_feature = encoded_seqs[:, -1, :]

        # Converte item_indices para tensor se necessário.
        if not isinstance(item_indices, torch.Tensor):
            item_indices_tensor = torch.tensor(item_indices, dtype=torch.long, device=device)
        else:
            item_indices_tensor = item_indices.to(device)

        # Obtém os embeddings dos itens candidatos e ajusta as dimensões para multiplicação.
        candidate_embs = self.item2emb(item_indices_tensor).transpose(0, 1)
        predictions = final_feature.matmul(candidate_embs)

        return predictions
