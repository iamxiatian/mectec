from typing import List, Optional, Tuple, Union
from mectec.vocab import pos_vocab

import torch
from torch import nn

class PartOfSpeechLayer(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    与标准的BertEmbedding相比，多了一个config.extra_vocab_size来表示额外信息的词汇表大小，
    extra_ids表示额外输入信息的id，例如词性信息或者拼音信息
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pos_embeddings = nn.Embedding(len(pos_vocab), self.hidden_size, padding_idx=pos_vocab.pad_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.transformers = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8, norm_first=True, dropout=0.2),
                num_layers = 2,
                norm=nn.LayerNorm(self.hidden_size)
            )
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: Optional[torch.LongTensor]) -> torch.Tensor:
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, 0 : seq_length]

        embeddings = self.word_embeddings(input_ids)
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.transformers(embeddings)
        
        return embeddings