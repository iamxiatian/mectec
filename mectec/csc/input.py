
from typing import Optional
from dataclasses import dataclass
import torch

@dataclass
class Input:
    token_ids: torch.Tensor
    label_ids: Optional[torch.Tensor]
    dtag_ids: Optional[torch.Tensor]    
    mask: torch.Tensor
    segment_ids: torch.Tensor
    seq_len: torch.Tensor
    pinyin_ids: Optional[torch.Tensor]
    glyph_embeddings: Optional[torch.Tensor]
    
    def to(self, device):
        self.token_ids.to(device)
        self.mask.to(device)
        self.segment_ids.to(device)
        self.seq_len.to(device)
        if self.label_ids: self.label_ids.to(device)
        if self.dtag_ids: self.dtag_ids.to(device)
        if self.pinyin_ids: self.pinyin_ids.to(device)
        if self.glyph_embeddings: self.glyph_embeddings.to(device)
