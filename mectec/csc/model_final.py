from typing import Optional
import copy
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import BertConfig, BertModel
from mectec.vocab import glyph_vocab, pinyin_vocab, megec_vocab
from mectec import conf
from mectec.nn.bert import BertEmbeddings

from xiatian import L

class MectecFinal(nn.Module):
    '''
    MECTEC（Multi-modal Enhanced Chinese Text Error Correction）
    
    1. 代码中其中单独的字母d表示detect
    2. Final模型是实验模型中的gold7模型
    3. 当前模型是最佳模型，模型结构与论文保持一致
    '''
    
    def __init__(self, base_model,dropout_rate, label_loss_weight=0.5):
        super(MectecFinal, self).__init__()

        self.config = BertConfig.from_pretrained(base_model)  # 加载模型超参数
        self.base_model = BertModel.from_pretrained(base_model)  # 加载模型
        self.predictor_dropout = nn.Dropout(dropout_rate)
        self.label_num = megec_vocab.num_labels # labels：20000多
        self.d_tag_num = megec_vocab.num_dtags  # tags大小；正确或者不正确
        self.tag_error_index = 1  # INCORRECT在tags中的索引
        self.hidden_size = self.config.hidden_size
        self.pinyin_dim = self.hidden_size
        self.glyph_dim = self.hidden_size
        
        self.pinyin_layer = nn.Sequential(
            nn.Embedding(len(pinyin_vocab), 
                         self.pinyin_dim, 
                         padding_idx=pinyin_vocab.pad_id),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.pinyin_dim, 
                                           nhead=8,
                                           activation='gelu',
                                           batch_first=True,
                                           norm_first=True),
                num_layers = 3,
                norm=nn.LayerNorm(self.hidden_size)),
            nn.Dropout(0.5)
        )
    
        # 汉字字形特征
        if not glyph_vocab.loaded:
            glyph_vocab.load()    
        weight = t.as_tensor(glyph_vocab.vocab, dtype=t.float)
        self.glyph_layer = nn.Sequential(
            t.nn.Embedding(self.config.vocab_size, 768, _weight=weight) ,
            nn.Linear(768, 128),  # 400/92
            nn.GELU(),  # nn.GELU(), nn.ReLU() nn.Tanh()
            nn.Dropout(0.1),
            nn.Linear(128, 64),  # 400/92
            nn.GELU(),  # nn.GELU(), nn.ReLU() nn.Tanh()
            nn.Dropout(0.1),
            nn.Linear(64, self.hidden_size),
            nn.Dropout(0.5)
        )
        
        # 检测层
        #self.d_gate = nn.Linear(self.hidden_size*2, 2)
        self.detect_linear = nn.Linear(self.hidden_size, self.d_tag_num)  
        
        self.res_layer =  nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0.5)
        )
        
        # 纠正层
        self.gec_layer = nn.Linear(self.hidden_size, self.label_num)
        
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        
        self.label_loss_weight = label_loss_weight
        self.detect_loss_weight = 1 - self.label_loss_weight
        L.debug('labels:',self.label_num, 'tags:', self.d_tag_num)

    @autocast()
    def forward(self, input_ids, 
                type_ids=None,
                mask=None, 
                label_ids=None,
                dtag_ids=None,
                pinyin_ids=None, 
                is_test=False):
        hanzi_embeds = self.base_model.embeddings(input_ids, type_ids)
        batch_size, sequence_length, _ = hanzi_embeds.size()
        base_out, _ = self.base_model(inputs_embeds=hanzi_embeds, 
                                      attention_mask=mask, 
                                      return_dict=False)   
        
        logits_d = self.detect_linear(base_out)
        
        pinyin_out = self.pinyin_layer(pinyin_ids)
        glyph_out = self.glyph_layer(input_ids)
        res_out = self.res_layer(hanzi_embeds)
        merge_feature = res_out + base_out + pinyin_out + glyph_out
        logits_labels = self.gec_layer(merge_feature)
        
        if is_test:
            label_probs = F.softmax(logits_labels, dim=-1).view(
                [batch_size, sequence_length, self.label_num])
            d_probs = F.softmax(logits_d, dim=-1).view(
                [batch_size, sequence_length, self.d_tag_num])
            err_probs = d_probs[:, :, self.tag_error_index] * mask
            # 句子中最大错误词语的错误概率和在句子中的位置
            max_dtag_error_prob, max_dtag_error_idx = t.max(err_probs, dim=-1)

            output_dict = {
                    "logits_labels": logits_labels,
                    "logits_d_tags": logits_d,
                    "class_probabilities_labels": label_probs,
                    "class_probabilities_d_tags": d_probs,
                    "max_error_probability": max_dtag_error_prob,
                    "max_error_index": max_dtag_error_idx
                }
        else:
            output_dict = {
                    "logits_labels": logits_labels, 
                    "logits_d_tags": logits_d
                }

        if label_ids is not None and dtag_ids is not None:
            loss_labels = self.loss_fct(logits_labels.view(-1, self.label_num), 
                                        label_ids.view(-1))
            loss_d = self.loss_fct(logits_d.view(-1, self.d_tag_num), 
                                   dtag_ids.view(-1))
            
            loss_total = self.label_loss_weight * loss_labels \
                + self.detect_loss_weight * loss_d 
                
            output_dict["loss_total"] = loss_total    
            
        return output_dict
