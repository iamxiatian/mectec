import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel
from mectec.vocab import megec_vocab, pinyin_vocab
from mectec import conf

class Acceptance(nn.Module):
    def __init__(self, base_model,
                dropout_rate, 
                label_loss_weight):
        super(Acceptance, self).__init__()

        self.config = BertConfig.from_pretrained(base_model)  # 加载模型超参数
        self.base_model = BertModel.from_pretrained(base_model)  # 加载模型
        self.predictor_dropout = nn.Dropout(dropout_rate)
        self.hidden_size = self.config.hidden_size
        self.correct_label_num = megec_vocab.num_labels  # labels：20000多
        self.detect_tag_num = megec_vocab.num_dtags  # tags大小：4
        self.incorr_index = 1  # INCORRECT在tags中的索引
        self.pinyin_dim = self.hidden_size
        self.glyph_dim = self.hidden_size
        # self.dropout_rate = 0
        print('labels:',self.correct_label_num, 'tags:', self.detect_tag_num)
        
        # 纠正层
        self.tag_labels_projection_layer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size + 6, self.correct_label_num),  # 拼接了接受度输出和检测输出          
        )
        
        # 检测层
        self.tag_detect_projection_layer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size, self.detect_tag_num)            
        )
        
        # self.classifier = IntentClassifier(self.config.hidden_size, self.num_labels, self.dropout_rate)
        self.label_loss_weight = label_loss_weight
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)
        
        # self.sentence_dropout = nn.Dropout(0.5)
        # self.sentence_classifier = nn.Linear(self.config.hidden_size, 1)
        # self.sigmoid = nn.Sigmoid()
        
        self.glyph_layer = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.2),
            nn.Linear(768, 64),  # 400/92
            nn.Tanh(),  # nn.GELU(), nn.ReLU() nn.Tanh()
            nn.Linear(64, self.glyph_dim)
        )

        self.speech_layer = nn.Sequential(
            nn.Embedding(len(pinyin_vocab), self.pinyin_dim, padding_idx=pinyin_vocab.pad_id),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.pinyin_dim, nhead=6, norm_first=True, dropout=0.2),
                num_layers = 2,
                norm=nn.LayerNorm(self.hidden_size)
            )
        )
        
        # 模型决定如何使用原始嵌入层和隐藏层的权重
        self.resnet_gate = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2*self.hidden_size+2, 2)     
        )
        
        # 音形义输出结果的权重计算
        self.megec_gate = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(3*self.hidden_size+6, 3)   
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=12, norm_first=False, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = 3, norm=nn.LayerNorm(self.hidden_size))
                
        # 接受度
        self.accept_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.config.hidden_size, 2),            
        )
        

    @autocast()
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
                correct_label_ids=None, detect_tag_ids=None,
                tgt_tokens=None, tgt_masks=None, tgt_segments=None,
                pinyin_ids=None, glyph_embeddings=None,                 
                is_test=False):
        embedding_out = self.base_model.embeddings(input_ids)

        batch_size, sequence_length, _ = embedding_out.size()
        sequence_output, pooler_output = self.base_model(input_ids=input_ids,
                                                        token_type_ids=token_type_ids,
                                                        attention_mask=attention_mask,
                                                        return_dict=False)       
        # 利用pooler_output进行句子通顺都判断
        #logits_accept = self.accept_projection_layer(self.accept_dropout(pooler_output))
        logits_accept = self.accept_layer(pooler_output)
        
        #把logits_accept由[batch_size, 2] =>[batch_size, seq_len, 2] 
        accept_factor = logits_accept.unsqueeze(1)        
        accept_factor = accept_factor.expand(len(input_ids),len(input_ids[0]),len(logits_accept[-1]))
        
        concated_outputs = torch.cat((sequence_output, embedding_out, accept_factor), dim=-1)
        resnet_gated_values = self.resnet_gate(concated_outputs)        
        # g0 = torch.sigmoid(resnet_gated_values[:,:,0].unsqueeze(-1))
        # g1 = torch.sigmoid(resnet_gated_values[:,:,1].unsqueeze(-1))        
        # resnet_seq = g0 * sequence_output + g1 * embedding_out
        g = torch.softmax(resnet_gated_values, dim=-1)
        g0, g1 = g[:,:,0].unsqueeze(-1), g[:,:,1].unsqueeze(-1)              
        resnet_seq = g0 * sequence_output + g1 * embedding_out
        
        logits_d = self.tag_detect_projection_layer(resnet_seq)
        
        speech_output = self.speech_layer(pinyin_ids)  # [batch X seq_len X label_num ]
        glyph_output = self.glyph_layer(glyph_embeddings)
        concated_outputs = torch.cat((resnet_seq, speech_output, glyph_output, logits_d, accept_factor), dim=-1)
        megec_gated_values = self.megec_gate(concated_outputs)
        
        # g0 = torch.sigmoid(megec_gated_values[:,:,0].unsqueeze(-1))
        # g1 = torch.sigmoid(megec_gated_values[:,:,1].unsqueeze(-1))
        # g2 = torch.sigmoid(megec_gated_values[:,:,2].unsqueeze(-1))        
        # resnet_seq = g0*resnet_seq + g1*speech_output + g2*glyph_output
        
        g = torch.softmax(megec_gated_values, dim=-1)
        g0, g1, g2 = g[:,:,0].unsqueeze(-1), g[:,:,1].unsqueeze(-1), g[:,:,2].unsqueeze(-1)
        resnet_seq = g0*resnet_seq + g1*speech_output + g2*glyph_output
        
        # pinyin_embedding_output = torch.cat((pinyin_embedding_output, shape_weight), dim=-1)
        # resnet_seq = torch.cat((resnet_seq, pinyin_embedding_output, shape_weight), dim=-1)

        resnet_seq = self.transformer(resnet_seq)

        # 把logits_accept由[batch_size, 2] =>[batch_size, seq_len, 2] 
        # y = torch.softmax(logits_accept, dim=-1).unsqueeze(1)  
        final_seq = torch.cat((resnet_seq, logits_d, accept_factor), dim=-1)
        logits_labels = self.tag_labels_projection_layer(final_seq)

        # 如果检测到错误，但没有纠正到错误，应该重新掩盖掉错误位置的信息，利用上下文进行检测
        # 但此时，应保留其他参数信息，只是重新计算logits_labels

        if is_test:
            class_probabilities_labels = F.softmax(logits_labels, dim=-1).view(
                [batch_size, sequence_length, self.correct_label_num])
            class_probabilities_d = F.softmax(logits_d, dim=-1).view(
                [batch_size, sequence_length, self.detect_tag_num])
            erro_probs = class_probabilities_d[:, :, self.incorr_index] * attention_mask
            incorr_prob = torch.max(erro_probs, dim=-1)[0]
            error_index = torch.max(erro_probs, dim=-1)[1]

            probability_change = [conf.keep_confidence, conf.del_confidence] + [0] * (self.correct_label_num - 2)
            class_probabilities_labels += torch.FloatTensor(probability_change).repeat(
                (batch_size, sequence_length, 1)).to(class_probabilities_labels.device)

            output_dict = {"logits_labels": logits_labels,
                            "logits_accept": logits_accept,
                            "logits_d_tags": logits_d,
                            "class_probabilities_labels": class_probabilities_labels,
                            "class_probabilities_d_tags": class_probabilities_d,
                            "max_error_probability": incorr_prob,
                            "max_error_index": error_index}
        else:
            output_dict = {"logits_labels": logits_labels, "logits_d_tags": logits_d}

        if correct_label_ids is not None and detect_tag_ids is not None and tgt_tokens is not None:
            # 计算通顺度损失, 目标句子都是通顺句
            _, tgt_pooler_output = self.base_model(input_ids=tgt_tokens, token_type_ids=tgt_segments, attention_mask=tgt_masks, return_dict=False)
            tgt_accept_logits = self.accept_layer(tgt_pooler_output)
            tgt_accept_labels = torch.zeros_like(tgt_accept_logits)
            tgt_accept_labels[:,0]  = 1
            # 训练时原始句子都是不通顺的，通顺度得分为0
            src_accept_labels = torch.zeros_like(logits_accept)
            src_accept_labels[:,1] = 1
            
            # 如果原句子和目标句子相同，则标签需要修改为1
            for i in range(len(input_ids)):
                if input_ids[i].equal(tgt_tokens[i]):
                    src_accept_labels[i][0] = 1
                    src_accept_labels[i][1] = 0
                    
            # 两个句子的整体通顺度得分
            loss_accept = 0.5*self.loss_fct(logits_accept, src_accept_labels) + 0.5*self.loss_fct(tgt_accept_logits, tgt_accept_labels)                        
            loss_labels = self.loss_fct(logits_labels.view(-1, self.correct_label_num), correct_label_ids.view(-1))
            loss_d = self.loss_fct(logits_d.view(-1, self.detect_tag_num), detect_tag_ids.view(-1))
            
            loss_total = self.label_loss_weight * loss_labels + (1 - self.label_loss_weight) * loss_d
            loss_total = 0.8*loss_total + 0.2*loss_accept
            output_dict["loss_total"] = loss_total
        return output_dict
