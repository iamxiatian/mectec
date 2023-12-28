import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel
from mectec import conf

# WHOAMI在词汇表中的id
WHOAMI_ID = conf.bert_tokenizer._convert_token_to_id(conf.GEC_TOKEN_WHOAMI)

class SelectorModel(nn.Module):
    """
    选择最佳的纠错结果。例如：
    
        原句：我非常高兴你弄过来参加我的生曰派对。
        预测：我非常高兴你能过来参加我的生日派对。
        正报：弄 => 能, 曰 => 日
        
        原句：我跟我的同学学数学。我们对号码有兴趣。 
        预测：我跟我的同学学数学。我们对好码有兴趣。
        误报：号 => 好
        
        选择器处理方式，待判断的汉字，用特殊token [WHOAMI]替换，其他错误用[MASK]遮盖：
        
        [CLS]我非常高兴你[WHOAMI]过来参加我的生[MASK]派对。[SEP]弄[SEP]能[SEP]
        
        [WHOAMI]的结果为0，表示保留原句结果，为1表示使用预测后的结果.
        
        注意：WHOAMI只是占位符号，该位置的字符传入到模BERT中的是MASK，以便复用预训练的结果
    """
    def __init__(self, dropout_rate=0.1):
        super(SelectorModel, self).__init__()
        base_model = conf.SELECT_BASE_MODEL
        self.config = BertConfig.from_pretrained(base_model)  # 加载模型超参数
        self.base_model = BertModel.from_pretrained(base_model)  # 加载模型
        self.predictor_dropout = nn.Dropout(dropout_rate)
        self.hidden_size = self.config.hidden_size
        self.label_num = 2
        
        # 选择层
        self.select_layer = nn.Linear(self.hidden_size*3, self.label_num) 
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)
    
    @autocast()
    def forward(self, input_ids, type_ids, mask, 
                labels=None, 
                is_test:bool=False):
        # 挑选出[WHOAMI]和SEP的位置，一个输入共有三个SEP，所以每个句子共可到4个位置，如下：
        # tensor([ 0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3, ....])
        #tensor([12, 32, 34, 36, 23, 54, 56, 58, 19, 56, 58, 60, 13,  ... ])
        stc_ids, indices = t.where((input_ids == WHOAMI_ID)|(input_ids == 102))
        
        # 形成批次，变成batch_sizex4的形式，再去掉最后一个的SEP, 形成batch_sizex3的形式
        indices = indices.view(-1, 4)[:, :-1] 

        # 排除第0个，即包含WHOAMI的字句，其他位置加1，定位到SEP后面的位置
        # 形成[WHOAMI]、[FIRST]、[SECOND]的形式，都是下标位置
        indices[:, 1:].add_(1)  
        
        # 把input_ids中的WHOAMI位置替换为MASK
        src = t.zeros_like(input_ids)
        src = src.fill_(103) # 全部设置为MASK的ID
        # 将input_ids中WHOAMI位置的元素值设置为103
        input_ids.scatter_(-1, indices[:, [0]], src)
        
        embedding_out = self.base_model.embeddings(input_ids)
        batch_size, _, _ = embedding_out.size()
        sequence_output, _ = self.base_model(input_ids=input_ids,
                                             token_type_ids=type_ids,
                                             attention_mask=mask,
                                             return_dict=False)

        
        # 进一步根据下标获取对应位置的隐层向量
        # 形成如下形式：[0,0,0,1,1,1,2,2,2...]
        ids = stc_ids.view(-1,4)[:, :-1].reshape(-1) 
        out = sequence_output[ids, indices.reshape(-1)]

        # 隐藏向量转变成batch x 3(WHOAMI, FIRST, SECOND) x hidden_size
        # 隐藏向量转变成batch x 3倍的hidden_size
        out = out.view(batch_size, -1)
        logits = self.select_layer(out)

        if not is_test:
            # compute loss for target with class indices
            return self.loss_fct(logits, labels)
        
        # 不是训练处理，则返回预测结果，得分，如果指定了label，同时返回loss
        loss = self.loss_fct(logits, labels) if labels is not None else None
        probabilities = F.softmax(logits, dim=-1)
        # 得分scores，和标记labels，均为batch_size长度
        scores, preds = t.max(probabilities, dim=-1)
        return  preds, scores, loss
