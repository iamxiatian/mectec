import pickle
import torch as t
from xiatian import L
from mectec import conf

class GlyphVocab:
    def __init__(self) -> None:
        self.loaded = False
    
    def load(self) -> None:
        if self.loaded: return None
        
        self.loaded = True
        glyph_embedding_file = conf.glyph_embedding_file

        # 加载字形嵌入信息, 每个汉字对应一行向量，因此是一个二维数组
        L.info(f'正在加载字形文件{glyph_embedding_file}...')      
        with open(glyph_embedding_file, 'rb') as f:
            hanzi_embedding_array = pickle.load(f)
        self.vocab:list[list[float]] = hanzi_embedding_array
        num_tokens = len(hanzi_embedding_array)
        L.info(f'共加载{num_tokens}个字形向量')         
        
    
    def convert_ids_to_embeddings(self, 
                                  token_ids, 
                                  max_len, 
                                  padding:bool=True) -> list[list[float]]:
        """根据汉字在vocab.txt中的id，查表得到其向量。输入的是id序列，输出是一个二维向量

        Args:
            token_ids (list[int]): 词符ID序列
            max_len (int): 补齐的最大长度
            padding (bool, optional): 是否补齐到max_len. Defaults to True.

        Returns:
            list[list[float]]: 对应的嵌入二维向量
        """
        embeddings = [self.vocab[token_id] for token_id in token_ids]
        if padding and len(embeddings) < max_len:
            embeddings.extend([[0.0] * 768] * (max_len - len(embeddings)))

        return embeddings[:max_len]
    
    def convert_id_to_embedding(self, token_id:int) -> list[float]:
        return self.vocab[token_id]
    
    def get_batch_embeddings(self, batch_token_ids) -> t.Tensor:
        batch_embeddings = [] # list[list[list[float]]]
        for token_ids in batch_token_ids:
            sentence_embeddings = [self.vocab[token_id] for token_id in token_ids]
            batch_embeddings.append(sentence_embeddings)
        return t.as_tensor(batch_embeddings, dtype=t.float)
    
    def similar_ids(self, token_id:int, top_k:int) -> list[int]:
        """
        计算和token_id位置的字符形状最相似的前top_k个结果，返回所在位置，该位置对应了
        在Bert的词汇表中的位置。
        """
        embedding = self.vocab[token_id]
        sources = t.FloatTensor(embedding)
        # 变成词汇表大小*768
        sources = sources.expand(len(self.vocab), -1)
        
        targets = t.FloatTensor(self.vocab)
        scores = t.cosine_similarity(sources, targets)
        return t.topk(scores, k=top_k+1).indices.tolist()[1:]