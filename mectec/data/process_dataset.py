from collections import defaultdict
from tqdm import tqdm
from mectec.conf import bert_tokenizer
from mectec.csc import FileDataset
from mectec.util import nlp, BetterFile
import pickle

def generate_pos_vocab():
    """生成词性对应的词汇表"""
    pairs = FileDataset('./data/train_public.txt', bert_tokenizer, 180, True).pairs
    sentences = [pair[0] for pair in pairs]

    vocab = defaultdict(int)
    for sentence in tqdm(sentences):
        _, tags = nlp.tag_text(sentence)
        for tag in tags:
            vocab[tag] = vocab[tag] + 1
        
    BetterFile('./vocabulary/word_type.txt').save_lines(vocab.keys())


def generate_train_cache():
    """缓存到文件中，避免每次加载速度过慢"""
    ds =  FileDataset('./data/train_public.txt', bert_tokenizer, 180, True)
    cached_file = './data/train_enhanced_pos.pkl'
    output = [ds[idx] for idx in range(len(ds))]
    with open(cached_file, 'wb') as f: 
        pickle.dump(output, f)    
    
generate_train_cache()