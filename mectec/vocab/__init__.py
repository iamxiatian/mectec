from . megec_vocab import MegecVocab
from . pinyin import PinyinVocab
from . glyph import GlyphVocab
from . pos import PosVocab

__all__ = ['megec_vocab', 'pinyin_vocab', 'glyph_vocab', 'pos_vocab']

megec_vocab = MegecVocab()
pinyin_vocab = PinyinVocab()
glyph_vocab = GlyphVocab()
pos_vocab = PosVocab()

# def get_similar_glyph_chars(token:str, top:int) -> list[str]:
#     from .. conf import bert_tokenizer
#     token_id = bert_tokenizer._convert_token_to_id(token)
#     return glyph_vocab.get_