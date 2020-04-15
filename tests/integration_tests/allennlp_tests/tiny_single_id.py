import itertools
from typing import Dict
from typing import List

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
import torch


@TokenIndexer.register("tiny_single_id")
class SingleIdTokenIndexer(TokenIndexer):
    """Tiny implementation of SingleIdTokenIndexer.

    This class is based on allennlp SingleIdTokenIndexer.
    https://github.com/allenai/allennlp/blob/master/
    allennlp/data/token_indexers/single_id_token_indexer.py

    """

    def __init__(
        self,
        lowercase_tokens: bool = False,
        start_tokens: List[str] = None,
        end_tokens: List[str] = None,
        token_min_padding_length: int = 0,
    ) -> None:
        super().__init__(token_min_padding_length)
        self.lowercase_tokens = lowercase_tokens

        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]) -> None:
        text = token.text
        if self.lowercase_tokens:
            text = text.lower()
        counter["tokens"][text] += 1

    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary, index_name: str
    ) -> Dict[str, List[int]]:
        indices: List[int] = []

        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            text = token.text
            if self.lowercase_tokens:
                text = text.lower()
            indices.append(vocabulary.get_token_index(text, "tokens"))

        return {index_name: indices}

    def get_padding_lengths(self, token: int) -> Dict[str, int]:
        return {}

    def as_padded_tensor(
        self,
        tokens: Dict[str, List[int]],
        desired_num_tokens: Dict[str, int],
        padding_lengths: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        return {
            key: torch.LongTensor(pad_sequence_to_length(val, desired_num_tokens[key]))
            for key, val in tokens.items()
        }
