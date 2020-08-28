import itertools
from typing import Dict
from typing import List
from typing import Optional

from allennlp.data.token_indexers.token_indexer import IndexedTokenList
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary


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
        start_tokens: Optional[List[str]] = None,
        end_tokens: Optional[List[str]] = None,
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
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[int]]:
        indices: List[int] = []

        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            text = token.text
            if self.lowercase_tokens:
                text = text.lower()
            indices.append(vocabulary.get_token_index(text, "tokens"))

        return {"tokens": indices}

    def get_empty_token_list(self) -> IndexedTokenList:
        return {"tokens": []}
