from typing import List, Tuple

import bentoml
from ckiptagger import NER, POS, WS


@bentoml.service
class CKIPTaggerService:
    ckiptagger = bentoml.models.get("ckiptagger")

    def __init__(self):
        path = self.ckiptagger.path
        disable_cuda = True

        self.ws_model = WS(path, disable_cuda)
        self.pos_model = POS(path, disable_cuda)
        self.ner_model = NER(path, disable_cuda)

    @bentoml.api(batchable=True)
    def ws(self, texts: List[str]) -> List[List[str]]:
        return self.ws_model(texts)

    @bentoml.api(batchable=True)
    def pos(self, texts: List[str]) -> List[List[Tuple[str, str]]]:
        """
        Returns
        -------
        List[List[Tuple[str, str]]]
            A list of lists of (word, pos) tuples for each input text.
        """
        ws_lists = self.ws_model(texts)
        pos_lists = self.pos_model(ws_lists)
        result = []
        for ws_list, pos_list in zip(ws_lists, pos_lists):
            result.append([(word, pos) for word, pos in zip(ws_list, pos_list)])
        return result

    @bentoml.api(batchable=True)
    def ner(self, texts: List[str]) -> List[List[Tuple[int, int, str, str]]]:
        """
        Returns
        -------
        List[List[Tuple[int, int, str, str]]]
            A list of named entities in the input texts. Each named entity is
            represented as a tuple of (start, end, entity_type, entity_text).
        """
        ws_lists = self.ws_model(texts)
        pos_lists = self.pos_model(ws_lists)
        ner_lists = self.ner_model(ws_lists, pos_lists)
        return ner_lists
