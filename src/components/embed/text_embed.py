from os.path import dirname, join
from typing import Optional

import torch
from transformers import BertModel, BertTokenizer

from src.components.embed.embed import Embedding
from src.module.utils import text as text_utils


# TEXT_EMBED_MODEL = join(dirname(dirname(dirname(dirname(__file__)))), 'model', 'text_embedding_model')


class TextEmbedding(Embedding):
    def __init__(self,local_path, *args, **kwargs):
        super(TextEmbedding, self).__init__(*args, **kwargs)
        self.local_path = local_path
        self._setup_model()
        
        # self.tokenizer = BertTokenizer.from_pretrained(TEXT_EMBED_MODEL, local_files_only=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.local_path, local_files_only=True)

    def _setup_model(self):
        # @TODO: s3에서 모델 다운받도록 변경
        # self.model = BertModel.from_pretrained(TEXT_EMBED_MODEL, local_files_only=True)
        self.model = BertModel.from_pretrained(self.local_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def run(self, text: str) -> Optional[torch.Tensor]:
        """
        임베딩을 구하는 함수

        :param text: text
        :return embedding vector
        """
        if text and (isinstance(text, int) or isinstance(text, float)):
            text = str(text)

        if not text or not isinstance(text, str):
            return None

        text = text_utils.humanize(text)
        output = self.model(**self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=150, truncation=True).to(self.device))

        return output.pooler_output.squeeze().detach().cpu()
