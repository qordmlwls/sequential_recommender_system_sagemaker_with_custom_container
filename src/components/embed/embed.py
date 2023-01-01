from abc import *
from typing import NoReturn

from src.module.utils import gpu as gpu_utils


class Embedding:
    def __init__(self):
        self.device = gpu_utils.get_gpu_device()
        self.model = None

    @abstractmethod
    def run(self, obj: any) -> any:
        return

    @abstractmethod
    def _setup_model(self) -> NoReturn:
        return
