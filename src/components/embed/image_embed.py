from typing import Optional

import PIL
import torch

from torchvision import transforms
from efficientnet_pytorch import EfficientNet

from src.components.embed.embed import Embedding


class ImageEmbedding(Embedding):
    def __init__(self, *args, **kwargs):
        super(ImageEmbedding, self).__init__(*args, **kwargs)

        self._setup_model()

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _setup_model(self):
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _image_embedding_normalize(x: torch.Tensor) -> torch.Tensor:
        """
        For EfficientNet
        image embedding vector [-6, 16] -> [-1, 1]
        :param x: image embedding vector
        :return: [-1, 1] embedding vector
        """

        return (1 / 11) * x - (5 / 11)

    def _image_preprocessing(self, img: PIL.Image) -> torch.Tensor:
        """
        image preprocessing
        :input: image
        :return: processed image
        """

        img = self.transforms(img)
        return img.type(torch.FloatTensor)

    def run(self, img: PIL.Image) -> Optional[torch.Tensor]:
        """
        embedding generator
        :param img: PIL.Image
        :return: image_embedding
        """
        processed_img = self._image_preprocessing(img)
        output = self.model(processed_img.unsqueeze(0).to(self.device))

        return self._image_embedding_normalize(output.squeeze().detach().cpu())
