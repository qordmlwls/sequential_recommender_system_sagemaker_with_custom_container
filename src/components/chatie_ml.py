
class ChatieML:
    @staticmethod
    def test():
        print('hi')

    @staticmethod
    def train():
        from .train import Trainer

        trainer = Trainer()
        trainer.train()
