import os
from typing import Any, Dict, List, NoReturn, Union, Tuple
from torch import Tensor
from pandas import DataFrame

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.module.utils.data import sort_users_itemwise, make_items_tensor, prepare_batch_static_size
from src.module.utils.metrics import calc_precision
from src.module.utils import gpu as gpu_utils
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.utils.data import Dataset, DataLoader
import torch
import json
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pandas.core.indexes.numeric import Int64Index
import pytorch_lightning



class Gru4Rec(LightningModule):

    def __init__(self, **args):
        super().__init__()
        self.args = args
        self.gru = nn.GRU(int(args['input_size']/args['frame_size']), hidden_size=args['hidden_size'], num_layers=args['num_layers'], dropout=args['drop_out'])
        self.linear = nn.Linear(args['hidden_size'], args['num_items'])
        self.layer_norm = nn.LayerNorm(args['hidden_size'])
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, data: Tensor) -> Tensor:
        """

        :param data: input
        :return: model output
        """
        data, hidden = self.gru(data.reshape(data.size(0), self.args['frame_size'], int(data.size(1)/self.args['frame_size'])))
        maximum = data.max()
        if maximum == np.Inf or maximum == -np.Inf:
            raise ValueError('ValueError: output diverges')
        data = F.relu(data[:, -1, :].squeeze(dim=1))
        data = F.relu(self.layer_norm(data))
        action_scores = self.linear(data)
        return action_scores

    def training_step(self, batch: Dict, batch_idx: Any) -> Dict:
        data, target = batch['state'], batch['action']
        logits = self(data)
        loss = self.criterion(logits, target)
        return {'loss': loss}

    def validation_step(self, batch: Dict, batch_idx: Any) -> Dict:
        rank = self.args['frame_size']
        data, target = batch['state'], batch['action']
        logits = self(data)
        loss = self.criterion(logits, target)
        if 'cuda' in str(logits.device):
            _, indices = torch.topk(logits, rank, dim=1)
            _, target = torch.topk(target, rank, dim=1)
        else:  # mps or cpu
            _, indices = torch.topk(logits.detach().cpu(), rank, dim=1)
            _, target = torch.topk(target.detach().cpu(), rank, dim=1)

        indices = indices.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        precisions = []
        for i in range(indices.shape[0]):
            # precision을 맞춰야 하기 때문에 target(rating)이 1인 것만을 추린다 0만 있을 경우 nan처리
            tmp_target = np.array([tg for idx, tg in enumerate(target[i]) if _[i][idx] == 1])
            precisions.append(calc_precision(indices[i], tmp_target))
        return {
            'val_loss': loss,
            'val_maps': precisions
        }

    def training_epoch_end(self, outputs: List[Union[Tensor, Dict[str, Any]]]) -> NoReturn:
        """
        train epoch 종료시
        :param outputs: from each train step
        :return: avg_loss
        """
        try:
            if len(outputs[0]['loss']) > 0:
                loss = torch.tensor([0 for i in range(len(outputs[0]['loss']))], dtype=torch.float)
        except TypeError:
            loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        loss = loss / len(outputs)
        loss = torch.mean(loss)
        self.log('train_loss', float(loss), on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Union[Tensor, Dict[str, Any]]]):
        """
        validation epoch 종료시
        :param outputs: from each validation step
        :return: avg_loss, maps
        """
        # pytorch_lightning 1.7.9에서는 val_loss가 스칼라가 아니라 벡터로 나와서 이에 대한 대비를 함
        try:
            if len(outputs[0]['val_loss']) > 0:
                loss = torch.tensor([0 for i in range(len(outputs[0]['val_loss']))], dtype=torch.float)
        except TypeError:
            loss = torch.tensor(0, dtype=torch.float)
        maps = []
        for i in outputs:
            loss += i['val_loss'].cpu().detach()
            maps += i['val_maps']
        loss = loss / len(outputs)
        loss = torch.mean(loss)
        maps = np.nanmean([np.nanmean(i.cpu().detach()) for i in maps])

        self.log('val_loss', float(loss), on_epoch=True, prog_bar=True)
        self.log('val_maps', float(maps), on_epoch=True, prog_bar=True)

        return {
            'val_loss': loss,
            'val_maps': maps
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_maps'
            }
        }


class GruTrainer:

    def __init__(self, **args):
        self.args = args
        self.items_embeddings_tensor = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.device = gpu_utils.get_gpu_device()

    def _prepare_dataset(self, df: DataFrame) -> Tuple[Int64Index, Dict]:
        """
        refine dataset
        :param df:
        :return:
        """
        frame_size = self.args['frame_size']
        le = LabelEncoder()
        le.fit(df.reader_id)
        df['reader_id'] = le.transform(df['reader_id'])
        df['reader_id'] = df['reader_id'].astype(int)
        users = df[['reader_id', 'book_id']].groupby(['reader_id']).size()
        users = users[users > frame_size].sort_values(ascending=False).index
        ratings = df.sort_values(by='when').set_index('reader_id').drop('when', axis=1).groupby('reader_id')

        # Groupby user
        user_dict = {}

        def app(x):
            userid = x.index[0]
            user_dict[int(userid)] = {}
            user_dict[int(userid)]['items'] = x['book_id'].values
            user_dict[int(userid)]['ratings'] = x['liked'].values
        ratings.apply(app)
        return users, user_dict

    def _prepare_batch_wrapper(self, x):
        batch = prepare_batch_static_size(
            x,
            self.items_embeddings_tensor,
            frame_size=self.args['frame_size'],
            num_items=self.args['num_items']
        )
        return batch

    def _set_config(self, df: DataFrame, model_dir: str):
        """
        prepare dataset, dataloader and set config
        :return:
        """
        users, user_dict = self._prepare_dataset(df)
        train_users, val_users = train_test_split(users, test_size=self.args['validation_size'])
        train_users = sort_users_itemwise(user_dict, train_users)[2:]
        val_users = sort_users_itemwise(user_dict, val_users)

        train_dataset = UserDataset(train_users, user_dict)
        val_dataset = UserDataset(val_users, user_dict)

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.args['batch_size'],
                                           shuffle=True,
                                           num_workers=self.args['cpu_workers'],
                                           collate_fn=self._prepare_batch_wrapper
                                           )

        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=self.args['batch_size'],
                                         shuffle=True,
                                         num_workers=self.args['cpu_workers'],
                                         collate_fn=self._prepare_batch_wrapper
                                         )

        test_batch = next(iter(self.val_dataloader))
        data, label = test_batch['state'], test_batch['action']
        self.args.update({'input_size': len(data[0])})

        # save hyperparameters
        with open(os.path.join(model_dir, 'model_config.json'), "w") as jsonfile:
            json.dump(self.args, jsonfile)

    def run(self, embeddings: Dict, df: DataFrame, model_dir: str):
        ##TODO: DB에서 불러온 embeddings를 dict으로 바꾸는 로직 추가
        seed_everything(self.args['random_seed'])

        self.items_embeddings_tensor = make_items_tensor(embeddings)
        self._set_config(df, model_dir)

        # callbacks
        chk_callback = ModelCheckpoint(
            dirpath=model_dir,
            filename='recommender_model_{epoch:02d}_{val_maps:.5f}',
            verbose=True,
            save_top_k=1,
            monitor='val_maps',
            mode='max'
        )
        earlystop_callback = EarlyStopping(
            monitor='val_maps',
            patience=3,
            min_delta=0.001,
            verbose=True,
            mode='max'
        )

        model = Gru4Rec(**self.args)
        if pytorch_lightning.__version__ == '1.7.6':
            trainer = Trainer(
                callbacks=[chk_callback, earlystop_callback],
                max_epochs=self.args['n_epochs'],
                fast_dev_run=self.args['test_mode'],
                num_sanity_val_steps=None if self.args['test_mode'] else 0,

                # For gpu Setup
                deterministic=True if self.device != torch.device('cpu') else False,
                gpus=-1 if self.device != 'cpu' else None,
                precision=16 if self.args['fp16'] else 32,
                # accelerator='dp'  # pytorch-lightning 1.4.9
                accelerator='cuda' if torch.cuda.is_available() else None,  # pytorch-lightning 1.7.6
                strategy='dp' if torch.cuda.is_available() else None  # pytorch-lightning 1.7.6
            )
        elif pytorch_lightning.__version__ == '1.4.9':
            trainer = Trainer(
                callbacks=[chk_callback, earlystop_callback],
                max_epochs=self.args['n_epochs'],
                fast_dev_run=self.args['test_mode'],
                num_sanity_val_steps=None if self.args['test_mode'] else 0,

                # For gpu Setup
                deterministic=True if self.device != torch.device('cpu') else False,
                gpus=-1 if self.device != 'cpu' else None,
                precision=16 if self.args['fp16'] else 32,
                accelerator='dp'  if torch.cuda.is_available() else None # pytorch-lightning 1.4.9
                # accelerator='cuda' if torch.cuda.is_available() else None,  # pytorch-lightning 1.7.6
                # strategy='dp' if torch.cuda.is_available() else None  # pytorch-lightning 1.7.6
            )            
        else:
            raise Exception("pytorch lightning version should be 1.7.6 or 1.4.9")

        trainer.fit(model=model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)


class UserDataset(Dataset):

    """
    Low Level API: dataset class user: [items, ratings], Instance of torch.DataSet
    """

    def __init__(self, users, user_dict: Dict):
        """

        :param users: integer list of user_id. Useful for train/test splitting
        :type users: list<int>.
        :param user_dict: dictionary of users with user_id as key and [items, ratings] as value
        :type user_dict: (dict{ user_id<int>: dict{'items': list<int>, 'ratings': list<int>} }).

        """

        self.users = users
        self.user_dict = user_dict

    def __len__(self) -> int:
        """
        useful for tqdm, consists of a single line:
        return len(self.users)
        """
        return len(self.users)

    def __getitem__(self, idx: int) -> Dict:
        """
        getitem is a function where non linear user_id maps to a linear index. For instance in the ml20m dataset,
        there are big gaps between neighbouring user_id. getitem removes these gaps, optimizing the speed.

        :param idx: index drawn from range(0, len(self.users)). User id can be not linear, idx is.
        :type idx: int

        :returns:  dict{'items': list<int>, rates:list<int>, sizes: int}
        """
        idx = self.users[idx]
        group = self.user_dict[idx]
        items = group["items"][:]
        rates = group["ratings"][:]
        size = items.shape[0]
        return {"items": items, "rates": rates, "sizes": size, "users": idx}

