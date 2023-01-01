import time
import datetime
from typing import List, Dict

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def sort_users_itemwise(user_dict: Dict, users: List) -> pd.Series:
    """
    item 기준 sort 함수
    :param user_dict: user information
    :param users: 학습 데이터
    :return: sorting된 학습 데이터
    """
    return (
        pd.Series(dict([(i, user_dict[i]["items"].shape[0]) for i in users]))
        .sort_values(ascending=False)
        .index
    )


def make_items_tensor(items_embeddings_id_dict: Dict) -> torch.Tensor:
    """
    encoding id sorting하고 순서대로 임베딩 쌓아서 반환하는 함수
    :param items_embeddings_id_dict: Key- encoding_id, Value- embedding vector
    :return: 임베딩 tensor
    """
    keys = list(sorted(items_embeddings_id_dict.keys()))
    items_embeddings_tensor = torch.stack(
        [items_embeddings_id_dict[i] for i in keys]
    )
    return items_embeddings_tensor


def prepare_batch_static_size(
        batch, item_embeddings_tensor: torch.Tensor, frame_size=10, num_items=500
) -> Dict:
    """
    학습 데이터를 time series 형태에 맞춰 변환하고, label encoding을 진행하는 함수
    :param batch: 학습 배치
    :param item_embeddings_tensor: 임베딩 벡터
    :param frame_size: 학습 단위
    :param num_items: 추천 목록
    :return: 변환된 학습 데이터
    """
    item_t, ratings_t, sizes_t, users_t = [], [], [], []
    for i in range(len(batch)):
        item_t.append(batch[i]["items"])
        ratings_t.append(batch[i]["rates"])
        sizes_t.append(batch[i]["sizes"])
        users_t.append(batch[i]["users"])
    item_t = np.concatenate([rolling_window(i, frame_size) for i in item_t], 0)
    ratings_t = np.concatenate(
        [rolling_window(i, frame_size) for i in ratings_t], 0
    )

    item_t = torch.tensor(item_t)
    users_t = torch.tensor(users_t)
    ratings_t = torch.tensor(ratings_t).float()
    sizes_t = torch.tensor(sizes_t)

    batch = {"items": item_t, "users": users_t, "ratings": ratings_t, "sizes": sizes_t}

    return batch_contstate_discaction(
        batch=batch,
        item_embeddings_tensor=item_embeddings_tensor,
        frame_size=frame_size,
        num_items=num_items
    )


def rolling_window(data: np.array, window: int) -> np.array:
    """
    preprocess time series data
    :param data: data
    :param window: frame size
    :return: processed data
    """
    data = np.array(data, dtype='int64')
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def get_irsu(batch) -> List[torch.Tensor]:
    """
    process each batch
    배치에서 데티어 가져오는 함수
    :param batch: batch
    :return: parsed batch
    """
    items_t, ratings_t, sizes_t, users_t = (
        batch["items"],
        batch["ratings"],
        batch["sizes"],
        batch["users"],
    )
    return items_t, ratings_t, sizes_t, users_t


def batch_contstate_discaction(
        batch: Dict, item_embeddings_tensor: torch.Tensor, frame_size: int, num_items: int, *args, **kwargs
) -> Dict:
    """
    Embed Batch: continuous state discrete action
    학습 데이터를 time series 형태에 맞춰 변환하고, label encoding을 진행하는 함수
    """

    items_t, ratings_t, sizes_t, users_t = get_irsu(batch)
    items_emb = item_embeddings_tensor[items_t.long()]
    # for binary classificiation
    num_users = len(users_t)
    user_len_index = torch.cumsum(sizes_t - (frame_size - 1), dim=0)
    user_index = torch.cat([torch.tensor([0]), torch.tensor([user_len_index[i] for i in range(num_users)])], dim=0)

    b_size = ratings_t.size(0) # - num_users
    item_embedding = items_emb.view(b_size, -1)

    emb_input = torch.cat([item_embedding[user_index[idx]:user_index[idx+1]-1, :] for idx in range(len(user_index)-1)], dim=0)
    ratings_input = torch.cat([ratings_t[user_index[idx]:user_index[idx+1]-1, :] for idx in range(len(user_index)-1)], dim=0)
    # 최종 input
    state = torch.cat([emb_input, ratings_input], dim=1).float()

    # target label encoding, 최종 target
    items_target = torch.cat([items_t[user_index[idx]+1:user_index[idx+1], :] for idx in range(len(user_index)-1)], dim=0)
    ratings_target = torch.cat([ratings_t[user_index[idx]+1:user_index[idx+1], :] for idx in range(len(user_index)-1)], dim=0)
    target = torch.zeros(items_target.size(0), num_items)
    for row in range(items_target.size(0)):
        for col in range(items_target.size(1)):
            target[row][items_target[row][col]] = ratings_target[row][col]

    batch = {
        "state": state,
        "action": target,
        "meta": {"users": users_t, "sizes": sizes_t},
    }
    return batch


# deprecated(추후 삭제 예정)
def get_base_batch(batch, device=torch.device("cuda"), done=True):
    b = [
        batch["state"],
        batch["action"],
        batch["reward"].unsqueeze(1),
        batch["next_state"],
    ]
    if done:
        b.append(batch["done"].unsqueeze(1))
    else:
        batch.append(torch.zeros_like(batch["reward"]))
    return [i.to(device) for i in b]


def string_time_to_unix(s: str) -> int:
    return int(time.mktime(datetime.datetime.strptime(s, "%m/%d/%Y, %H:%M:%S").timetuple()))


def refine_interaction_data(interaction_data):
    """
    interaction_data preprocessing
    :param interaction_data:
    :return: encoding_data, processed interaction_data
    """
    interaction_data.loc[(interaction_data['event_type'] == 'READ') & (interaction_data['event_value'] < 150), ['event_value']] = 0
    interaction_data.loc[
        (interaction_data['event_type'] == 'READ') & (interaction_data['event_value'] >= 150), ['event_value']] = 1

    ### 학습에 쓰기 위해 레이블 인코딩을 해준다 @TODO: 배치 데이터 파이프라인 구축 후 주석 해제
    le = LabelEncoder()
    le.fit(interaction_data.story_id)
    interaction_data['encoding_id'] = le.transform(interaction_data['story_id'])
    interaction_data['encoding_id'] = interaction_data['encoding_id'].astype(int)
    encoding_data = interaction_data[['story_id', 'encoding_id']]
    encoding_data.drop_duplicates(subset=['encoding_id'], inplace=True)
    encoding_data = encoding_data.sort_values(by=['encoding_id'])
    ### 학습에 쓰일 수 있게 전처리를 완료하고 리턴한다
    interaction_data = interaction_data[['user_id', 'event_value', 'event_timestamp', 'encoding_id']]
    interaction_data.columns = ['reader_id', 'liked', 'when', 'book_id']
    interaction_data = interaction_data.sort_values(by=['when'])
    return encoding_data, interaction_data


def image_embedding_normalize(x):
    """
    image embedding vector [-6, 16] -> [-1, 1]
    :param x: image embedding vector
    :return: [-1, 1] embedding vector
    """
    return (1/11)*x - (5/11)


def robust_scaling(df: pd.DataFrame, maximum: int, minimum: int) -> pd.DataFrame:
    for colmn in df.columns:
        scaler  = MinMaxScaler(feature_range=(minimum, maximum))
        df[colmn] = df[colmn].fillna(0).astype(float, errors='ignore')
        trd = df[colmn].describe()['75%']
        first = df[colmn].describe()['25%']
        tmp_df = df.loc[((df[colmn]<trd)&(df[colmn]>first)),[colmn]]
        scaler.fit(tmp_df)
        
        high_df = df.loc[df[colmn]>=trd,[colmn]] 
        low_df = df.loc[df[colmn]<=first,[colmn]] 

        df.loc[tmp_df.index,[colmn]] = scaler.fit_transform(df.loc[tmp_df.index,[colmn]])
        df.loc[high_df.index,[colmn]] = maximum
        df.loc[low_df.index,[colmn]] = minimum
        # df.loc[((df[colmn]<trd)&(df[colmn]>first)),[colmn]] = scaler.fit_transform(df.loc[((df[colmn]<trd)&(df[colmn]>first)),[colmn]])
    return df

if __name__ == '__main__':
    df = pd.DataFrame(np.random.rand(6000,5))
    robust_scaling(df, 1, -1)
    print('s')