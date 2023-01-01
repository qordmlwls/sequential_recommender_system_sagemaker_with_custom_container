from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch
from typing import Dict, Tuple, List
from tqdm import tqdm
import numpy as np
from src.module.utils import gpu as gpu_utils
from src.module.utils.data import robust_scaling


device = gpu_utils.get_gpu_device()

def _preprocess(embeddings: Dict, feature_df: pd.DataFrame) -> Tuple[torch.Tensor, List]:

    
    
    embeddings = dict(sorted(embeddings.items()))
    embedding_list = []
    story_id_list = []
    for key, value in embeddings.items():
        # 임베딩 없는 경우 제외 
        try:
            embedding_list.append(value['whole_embedding'])
            story_id_list.append(key)
        except KeyError:
            pass
    
    feature_df.sort_values(by=['story_id'], inplace=True)
    feature_df = feature_df.loc[feature_df['story_id'].isin(story_id_list)]
    feature_df.fillna(0.0, inplace=True)
    tmp_dic = {story_id: v for story_id, v in zip(feature_df['story_id'], feature_df['tap_count'])}
    # 작품 데이터 없는 경우 제외(비공개 등)
    idx_dic = {}
    for idx, story_id in enumerate(story_id_list):
        try:
            tmp_dic[story_id]
            idx_dic[idx] = story_id
        except KeyError:
            pass
    story_id_list = list(idx_dic.values())
    embedding_list = [embedding_list[idx] for idx in idx_dic.keys()]
    # feature 생성
    tmp_emb = torch.stack(embedding_list, dim=0)
    feature = torch.from_numpy(feature_df.drop(columns=['story_id']).values)
    return torch.cat([tmp_emb, feature], dim=1).to(device), story_id_list

# 유사도 측정 모듈 
def _cal_similarity(feature: torch.Tensor, batch_size: int) -> torch.Tensor:
    
    row_norm = feature.norm(dim=1).unsqueeze(1).split(batch_size, dim=0)
    
    # elementwise division for solving cuda out of memory error
    batches = feature.split(batch_size, dim=0)
    feature_list = []
    for batch, norm in tqdm(zip(batches, row_norm)):
        feature_list.append((batch / norm).detach().cpu())
    feature_norm = torch.cat(feature_list, dim=0).to(device)
    feature_list = []
    for batch in tqdm(batches):
        # batch별로 normalize를 해줘야 [-1, 1] 사이의 값이 나온다
        batch_norm = batch / batch.norm(dim=1).unsqueeze(1)
        feature_list.append(torch.mm(feature_norm, batch_norm.transpose(0,1)).detach().cpu()) # cpu로 내려줘야 gpu메모리를 안먹는다( cuda out of mememory 방지)
    content_similarity = torch.cat(feature_list, dim=1)
    return content_similarity

# normalize와 scaling을 수행하는 함수 
def _normalize_scaling(score: float) -> float:
    """
    [-1, 1] -> [0, 1] and log transformation
    """
    score = (score + 1)/2
    return np.log(100*(score + 1))


# 유사도 top n개를 리턴하는 함수
def _top_n(content_similarity: torch.Tensor, num_content: int, story_id_list: List) -> Dict:

    sim_score, indices = torch.topk(content_similarity, num_content+1, dim=0)
    return {story_id_list[int(idx)]:float(_normalize_scaling(score)) for rank, idx, score in zip([i for i in range(num_content)], indices.detach().cpu().numpy(), sim_score.detach().cpu().numpy()) if rank !=0}
    

def get_similarity(feature_df: pd.DataFrame, embeddings: Dict, num_content: int, batch_size: int) -> Dict:
    """
    :params:
        feature_df: 응원수, 탭수, 조회수, 댓글 수 dataframe
        embeddings: 전체 임베딩 데이터
        num_content: 각 작품별 저장할 유사한 작품 개수 
    :returns:
        각 작품별 유사한 작품 목록 & 유사도 
    """
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # feature_df[['tap_count', 'view_count', 'comment_count', 'cheer_count']] = 10*scaler.fit_transform(feature_df[['tap_count', 'view_count', 'comment_count', 'cheer_count']])
    
    # apply robust scaling
    feature_df[['tap_count', 'view_count', 'comment_count', 'cheer_count']] = 100*robust_scaling(feature_df[['tap_count', 'view_count', 'comment_count', 'cheer_count']], maximum=1, minimum=-1)
    feature, story_id_list = _preprocess(embeddings=embeddings, feature_df=feature_df)
    content_sim_dic = {}
    content_similarity = _cal_similarity(feature=feature, batch_size=batch_size)
    for idx, story_id in tqdm(enumerate(story_id_list)):
        content_sim_dic[story_id] = _top_n(content_similarity[idx], num_content, story_id_list=story_id_list)
        
    return content_sim_dic