import torch 
from src.components.train.train import Gru4Rec
from src.module.utils import gpu as gpu_utils
import logging
import sys
import os
import json
import pandas as pd
from typing import Dict, Tuple
import pickle
import numpy as np
import random
from src.module.db.bigquery import BigqueryDB



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


MODEL_DIR = '/opt/ml/model'

device = gpu_utils.get_gpu_device()
bigquerydb = BigqueryDB()

with open(os.path.join(MODEL_DIR, 'model_config.json'), "r") as jsonfile:
    model_config = json.load(jsonfile)
embeddings = pickle.load(open(os.path.join(MODEL_DIR, 'whole_embedding.pickle'), 'rb'))
embeddings = {key:value['whole_embedding'] for key, value in embeddings.items() if isinstance(value['image_embedding'], torch.Tensor) 
                            and isinstance(value['text_embedding'], torch.Tensor)}
encoding_df = pd.read_csv(os.path.join(MODEL_DIR, 'encoding_df.csv'))
encoding_df.sort_values(by=['encoding_id'], inplace=True)
encoding_df.drop(columns='encoding_id', inplace=True)
# encoding_dic = {k:v for k, v in zip(encoding_dic['encoding_id'], encoding_dic['story_id'])}
sim_dic = pickle.load(open(os.path.join(MODEL_DIR, 'sim_dic.pickle'), 'rb'))
def model_fn(model_dir):
    # model = Gru4Rec(**model_config)
    file_exist = False
    try:
        for file_name in os.listdir(model_dir):
            if 'ckpt' in file_name:
                model = Gru4Rec.load_from_checkpoint(os.path.join(model_dir, file_name), **model_config)
                file_exist = True
                break
        if not file_exist:
            raise Exception('model file not exist')
    except Exception as e:
        print('input_fn Error: load model fail') 
        raise FileNotFoundError
    model.to(device).eval()
    return model

def _prepare_data(interaction_data: pd.DataFrame, embeddings: Dict) -> Tuple[torch.Tensor, int]:
    interaction_data.loc[(interaction_data['event_type'] == 'READ') & (interaction_data['event_value'] < 150), ['event_value']] = 0
    interaction_data.loc[
        (interaction_data['event_type'] == 'READ') & (interaction_data['event_value'] >= 150), ['event_value']] = 1
    interaction_data['event_value'] = interaction_data['event_value'].astype(int)
    interaction_data.sort_values(by='event_timestamp', inplace=True)
    story_embeddings = torch.cat([embeddings[story_id] for story_id in interaction_data['story_id'].values]).unsqueeze(dim=0)
    last_story_id = interaction_data['story_id'].values[-1]
    ratings = torch.tensor([interaction_data['event_value'].values])
    input_data = torch.cat([story_embeddings, ratings], dim=1).float().to(device)
    return input_data, last_story_id


def _load_interaction_data(user_id: int, frame_size: int) -> pd.DataFrame:
    # interaction_data = bigquerydb.get_inference_data(user_id=user_id, embedding_id_list=tuple(embeddings.keys()), frame_size=frame_size)
    interaction_data = bigquerydb.get_inference_data(user_id=user_id, embedding_id_list=tuple(), frame_size=frame_size)
    return interaction_data

# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    user_id = int(json.loads(request_body)["user_id"])
    if json.loads(request_body)['request_type'] == 'personal':
        interaction_data = _load_interaction_data(user_id=user_id, frame_size=model_config['frame_size'])
        # NA인 경우 채움
        interaction_data.fillna(random.sample(embeddings.keys(),1)[0])
        interaction_data.loc[~interaction_data['story_id'].isin(embeddings.keys()), ['story_id']] = random.sample(embeddings.keys(),1)[0]
        if len(interaction_data) == model_config['frame_size']:
            input_data, last_story_id = _prepare_data(interaction_data, embeddings)
        elif len(interaction_data) < model_config['frame_size'] and len(interaction_data) > 0:
            input_data = None
            last_story_id = interaction_data['story_id'].values[-1]
        elif len(interaction_data) == 0:
            input_data = None
            last_story_id = random.sample(sim_dic.keys(),1)[0]
        else:
            raise ValueError('input_fn Error: invalid input format')
    elif json.loads(request_body)['request_type'] == 'content':
        input_data = None
        last_story_id = json.loads(request_body).get('story_id')
        if last_story_id is None:
            last_story_id = random.sample(sim_dic.keys(),1)[0]
    # 마지막 읽은 작품이 오늘 새로 등록된 작품인 경우 랜덤 샘플링
    try:
        sim_dic[last_story_id]
    except KeyError:
        last_story_id = random.sample(sim_dic.keys(),1)[0]
    except TypeError:
        # 유사 컨텐츠 추천 요청에 story_id이 None인 경우 
        last_story_id = random.sample(sim_dic.keys(),1)[0]
        
    return {
            'user_id': user_id,
            'input_data': input_data,
            'last_story_id': last_story_id
          }

# @TODO: 유사한 작품 추천을 분기시킬것인가?
# inference
def predict_fn(input_object, model):
    if isinstance(input_object['input_data'], torch.Tensor):
        logits = model(input_object['input_data'])
        # for log transformation 유사도 점수와 1정도 차이나서 좀 더 조정
        logits = torch.log(100*(torch.sigmoid(logits) + 1.8)).detach().cpu().numpy()
        encoding_df['score'] = logits.squeeze(0)
        # scaling score 기준 정렬(콜드 스타트 해결) 2:8 비율로 섞는다
        sim_df = pd.DataFrame(sim_dic[input_object['last_story_id']].items(), columns=['story_id', 'score'])[:int(model_config['return_item_num']*model_config['ratio_cold_start'])]
        result_df = pd.concat([encoding_df[:model_config['return_item_num']], sim_df])
        result_df.sort_values(by=['score'], inplace=True)
        story_ids = list(dict.fromkeys(result_df['story_id'].values))[:model_config['return_item_num']]
    else:
        # inference를 하지 않을경우 유사작품 목록 반환
        story_ids = list(dict(sorted(sim_dic[input_object['last_story_id']].items(), key=lambda x: x[1])).keys())[:model_config['return_item_num']]
    return {"predictions" : story_ids}

def output_fn(predictions, response_content_type):
    return json.dumps(predictions)


# for testing

# if __name__ == '__main__':
#     request = json.dumps({"user_id":"56", "equest_type":"personal"})
#     data = input_fn(request_body=request, request_content_type="application/json")
#     model = model_fn(MODEL_DIR)
#     prediction = predict_fn(input_object=data, model=model)
#     output = output_fn(predictions=prediction)