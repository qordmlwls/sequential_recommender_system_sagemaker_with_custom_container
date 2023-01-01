from src.components.preprocess.preprocess import refine_interaction_data
import pandas as pd
import os
import argparse
from src.module.db.bigquery import BigqueryDB
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
import torch
import random
from tqdm import tqdm
import tarfile
import shutil


WHOLE_EMBEDDING_DIR = "/opt/ml/processing/whole_embedding"
INTERACTION_DATA_DIR = "/opt/ml/processing/interaction_data"
ENCODING_DATA_DIR = "/opt/ml/processing/encoding_data"
EMBEDDING_DIR = "/opt/ml/processing/embedding"
CODE_DIR = "/opt/ml/processing/code"

whole_embedding_name = "whole_embedding.pickle"
encoding_name = "encoding_data.csv"
interaction_name = "interaction_data.csv"
embedding_name = "embedding.pickle"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--month_len', type=int, default=2, help='input intetaction data length')
    parser.add_argument('--num_items', type=int, default=30000, help='maximum of the number of recommendations')
    parser.add_argument('--max_tap_cnt', type=int, default=3000, help='maximum of tap count of content')
    parser.add_argument('--test_data_num', type=int, default=0, help='train data number for pipe line testing default None')
    parameters = parser.parse_args()

    # load whole embeddings
    whole_embeddings = pickle.load(open(os.path.join(WHOLE_EMBEDDING_DIR, whole_embedding_name), 'rb'))
    # None 아닌 것만 
    embedding_id_list = set()
    for key, value in whole_embeddings.items(): 
        if isinstance(value['image_embedding'], torch.Tensor) and isinstance(value['text_embedding'], torch.Tensor):
            embedding_id_list.add(key)
    # load interaction_data until the number of story_id == num_items 
    now = datetime.now()
    date_end = now.strftime('%Y-%m-%d')
    before_time = now - relativedelta(month=parameters.month_len)
    date_start = before_time.strftime('%Y-%m-%d')
    bigquery = BigqueryDB()
    interaction_data_list = []
    while True:
        # 임베딩 있는거만 가져옴
        interaction_data = bigquery.get_interaction_data(date_start=date_start, date_end=date_end, max_tap_cnt=parameters.max_tap_cnt, embeddong_id_list=tuple())
        interaction_data = interaction_data.loc[interaction_data['story_id'].isin(embedding_id_list)]
        interaction_data_list.append(interaction_data)
        story_id_list = list(set(interaction_data['story_id']))
        if len(story_id_list) > parameters.num_items:
            break
        # 데이터가 모자르면 한 주 데이터 더 불러옴
        else:
            before_time = before_time - relativedelta(weeks=1)
            date_end = date_start
            date_start = before_time.strftime('%Y-%m-%d')
    interaction_data = pd.concat(interaction_data_list)
    interaction_data.reset_index(inplace=True, drop=True)
    
    # sampling data, refine data
    sampled_ids = random.sample(story_id_list, parameters.num_items)
    interaction_data = interaction_data.loc[interaction_data['story_id'].isin(sampled_ids), :]
    encoding_data, interaction_data = refine_interaction_data(interaction_data)

    # sample embeddings
    embeddings = dict()
    for story_id in tqdm(sampled_ids):
        # embeddings[story_id] = torch.cat([whole_embeddings[story_id]['image_embedding'], 
        #                                     whole_embeddings[story_id]['text_embedding']], dim=0)
        embeddings[story_id] = whole_embeddings[story_id]['whole_embedding']
    if parameters.test_data_num != 0:    
        interaction_data = interaction_data[:parameters.test_data_num]
    # save files & code
    encoding_data.to_csv(os.path.join(ENCODING_DATA_DIR, encoding_name), mode='w', encoding='utf-8-sig')
    interaction_data.to_csv(os.path.join(INTERACTION_DATA_DIR, interaction_name), mode='w', encoding='utf-8-sig')
    with open(os.path.join(EMBEDDING_DIR, embedding_name), 'wb') as f:
        pickle.dump(embeddings, f)                 
    with tarfile.open('code.tar.gz', 'w:gz') as f:
        f.add('src')
        f.add('train.py')
        f.add('inference.py')
    shutil.move('code.tar.gz', os.path.join(CODE_DIR, 'code.tar.gz'))

    

