
from src.components.train.train import GruTrainer
import pandas as pd
import pickle
import os
import argparse
import json
from distutils.dir_util import copy_tree


INPUT_DIR = "/opt/ml/input/data"
MODEL_DIR = "/opt/ml/model"
ch_name_train = "interaction_data"
ch_name_emb = "embedding"
ch_name_encoding = "encoding_data"
ch_name_sim = "sim_dic"
ch_name_whole = "whole_embedding"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_size', type=int, default=25, help='input size of sequence length')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_items', type=int, default=30000, help='maximum of the number of recommendations')  
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--drop_out', type=float, default=0.2)
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--cpu_workers', type=int, default=8 if os.cpu_count() > 8 else 1)
    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--validation_size', type=float, default=0.2)
    parser.add_argument('--return_item_num', type=int, default=300, help='recommend item num')
    parser.add_argument('--ratio_cold_start', type=int, default=0.2, help='sim:personal ratio for solving cold start problem')
    parser.add_argument('--test_data_num', type=int, default=0, help='train data number for pipe line testing default None')
    parameters = parser.parse_args()
    model_config = parameters.__dict__
    

    # load data
    train_input_files = [os.path.join(INPUT_DIR, file) for file in os.listdir(INPUT_DIR) if ch_name_train in file and 'manifest' not in file]
    embedding_input_files = [os.path.join(INPUT_DIR, file) for file in os.listdir(INPUT_DIR) if ch_name_emb in file and 'manifest' not in file and ch_name_whole not in file]
    encoding_input_files  = [os.path.join(INPUT_DIR, file) for file in os.listdir(INPUT_DIR) if ch_name_encoding in file and 'manifest' not in file] 
    whole_embedding_input_files = [os.path.join(INPUT_DIR, file) for file in os.listdir(INPUT_DIR) if ch_name_whole in file and 'manifest' not in file]
    sim_input_files = [os.path.join(INPUT_DIR, file) for file in os.listdir(INPUT_DIR) if ch_name_sim in file and 'manifest' not in file]
    raw_train = [pd.read_csv(file, engine='python') for file in train_input_files]
    raw_encoding = [pd.read_csv(file, engine='python') for file in encoding_input_files]
    
    interaction_data = pd.concat(raw_train)
    encoding_data = pd.concat(raw_encoding)
    interaction_data.reset_index(inplace=True, drop=True)
    encoding_data.reset_index(inplace=True, drop=True)
    raw_embedding = [pickle.load(open(emb,'rb')) for emb in embedding_input_files]
    raw_whole = [pickle.load(open(emb,'rb')) for emb in whole_embedding_input_files]
    raw_sim = [pickle.load(open(sim,'rb')) for sim in sim_input_files]
    embedding_data = {key:value for d in raw_embedding for key, value in d.items()}
    whole_embedding_data = {key:value for d in raw_whole for key, value in d.items()}
    sim_dic = {key:value for d in raw_whole for key, value in d.items()}
    # save rest
    encoding_data.to_csv(os.path.join(MODEL_DIR, 'encoding_df.csv'), mode='w', encoding='utf-8-sig')
    with open(os.path.join(MODEL_DIR, 'whole_embedding.pickle'), 'wb') as f:
        pickle.dump(whole_embedding_data, f)
    del whole_embedding_data
    del raw_whole
    with open(os.path.join(MODEL_DIR, 'sim_dic.pickle'), 'wb') as f:
        pickle.dump(sim_dic, f)
    del sim_dic
    del raw_sim
    copy_tree('/opt/ml/code', '/opt/ml/model/code')
    #tmp
    if parameters.test_data_num != 0:    
        interaction_data = interaction_data[:parameters.test_data_num]
    
    # train model
    gru_trainer = GruTrainer(**model_config)
    gru_trainer.run(embedding_data, interaction_data, MODEL_DIR)
