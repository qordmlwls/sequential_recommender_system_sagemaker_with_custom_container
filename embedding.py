from src.components.embed.image_embed import ImageEmbedding
from src.components.embed.text_embed import TextEmbedding
from src.module.resource_manager import EmbedResourceManager, ResourceManager, EMBED_MANAGER
from src.module.db.bigquery import BigqueryDB
from src.components.similarity.similarity import get_similarity
import asyncio
import json
import pickle
import os
from src.module.resource_manager.embed.base import DEFAULT_CONVERT_JSON_PATH, ROOT_DIR
import PIL
from PIL import Image, ImageFile
from tqdm import tqdm
import argparse
import tarfile
import torch
from os.path import dirname
import gc
import shutil
ImageFile.LOAD_TRUNCATED_IMAGES = True


EMBEDDING_DIR = "/opt/ml/processing"
TEST_IMAGE_DIR = "/home"
INPUT_MODEL_DIR = f"{EMBEDDING_DIR}/input_model"
WHOLE_EMBEDDING_DIR = f"{EMBEDDING_DIR}/whole_embedding"
OUTPUT_EMBEDDING_DIR = f"{EMBEDDING_DIR}/output_embedding"
TEXT_EMBED_MODEL = f"{EMBEDDING_DIR}/text_embedding_model"
OUTPUT_HASH_DIR = f"{EMBEDDING_DIR}/output_hash"
OUTPUT_SIM_DIR = f"{EMBEDDING_DIR}/output_similarity"
OUTPUT_MODEL_DIR = f"{EMBEDDING_DIR}/output_model"

async def main():
    # download image
    embedd_manager = ResourceManager().get_manager(EMBED_MANAGER)

    await embedd_manager.download_image() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_content", type=int, default=300, help="maximum number of similar content")
    parser.add_argument("--batch_size", type=int, default=300, help='matrix multiplication split length for calculating similarity' )
    parser.add_argument("--bulk_size", type=int, default=100000, help='maximum number of measuring similarity')
    parameters = parser.parse_args()
    # unzip 
    tar_gz2_file = tarfile.open(os.path.join(INPUT_MODEL_DIR, "model.tar.gz"))
    tar_gz2_file.extractall(path=WHOLE_EMBEDDING_DIR)
    tar_gz2_file.close()
    tar_gz2_file = tarfile.open(os.path.join(TEXT_EMBED_MODEL, "text_embedding_model.tar.gz"))
    tar_gz2_file.extractall(path=EMBEDDING_DIR)
    tar_gz2_file.close()
    # load data
    bigquery = BigqueryDB()
    
    with open(os.path.join(WHOLE_EMBEDDING_DIR, "hash_keys.json"), "r") as f:
        before_hash_keys = json.load(f)
    content = bigquery.get_content_data()
    before_embedding = pickle.load(open(os.path.join(WHOLE_EMBEDDING_DIR, "whole_embedding.pickle"), "rb"))
    
    # 이전에 이미지 임베딩 구한 해시값들은 제외
    content_dic = {}
    for story_id, cover_image in tqdm(zip(content["id"], content["cover_image"])):
        try:
            if before_hash_keys[str(story_id)]["hash_key"] != json.loads(cover_image)["key"]:
                content_dic[str(story_id)] = json.loads(cover_image) 
        except KeyError:
            content_dic[str(story_id)] = json.loads(cover_image)
 
    # download image
    if not os.path.isdir(ROOT_DIR):
        if not os.path.isdir('temp'):
            os.mkdir('temp')
        if not os.path.isdir('temp/data'):
            os.mkdir('temp/data')   
        if not os.path.isdir('temp/data/story_cover'):
            os.mkdir('temp/data/story_cover')   
    with open(DEFAULT_CONVERT_JSON_PATH, "w") as f:
        json.dump(content_dic, f)
    # 압데이트 했거나 새로 업로드 
    asyncio.run(main())

    # image_embedding
    image_embedding = ImageEmbedding()
    image_dir = f"{ROOT_DIR}/images"
    test_image = Image.open(os.path.join(TEST_IMAGE_DIR, 'test_image.jpeg')).convert("RGB")
    test_emb = image_embedding.run(test_image)
    img_emb_size = test_emb.shape[0]
    img_emb_id_list = []
    for img_file in tqdm(os.listdir(image_dir)):
        try:
            story_id = int(img_file.split(".")[0])
        except ValueError: # story_id일 경우만
            continue
        try:
            image = Image.open(os.path.join(image_dir, img_file)).convert("RGB")
        except PIL.UnidentifiedImageError: ## 0바이트라는 의미
            continue
        # 이미지 해시값이 바뀐 임베딩만 업데이트 한다 새로 업로드 된 작품이면 생성한다
        tmp_emb = image_embedding.run(image)
        img_emb_id_list.append(story_id)
        try:
            # 업데이트 된경우
            before_embedding[story_id].update({"image_embedding":tmp_emb})
        except KeyError:
            # 새로 업로드 한 경우 
            before_embedding[story_id] = {"image_embedding":tmp_emb}
        try:
            before_hash_keys[str(story_id)].update({"hash_key":content_dic[str(story_id)]["key"]})
        except KeyError:
            before_hash_keys[str(story_id)] = {"hash_key":content_dic[str(story_id)]["key"]}
    # img_embedding 없음 None으로 채움 - 새로 등록됐지만 임베딩이 없는 경우
    img_story_id_list = [int(story_id) for story_id, dic in before_embedding.items() if "image_embedding" in dic.keys()]
    none_img_id_list = list(set([int(story_id) for story_id in content["id"]]) - set(img_story_id_list))
    for story_id in none_img_id_list:
        try:
            before_embedding[story_id].update({"image_embedding":None})
        except KeyError:
            before_embedding[story_id] = {"image_embedding":None}
    
    # text_embedding 
    text_embedding = TextEmbedding(local_path=TEXT_EMBED_MODEL)
    # tmp = {int(story_id):text for story_id, text in zip(content["id"], content["text"]) if str(story_id) in before_hash_keys.keys()}
    text_dic = {int(story_id):text for story_id, text in zip(content["id"], content["text"])}
    test_text = "안녕"
    test_emb = text_embedding.run(test_text)
    text_emb_size = test_emb.shape[0]
    text_emb_id_list = []
    none_text_id_list = []
    for story_id, text in tqdm(text_dic.items()):
        try:
            if before_hash_keys[str(story_id)]["text"] == text:
                continue
            else: # 제목 업데이트
                tmp_emb = text_embedding.run(text)
        except KeyError: # 새로 업로드
            
            # 텍스트가 바뀐 경우에만 업데이트 한다 새로 업로드한 작품이면 생성
            tmp_emb = text_embedding.run(text)
        if isinstance(tmp_emb, torch.Tensor):
            text_emb_id_list.append(story_id)
        else:
            # 새로 업로드 했지만 임베딩 없는 경우
            none_text_id_list.append(story_id)
        try:
            before_embedding[story_id].update({
                                                "text_embedding":tmp_emb
            })
        except KeyError:
            before_embedding[story_id] = {"text_embedding":tmp_emb}
        try:
            before_hash_keys[str(story_id)].update({"text":text})
        except KeyError:
            before_hash_keys[str(story_id)] = {"text":text}
    # concat - 바뀐 경우는 이미지 2개, 텍스트 2개 해서 2*2=4가지의 경우의 수고 각각 for문 돌린다.
    img_tmp = torch.FloatTensor([0.0 for i in range(img_emb_size)])
    text_tmp = torch.FloatTensor([0.0 for i in range(text_emb_size)])
    for story_id in set(img_emb_id_list) & set(text_emb_id_list):
        # 둘 다 임베딩 있는 경우
        before_embedding[story_id].update({'whole_embedding':torch.cat([before_embedding[story_id]['image_embedding'], before_embedding[story_id]['text_embedding']])})
    for story_id in set(img_emb_id_list) & set(none_text_id_list):
        # 이미지 임베딩 있고 텍스트 임베딩 없는 경우 
        before_embedding[story_id].update({'whole_embedding':torch.cat([before_embedding[story_id]['image_embedding'], text_tmp])})
    for story_id in set(none_img_id_list) & set(text_emb_id_list):
        # 이미지 임베딩 없고 텍스트 임베딩 있는 경우 
        before_embedding[story_id].update({'whole_embedding':torch.cat([img_tmp, before_embedding[story_id]['text_embedding']])})
    for story_id in set(none_img_id_list) & set(none_text_id_list):
        before_embedding[story_id].update({'whole_embedding':torch.cat([img_tmp, text_tmp])})
    # # save embedding, hash_key
    # with open(os.path.join(WHOLE_EMBEDDING_DIR, "whole_embedding.pickle"), "wb") as f:
    #     pickle.dump(before_embedding, f)
    # with open(os.path.join(WHOLE_EMBEDDING_DIR, "hash_keys.json"), "w") as f:
    #     json.dump(before_hash_keys, f)
    with open("whole_embedding.pickle", "wb") as f:
        pickle.dump(before_embedding, f)
    with open(os.path.join(OUTPUT_EMBEDDING_DIR, "whole_embedding.pickle"), "wb") as f:
        pickle.dump(before_embedding, f)
    with open(os.path.join(OUTPUT_HASH_DIR, "hash_keys.json"), "w") as f:
        json.dump(before_hash_keys, f)
    
    # get_similarity
    feature_df = content[["id", "tap_count", "view_count", "cheer_count", "comment_count"]]
    feature_df.rename(columns={'id': 'story_id'}, inplace=True)
    # shuffle ramdomly for split
    feature_df = feature_df.sample(frac=1).reset_index(drop=True)
    list_feature_df = [feature_df.loc[i:i+parameters.bulk_size-1,:] for i in range(0,len(feature_df), parameters.bulk_size)]
    
    index_list = []
    for idx, feature_df in enumerate(list_feature_df):
        
        content_sim_dic = get_similarity(feature_df=feature_df, embeddings=before_embedding, num_content=parameters.num_content, batch_size=parameters.batch_size)
        # save similarity oom error 막기 위해 나눠서 저장
        with open(os.path.join(EMBEDDING_DIR, f"sim_dic_{idx}.pickle"), "wb") as f:
            pickle.dump(content_sim_dic, f)
        index_list.append(idx)
        del content_sim_dic
        gc.collect()
    content_sim_dic = {}
    for idx in index_list:
        tmp = pickle.load(open(os.path.join(EMBEDDING_DIR, f"sim_dic_{idx}.pickle"), "rb"))
        content_sim_dic.update(tmp)
    # with open(os.path.join(WHOLE_EMBEDDING_DIR, "sim_dic.pickle"), "wb") as f:
    #     pickle.dump(content_sim_dic, f)
    with open("sim_dic.pickle", "wb") as f:
        pickle.dump(content_sim_dic, f)
    with open(os.path.join(OUTPUT_SIM_DIR, "sim_dic.pickle"), "wb") as f:
        pickle.dump(content_sim_dic, f)
    # zip 
    with tarfile.open(os.path.join(OUTPUT_MODEL_DIR, "model.tar.gz"), "w:gz") as mytar:
        # 경로 뺀 파일만 압축
        # mytar.add(os.path.join(OUTPUT_HASH_DIR, "hash_keys.json"))
        for file in os.listdir(WHOLE_EMBEDDING_DIR):
            if file == 'hash_keys.json' or file == 'model.tar.gz':
                continue
            tarinfo = mytar.gettarinfo(os.path.join(WHOLE_EMBEDDING_DIR, file), file)
            mytar.addfile(tarinfo, open(os.path.join(WHOLE_EMBEDDING_DIR, file), 'rb'))
        mytar.add("whole_embedding.pickle")
        mytar.add("sim_dic.pickle")
