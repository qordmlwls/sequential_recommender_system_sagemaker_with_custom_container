from typing import Tuple
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def refine_interaction_data(interaction_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    interaction_data preprocessing
    :param interaction_data: tap수, 응원 데이터
    :return: encoding_data, processed interaction_data
    """
    interaction_data.loc[(interaction_data['event_type'] == 'READ') & (interaction_data['event_value'] < 150), ['event_value']] = 0
    interaction_data.loc[
        (interaction_data['event_type'] == 'READ') & (interaction_data['event_value'] >= 150), ['event_value']] = 1

    # 학습에 쓰기 위해 레이블 인코딩을 해준다
    le = LabelEncoder()
    le.fit(interaction_data.story_id)
    interaction_data['encoding_id'] = le.transform(interaction_data['story_id'])
    interaction_data['encoding_id'] = interaction_data['encoding_id'].astype(int)
    encoding_data = interaction_data[['story_id', 'encoding_id']]
    encoding_data.drop_duplicates(subset=['encoding_id'], inplace=True)
    encoding_data = encoding_data.sort_values(by=['encoding_id'])
    # 학습에 쓰일 수 있게 전처리를 완료하고 리턴한다
    interaction_data = interaction_data[['user_id', 'event_value', 'event_timestamp', 'encoding_id']]
    interaction_data.columns = ['reader_id', 'liked', 'when', 'book_id']
    interaction_data = interaction_data.sort_values(by=['when'])
    return encoding_data, interaction_data

