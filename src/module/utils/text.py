
import re
import emoji
from soynlp.normalizer import repeat_normalize
import unicodedata


def humanize(text: str) -> str:
    # noinspection DuplicatedCode
    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')

    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    url_pattern2 = re.compile(r'http[s]?\s?(com|kr)?\w*')
    url_pattern3 = re.compile(r'w{3}?\s?(com|kr)?\w*')

    phone_pattern = re.compile(r'010\s?[\w|\d]{4}\s?[\w|\d]{4}')

    email_pattern = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')

    num = re.compile(r'([일|이|삼|사|오|육|칠|팔|구|십]*\s?[십|백|천|만|억]+\s?)+')

    text = pattern.sub(' ', text)
    text = url_pattern.sub('', text)
    text = text.strip('\n')
    text = email_pattern.sub('', text)
    text = phone_pattern.sub('번호', text)
    text = url_pattern2.sub(' ', text)
    text = url_pattern3.sub(' ', text)
    text = num.sub('숫자', text)
    text = unicodedata.normalize('NFC', text)
    text = text.strip()
    text = repeat_normalize(text, num_repeats=2)

    return text
