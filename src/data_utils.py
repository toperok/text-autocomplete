import yaml
import pandas as pd
import re
import os
import html
import urllib.request
from sklearn.model_selection import train_test_split

def clean_string(text):

    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def run_pipeline(config_path='configs/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    url = cfg['paths']['raw_data_url']
    data_dir = cfg['paths']['data_dir']
    input_path = os.path.join(data_dir, 'tweets.txt')

    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"Скачиваю данные из {url}...")
        try:
            urllib.request.urlretrieve(url, input_path)
            print(f"Файл успешно сохранен в {input_path}")
        except Exception as e:
            print(f"Не удалось скачать файл: {e}")
            return
    else:
        print(f"Файл {input_path} уже существует.")

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    cleaned_lines = [clean_string(line) for line in lines]
    cleaned_lines = [line for line in cleaned_lines if len(line) > 2]
    
    df = pd.DataFrame(cleaned_lines, columns=['text'])

    processed_path = cfg['paths']['processed_path']
    df.to_csv(processed_path, index=False)
    print(f"Общий очищенный файл сохранен: {processed_path}")

    s = cfg['split']
    train_val, test = train_test_split(df, test_size=s['test_size'], random_state=s['random_state'])
    val_rel = s['val_size'] / (s['train_size'] + s['val_size'])
    train, val = train_test_split(train_val, test_size=val_rel, random_state=s['random_state'])

    train.to_csv(cfg['paths']['train_path'], index=False)
    val.to_csv(cfg['paths']['val_path'], index=False)
    test.to_csv(cfg['paths']['test_path'], index=False)

    print(f"Готово! Данные нарезаны. Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")