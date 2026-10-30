"""Классификация обращений RUDATA: интерфейс, продукт и тематика, уверенность по каждому полю
и признак ручной проверки.

Режимы:
- hybrid (по умолчанию) — интерфейс и продукт определяет градиентный бустинг, тематику — языковая модель
  qwen3:8b через Ollama. Уверенность в интерфейсе и продукте — согласие бустинга и языковой модели
  (1 — ответы совпали, 0 — разошлись), уверенность в тематике — вероятность ответа языковой модели;
- classic — только бустинг, Ollama не нужна; тематика в этом режиме определяется заметно хуже.

Модели и настройки берутся из папки models: classic_model.pkl (TF-IDF, SVD и модели бустинга),
llm_params.json (промпт, пороги уверенности, состав гибрида), access.json (логин → подключённые продукты).
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import requests
from scipy.sparse import hstack

MODELS_DIR = Path(__file__).parent / 'models'
OLLAMA_URL = 'http://localhost:11434'

FIELDS = ['interface', 'product', 'topic']
SPECIFIC_PRODUCTS = ['RD_RISK', 'RD_SOLVENCY', 'RDFOR', 'RD_SCORE', 'RD_AFFILIATE', 'SPPI_TEST']
MODES = ['hybrid', 'classic']

# загружаем модели и настройки один раз — при импорте модуля
classic = joblib.load(MODELS_DIR / 'classic_model.pkl')
llm_params = json.loads((MODELS_DIR / 'llm_params.json').read_text(encoding='utf-8'))
access = json.loads((MODELS_DIR / 'access.json').read_text(encoding='utf-8'))
classes = classic['classes']


# ---------- признаки для бустинга ----------

def clean_text(text):
    text = text.lower().replace('ё', 'е')
    text = re.sub(r'\s+', ' ', text)  # переносы строк и двойные пробелы — в один пробел
    return text.strip()


def remove_context(text):
    """Убирает названия продуктов, интерфейсов и предметные слова — признаки тематики строятся без них"""
    text = re.sub(classic['context_pattern'], ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def make_features(text):
    """Очищенный текст → TF-IDF → 300 столбцов SVD (для интерфейса и продукта)"""
    tfidf = hstack([classic['tfidf_word'].transform([text]), classic['tfidf_char'].transform([text])])
    return classic['svd'].transform(tfidf.tocsr().astype(np.float32))


def make_features_topic(text):
    """Очищенный текст без слов контекста → TF-IDF → 300 столбцов SVD (для тематики)"""
    tfidf = hstack([classic['tfidf_word_topic'].transform([text]), classic['tfidf_char_topic'].transform([text])])
    return classic['svd_topic'].transform(tfidf.tocsr().astype(np.float32))


def apply_access(probs, available):
    """Обнуляет вероятности продуктов, не подключённых логину, и пересчитывает остальные"""
    probs = probs.copy()
    for col, code in enumerate(classes['product']):
        if code in SPECIFIC_PRODUCTS and code not in available:
            probs[col] = 0
        if code == 'UNKNOWN' and len(available) <= 1:  # в разметке UNKNOWN бывает только при 2+ продуктах
            probs[col] = 0
    return probs / probs.sum()


def predict_classic(text, available):
    text_clean = clean_text(text)
    features = {'interface': make_features(text_clean),
                'product': make_features(text_clean),
                'topic': make_features_topic(remove_context(text_clean))}
    pred = {}
    for field in FIELDS:
        probs = classic['models'][field].predict_proba(features[field])[0]
        if field == 'product':
            probs_text = probs  # вероятности только по тексту
            probs = apply_access(probs, available)
            pred['product_by_access'] = bool(probs.argmax() != probs_text.argmax())
        best = int(probs.argmax())
        pred[field] = {'value': classes[field][best], 'confidence': float(probs[best])}
    return pred


# ---------- языковая модель через Ollama ----------

def format_products(available):
    return ', '.join(available) if available else 'нет подключённых продуктов'


def user_message(text, available):
    return f'Подключённые продукты клиента: {format_products(available)}\nОбращение: {text}'


def allowed_products(available):
    allowed = [code for code in classes['product'] if code in available]
    allowed.append('NO_PRODUCT')
    if len(available) >= 2:
        allowed.append('UNKNOWN')
    return allowed


def answer_schema(available):
    return {'type': 'object',
            'properties': {'interface': {'type': 'string', 'enum': classes['interface']},
                           'product': {'type': 'string', 'enum': allowed_products(available)},
                           'topic': {'type': 'string', 'enum': classes['topic']}},
            'required': FIELDS}


def ask_llm(text, available):
    response = requests.post(f'{OLLAMA_URL}/api/chat', timeout=600, json={
        'model': llm_params['model'],
        'messages': [{'role': 'system', 'content': llm_params['system_prompt']},
                     {'role': 'user', 'content': user_message(text, available)}],
        'format': answer_schema(available),
        'options': {'temperature': 0, 'seed': 42, 'num_ctx': llm_params['num_ctx']},
        'logprobs': True,
        'think': False,
        'stream': False,
        'keep_alive': '30m',
    })
    response.raise_for_status()
    result = response.json()
    if not result.get('logprobs'):
        raise ValueError('Ollama не вернула вероятности токенов (logprobs) — нужна более новая версия Ollama')
    return result['message']['content'], result['logprobs']


def field_probability(tokens, field):
    """Вероятность, с которой модель написала значение поля: произведение вероятностей его токенов"""
    text = ''.join(token['token'] for token in tokens)
    value = re.search(rf'"{field}"\s*:\s*"([^"]*)"', text)
    start, end = value.span(1)

    log_probability = 0
    position = 0
    for token in tokens:
        token_end = position + len(token['token'])
        if token_end > start and position < end:  # токен попадает на значение поля
            log_probability += token['logprob']
        position = token_end
    return float(np.exp(log_probability))


def predict_llm(text, available):
    content, tokens = ask_llm(text, available)
    answer = json.loads(content)
    return {field: {'value': answer[field], 'confidence': field_probability(tokens, field)} for field in FIELDS}


def ollama_status():
    """(True, '') — Ollama запущена и модель скачана; иначе (False, что не так)"""
    try:
        models = requests.get(f'{OLLAMA_URL}/api/tags', timeout=5).json()['models']
    except requests.RequestException:
        return False, 'Ollama не запущена'
    if llm_params['model'] not in [model['name'] for model in models]:
        return False, f'в Ollama нет модели {llm_params["model"]} — выполните: ollama pull {llm_params["model"]}'
    return True, ''


# ---------- итоговый ответ ----------

def is_consistent(values):
    if values['interface'] in ['API', 'WEB', 'EFIR', 'EXCEL_ADDIN']:
        return values['product'] == 'NO_PRODUCT'
    if values['interface'] in ['EXCEL_TEMPLATE', 'UNKNOWN']:
        return values['product'] != 'NO_PRODUCT'
    return True  # NONE — продукт может быть любым


def classify(text, login, mode='hybrid'):
    """Обращение и логин → три поля с уверенностью и признак ручной проверки"""
    known_login = login in access
    available = access.get(login, [])
    classic_pred = predict_classic(text, available)

    if mode == 'hybrid':
        llm_pred = predict_llm(text, available)
        pred = {}
        for field in FIELDS:
            source = classic_pred if llm_params['hybrid_source_by_field'][field] == 'classic' else llm_pred
            value = source[field]['value']
            if field in llm_params['agreement_fields']:
                confidence = 1.0 if classic_pred[field]['value'] == llm_pred[field]['value'] else 0.0
            else:
                confidence = source[field]['confidence']
            pred[field] = {'value': value, 'confidence': confidence}
        product_by_access = (classic_pred['product_by_access']
                             if llm_params['hybrid_source_by_field']['product'] == 'classic' else False)
    else:
        pred = {field: classic_pred[field] for field in FIELDS}
        product_by_access = classic_pred['product_by_access']

    thresholds = llm_params['thresholds'][mode]
    values = {field: pred[field]['value'] for field in FIELDS}
    reasons = []

    # правило заказчика: нет уверенности в продукте — UNKNOWN, а не угаданный продукт
    if pred['product']['confidence'] < thresholds['product']:
        values['product'] = 'UNKNOWN'
    # неизвестный логин — права не предполагаем
    if not known_login:
        values['product'] = 'UNKNOWN'
        reasons.append('логин не найден в справочнике доступов')

    unknown_fields = [field for field in FIELDS if values[field] == 'UNKNOWN']
    if unknown_fields:
        reasons.append('UNKNOWN: ' + ', '.join(unknown_fields))
    low_fields = [field for field in FIELDS if pred[field]['confidence'] < thresholds[field]]
    if low_fields:
        reasons.append('низкая уверенность: ' + ', '.join(low_fields))
    if not is_consistent(values):
        reasons.append('поля противоречат друг другу')

    return {
        'interface': {'value': values['interface'], 'confidence': round(pred['interface']['confidence'], 4)},
        'product': {'value': values['product'], 'confidence': round(pred['product']['confidence'], 4),
                    'by_access': product_by_access and values['product'] != 'UNKNOWN'},
        'topic': {'value': values['topic'], 'confidence': round(pred['topic']['confidence'], 4)},
        'manual_review': bool(reasons),
        'review_reasons': reasons,
        'mode': mode,
    }
