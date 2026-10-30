"""Пакетная классификация обращений из CSV-файлов.

Запуск:   python batch.py [файл.csv] [--mode hybrid|classic]
Без файла обрабатываются все CSV из папки input. Результаты — в папке results, файл <имя>_classified.csv.
Во входном файле нужны столбцы text (текст обращения) и client_id (логин); appeal_id и created_at
(дата поступления — нужна для уведомлений о всплесках, spikes.py) — по желанию.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

import classifier

HERE = Path(__file__).parent


def read_csv(path):
    # CSV из Excel часто сохраняется с разделителем «;» и в кодировке Windows — определяем автоматически
    for encoding in ['utf-8-sig', 'cp1251']:
        try:
            return pd.read_csv(path, sep=None, engine='python', encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'{path.name}: не удалось прочитать файл — сохраните его как «CSV UTF-8»')


def result_row(appeal_id, login, text, result):
    return {'appeal_id': appeal_id, 'client_id': login, 'text': text,
            'interface': result['interface']['value'], 'interface_confidence': result['interface']['confidence'],
            'product': result['product']['value'], 'product_confidence': result['product']['confidence'],
            'product_by_access': result['product']['by_access'],
            'topic': result['topic']['value'], 'topic_confidence': result['topic']['confidence'],
            'manual_review': result['manual_review'], 'review_reasons': '; '.join(result['review_reasons'])}


def process_file(path, mode):
    appeals = read_csv(path)
    missing = [column for column in ['text', 'client_id'] if column not in appeals.columns]
    if missing:
        print(f'{path.name}: нет столбцов {missing} — файл пропущен')
        return

    out_path = HERE / 'results' / f'{path.stem}_classified.csv'
    print(f'{path.name}: {len(appeals)} обращений → {out_path.name}')
    start = time.time()
    for number, row in enumerate(appeals.itertuples(), start=1):
        appeal_id = getattr(row, 'appeal_id', number)
        text, login = str(row.text), str(row.client_id)
        try:
            line = result_row(appeal_id, login, text, classifier.classify(text, login, mode))
        except Exception as error:  # ошибка на одном обращении не останавливает обработку файла
            line = {'appeal_id': appeal_id, 'client_id': login, 'text': text,
                    'manual_review': True, 'review_reasons': f'ошибка классификации: {error}'}
        if hasattr(row, 'created_at'):
            line['created_at'] = row.created_at  # дата поступления — для уведомлений о всплесках
        # каждую строку сразу дописываем в файл: если обработку прервать, готовое не потеряется
        pd.DataFrame([line]).to_csv(out_path, mode='w' if number == 1 else 'a', header=number == 1,
                                    index=False, sep=';', encoding='utf-8-sig' if number == 1 else 'utf-8')
        if number % 10 == 0 or number == len(appeals):
            print(f'  {number} / {len(appeals)}, {time.time() - start:.0f} с')


def main():
    parser = argparse.ArgumentParser(description='Пакетная классификация обращений RUDATA')
    parser.add_argument('file', nargs='?', help='CSV-файл; по умолчанию — все CSV из папки input')
    parser.add_argument('--mode', choices=classifier.MODES, default='hybrid')
    args = parser.parse_args()

    if args.mode == 'hybrid':
        ollama_ok, problem = classifier.ollama_status()
        if not ollama_ok:
            print(f'Режим hybrid недоступен: {problem}. Запустите Ollama или используйте --mode classic')
            return

    (HERE / 'results').mkdir(exist_ok=True)
    files = [Path(args.file)] if args.file else sorted((HERE / 'input').glob('*.csv'))
    if not files:
        print('Нет файлов: положите CSV в папку input или укажите файл')
    for path in files:
        process_file(path, args.mode)


if __name__ == '__main__':
    main()
