"""Уведомления о всплесках обращений.

Запуск:   python spikes.py [файл.csv]
Без файла берутся все результаты пакетной классификации из папки results (*_classified.csv).
Во входных данных нужны столбцы created_at (дата и время поступления), interface, product, topic.

Метод (подробно — в ноутбуке spikes.ipynb): окно — календарный день; норма — среднее за 28 предыдущих дней
того же типа (будни или выходные), только по прошлому; первые 28 дней — накопление нормы. Всплеск — не меньше
MIN_COUNT обращений за день, хотя бы в MIN_RATIO раз больше нормы и вероятность получить столько случайно
меньше ALPHA (распределение Пуассона). Результат — results/spikes.csv и список уведомлений на экране.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FEATURES = [['interface'], ['product'], ['topic'], ['interface', 'topic'], ['product', 'topic']]

BASELINE_DAYS = 28  # норма — по 28 предыдущим дням
WARMUP_DAYS = 28  # первые 28 дней — накопление нормы, без уведомлений
MIN_COUNT = 10  # не меньше 10 обращений за день
MIN_RATIO = 2  # хотя бы вдвое больше нормы
ALPHA = 0.0001  # вероятность получить столько случайно — меньше 0,01%
MIN_BASELINE = 0.5  # норма «почти ноль» считается равной 0,5 обращения в день


def poisson_tail(count, rate):
    """Вероятность получить count или больше обращений, если в среднем их rate"""
    below = sum(math.exp(-rate) * rate ** k / math.factorial(k) for k in range(count))
    return max(0.0, 1 - below)


def daily_counts(data, feature, days):
    value = data[feature].astype(str).agg(' × '.join, axis=1)
    return data.groupby([data['date'], value]).size().unstack(fill_value=0).reindex(days, fill_value=0)


def baseline(counts):
    norm = pd.DataFrame(np.nan, index=counts.index, columns=counts.columns)
    for i, day in enumerate(counts.index):
        past = counts.index[max(0, i - BASELINE_DAYS):i]  # только предыдущие дни
        same_type = [d for d in past if (d.weekday() >= 5) == (day.weekday() >= 5)]  # будни с буднями, выходные с выходными
        if same_type:
            norm.loc[day] = counts.loc[same_type].mean()
    return norm


def detect(data):
    # полный календарь: день без обращений — ноль
    days = list(pd.date_range(data['date'].min(), data['date'].max()).date)
    if len(days) <= WARMUP_DAYS:
        print(f'Нужна история больше {WARMUP_DAYS} дней, сейчас {len(days)} — норму не из чего посчитать')
        return pd.DataFrame()
    rows = []
    for feature in FEATURES:
        counts = daily_counts(data, feature, days)
        norm = baseline(counts)
        for day in days[WARMUP_DAYS:]:
            for value in counts.columns:
                count, usual = int(counts.at[day, value]), norm.at[day, value]
                rate = max(usual, MIN_BASELINE)
                if count >= MIN_COUNT and count >= MIN_RATIO * rate:
                    p_value = poisson_tail(count, rate)
                    if p_value < ALPHA:
                        rows.append({'date': day, 'feature': ' × '.join(feature), 'value': value, 'count': count,
                                     'usual': round(usual, 1), 'ratio': round(count / rate, 1), 'p_value': p_value})
    return pd.DataFrame(rows)


def notification(alert):
    usual = 'меньше 1' if alert['usual'] < 1 else f'{alert["usual"]:.1f}'.replace('.', ',')
    ratio = f'{alert["ratio"]:g}'.replace('.', ',')
    return (f'{alert["date"]:%d.%m.%Y} | {alert["feature"]} = {alert["value"]}: обращений за день — {alert["count"]}, '
            f'обычно {usual} в день, в {ratio} раза больше нормы')


def read_csv(path):
    for encoding in ['utf-8-sig', 'cp1251']:
        try:
            return pd.read_csv(path, sep=None, engine='python', encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'{path.name}: не удалось прочитать файл — сохраните его как «CSV UTF-8»')


def main():
    parser = argparse.ArgumentParser(description='Уведомления о всплесках обращений RUDATA')
    parser.add_argument('file', nargs='?', help='CSV-файл; по умолчанию — все *_classified.csv из папки results')
    args = parser.parse_args()

    files = [Path(args.file)] if args.file else sorted((HERE / 'results').glob('*_classified.csv'))
    if not files:
        print('Нет данных: укажите CSV-файл или сначала запустите пакетную классификацию')
        return
    data = pd.concat([read_csv(path) for path in files], ignore_index=True)
    missing = [column for column in ['created_at', 'interface', 'product', 'topic'] if column not in data.columns]
    if missing:
        print(f'Нет столбцов {missing}: для всплесков нужна дата поступления created_at и три поля')
        return

    created_at = pd.to_datetime(data['created_at'], utc=True)
    data['date'] = created_at.dt.tz_convert('Europe/Moscow').dt.date
    alerts = detect(data.dropna(subset=['interface', 'product', 'topic']))
    if alerts.empty:
        print('Всплесков не найдено')
        return

    alerts['notification'] = alerts.apply(notification, axis=1)
    out_path = HERE / 'results' / 'spikes.csv'
    out_path.parent.mkdir(exist_ok=True)
    alerts.to_csv(out_path, index=False, sep=';', encoding='utf-8-sig')
    print(f'Уведомлений: {len(alerts)}, дней с уведомлениями: {alerts["date"].nunique()} → {out_path}')
    for text in alerts['notification']:
        print(text)


if __name__ == '__main__':
    main()
