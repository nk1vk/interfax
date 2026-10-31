"""Уведомления о всплесках обращений.

Запуск:   python spikes.py [файл.csv] [--window hour|day]
Без файла берутся все результаты пакетной классификации из папки results (*_classified.csv).
Без --window считаются обе версии: по часам и по дням.
Во входных данных нужны столбцы created_at (дата и время поступления), interface, product, topic.

Метод (подробно — в ноутбуке spikes.ipynb): окно — календарный час или день; норма — среднее за 28 предыдущих
дней того же типа (будни или выходные), для часа — в тот же час суток, только по прошлому; первые 28 дней —
накопление нормы. Всплеск — не меньше min_count обращений за окно, хотя бы в MIN_RATIO раз больше нормы
и вероятность получить столько случайно меньше alpha (распределение Пуассона).
Результат — results/spikes_hourly.csv, results/spikes_daily.csv и список уведомлений на экране.
"""
import argparse
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FEATURES = [['interface'], ['product'], ['topic'], ['interface', 'topic'], ['product', 'topic']]

BASELINE_DAYS = 28  # норма — по 28 предыдущим дням
WARMUP_DAYS = 28  # первые 28 дней — накопление нормы, без уведомлений
MIN_RATIO = 2  # хотя бы вдвое больше нормы
MIN_BASELINE = 0.5  # норма «почти ноль» считается равной 0,5 обращения за окно

# остальные пороги зависят от окна: за час обращений в десятки раз меньше, а проверок в 24 раза больше
WINDOWS = {
    'date': {'min_count': 10, 'alpha': 0.0001, 'file': 'spikes_daily.csv'},  # не меньше 10 за день, случайно — меньше 0,01%
    'hour': {'min_count': 5, 'alpha': 0.000001, 'file': 'spikes_hourly.csv'},  # не меньше 5 за час, случайно — меньше 0,0001%
}


def poisson_tail(count, rate):
    """Вероятность получить count или больше обращений, если в среднем их rate"""
    below = sum(math.exp(-rate) * rate ** k / math.factorial(k) for k in range(count))
    return max(0.0, 1 - below)


def periods_of(data, window):
    # полный календарь окон: все дни или все 24 часа каждого дня; окно без обращений — ноль
    first, last = data['date'].min(), data['date'].max()
    if window == 'date':
        return list(pd.date_range(first, last).date)
    return list(pd.date_range(first, pd.Timestamp(last) + pd.Timedelta(hours=23), freq='h'))


def counts_by(data, feature, window):
    value = data[feature].astype(str).agg(' × '.join, axis=1)
    return data.groupby([data[window], value]).size().unstack(fill_value=0).reindex(periods_of(data, window), fill_value=0)


def baseline(counts):
    values = counts.to_numpy(dtype=float)
    position = {period: i for i, period in enumerate(counts.index)}  # номер строки каждого окна
    norm = np.full(values.shape, np.nan)
    for i, period in enumerate(counts.index):
        past = [period - timedelta(days=k) for k in range(1, BASELINE_DAYS + 1)]  # то же окно 1–28 дней назад
        same_type = [position[p] for p in past
                     if p in position and (p.weekday() >= 5) == (period.weekday() >= 5)]  # будни с буднями, выходные с выходными
        if same_type:
            norm[i] = values[same_type].mean(axis=0)
    return pd.DataFrame(norm, index=counts.index, columns=counts.columns)


def window_of(period, window):
    # для дня — дата; для часа — дата и интервал «10:00–11:00»
    if window == 'date':
        return {'date': period}
    return {'date': period.date(), 'hour': f'{period:%H:00}–{period + pd.Timedelta(hours=1):%H:00}'}


def detect(data, window):
    days = periods_of(data, 'date')
    if len(days) <= WARMUP_DAYS:
        print(f'Нужна история больше {WARMUP_DAYS} дней, сейчас {len(days)} — норму не из чего посчитать')
        return pd.DataFrame()
    start = days[WARMUP_DAYS] if window == 'date' else pd.Timestamp(days[WARMUP_DAYS])
    min_count, alpha = WINDOWS[window]['min_count'], WINDOWS[window]['alpha']
    rows = []
    for feature in FEATURES:
        counts = counts_by(data, feature, window)
        norm = baseline(counts)
        for period in counts.index:
            if period < start:
                continue
            for value in counts.columns[counts.loc[period] >= min_count]:  # проверяем только значения, где обращений достаточно
                count, usual = int(counts.at[period, value]), norm.at[period, value]
                rate = max(usual, MIN_BASELINE)
                if count >= MIN_RATIO * rate:
                    p_value = poisson_tail(count, rate)
                    if p_value < alpha:
                        rows.append({**window_of(period, window), 'feature': ' × '.join(feature), 'value': value,
                                     'count': count, 'usual': round(usual, 1), 'ratio': round(count / rate, 1),
                                     'p_value': p_value})
    return pd.DataFrame(rows)


def times_word(ratio):
    # «в 2 раза», «в 3,8 раза», но «в 5 раз», «в 100 раз»
    if ratio != int(ratio):
        return 'раза'
    return 'раза' if int(ratio) % 10 in (2, 3, 4) and int(ratio) % 100 not in (12, 13, 14) else 'раз'


def notification(alert):
    hourly = 'hour' in alert  # у почасовых уведомлений есть интервал hour
    name = 'час' if hourly else 'день'
    when = f'{alert["date"]:%d.%m.%Y}' + (f' {alert["hour"]}' if hourly else '')
    usual = 'меньше 1' if alert['usual'] < 1 else f'{alert["usual"]:.1f}'.replace('.', ',')
    ratio = f'{alert["ratio"]:g}'.replace('.', ',')
    return (f'{when} | {alert["feature"]} = {alert["value"]}: обращений за {name} — {alert["count"]}, '
            f'обычно {usual} в {name}, в {ratio} {times_word(alert["ratio"])} больше нормы')


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
    parser.add_argument('--window', choices=['hour', 'day'], help='только по часам или только по дням; по умолчанию — обе версии')
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

    moscow_time = pd.to_datetime(data['created_at'], utc=True).dt.tz_convert('Europe/Moscow')
    data['date'] = moscow_time.dt.date  # день поступления по Москве
    data['hour'] = moscow_time.dt.floor('h').dt.tz_localize(None)  # час поступления: 10:00 — это 10:00–11:00
    data = data.dropna(subset=['interface', 'product', 'topic'])

    windows = {'hour': ['hour'], 'day': ['date'], None: ['hour', 'date']}[args.window]
    for window in windows:
        title = 'По часам' if window == 'hour' else 'По дням'
        alerts = detect(data, window)
        if alerts.empty:
            print(f'{title}: всплесков не найдено')
            continue
        alerts['notification'] = alerts.apply(notification, axis=1)
        out_path = HERE / 'results' / WINDOWS[window]['file']
        out_path.parent.mkdir(exist_ok=True)
        alerts.to_csv(out_path, index=False, sep=';', encoding='utf-8-sig')
        print(f'\n{title}: уведомлений {len(alerts)}, дней с уведомлениями {alerts["date"].nunique()} → {out_path}')
        for text in alerts['notification']:
            print(text)

if __name__ == '__main__':
    main()
