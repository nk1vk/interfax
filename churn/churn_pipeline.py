"""Ежедневный прогноз риска расторжения договора клиентами RUDATA (задача 2.2.2).

Без аналитики — только то, что нужно для работы модели:
    python churn_pipeline.py score                    # прогноз на утро после последнего дня выгрузки
    python churn_pipeline.py score --date 2026-09-15  # прогноз на конкретную дату
    python churn_pipeline.py train                    # переобучение модели на свежей выгрузке

Исследование, выбор модели и проверка качества — в ноутбуке churn.ipynb.
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Константы задачи (те же, что в ноутбуке)
RAND_STATE = 42
HORIZON_DAYS = 28  # горизонт прогноза: расторжение в ближайшие 28 дней
WINDOWS = [7, 14, 28, 56]  # окна для признаков активности, в днях
BASELINE_DAYS = 112  # личная норма клиента — дни 29–112 до даты прогноза
MIN_HISTORY_DAYS = 28  # не прогнозируем, пока у клиента меньше 28 дней истории
VALID_MONTHS = 3  # при переобучении: последние 3 месяца с известным исходом — для подбора порога

ACCESS_COLS = ['access_rd_risk', 'access_rd_solvency', 'access_rdfor',
               'access_rd_score', 'access_rd_affiliate', 'access_sppi_test']
ACTIVITY_COLS = ['api_requests', 'product_sessions', 'is_active', 'api_endpoints', 'products_used']

# Признаки модели по типам (для предобработки логистической регрессии)
BINARY_FEATURES = ['api_access_enabled'] + ACCESS_COLS
NUM_FEATURES_LOG = ([f'api_requests_{w}d' for w in WINDOWS] + [f'product_sessions_{w}d' for w in WINDOWS]
                    + ['days_without_activity', 'days_without_api', 'days_without_product',
                       'contract_days'])
NUM_FEATURES = ([f'active_days_{w}d' for w in WINDOWS]
                + [f'api_endpoints_{w}d_mean' for w in WINDOWS]
                + [f'products_used_{w}d_mean' for w in WINDOWS]
                + ['api_requests_trend_7_28', 'product_sessions_trend_7_28', 'active_days_trend_7_28']
                + ['api_requests_vs_norm', 'api_endpoints_vs_norm',
                   'product_sessions_vs_norm', 'products_used_vs_norm']
                + ['n_products_access'])


def load_data(data_dir):
    """Читает три таблицы поставки."""
    clients = pd.read_csv(os.path.join(data_dir, 'clients.csv'), encoding='utf-8-sig',
                          parse_dates=['contract_start_date', 'churn_date', 'outcome_observed_until'])
    api_usage = pd.read_csv(os.path.join(data_dir, 'api_usage_daily.csv'), encoding='utf-8-sig',
                            parse_dates=['date'])
    product_usage = pd.read_csv(os.path.join(data_dir, 'product_usage_daily.csv'), encoding='utf-8-sig',
                                parse_dates=['date'])
    clients['n_products_access'] = clients[ACCESS_COLS].sum(axis=1)
    return clients, api_usage, product_usage


def build_daily(clients, api_usage, product_usage, data_start, calendar_end):
    """Дневная активность каждого клиента за дни действующего договора (дни без строк — нули)."""
    start = clients['contract_start_date'].clip(lower=data_start)
    end = (clients['churn_date'] - pd.Timedelta(days=1)).fillna(calendar_end).clip(upper=calendar_end)

    calendar = pd.concat([pd.DataFrame({'client_id': client_id, 'date': pd.date_range(s, e)})
                          for client_id, s, e in zip(clients['client_id'], start, end)],
                         ignore_index=True)

    api_daily = api_usage.groupby(['client_id', 'date']).agg(
        api_requests=('requests_count', 'sum'),
        api_endpoints=('endpoint_id', 'nunique')).reset_index()
    product_daily = product_usage.groupby(['client_id', 'date']).agg(
        product_sessions=('sessions_count', 'sum'),
        products_used=('product_id', 'nunique')).reset_index()

    daily = calendar.merge(api_daily, on=['client_id', 'date'], how='left')
    daily = daily.merge(product_daily, on=['client_id', 'date'], how='left')

    cols = ['api_requests', 'api_endpoints', 'product_sessions', 'products_used']
    daily[cols] = daily[cols].fillna(0).astype(int)
    daily['is_active'] = ((daily['api_requests'] > 0) | (daily['product_sessions'] > 0)).astype(int)
    return daily.sort_values(['client_id', 'date'], ignore_index=True)


def build_features(daily, clients, data_start):
    """Признаки на утро каждой даты: только по дням до даты прогноза."""
    panel = daily.merge(clients[['client_id', 'contract_start_date']], on='client_id', how='left')
    panel['history_days'] = (panel['date'] - panel['contract_start_date'].clip(lower=data_start)).dt.days

    # суммы за окна по прошлым дням: shift(1) убирает текущий день, rolling суммирует окно
    for w in WINDOWS + [BASELINE_DAYS]:
        for col in ACTIVITY_COLS:
            panel[f'{col}_{w}d'] = panel.groupby('client_id')[col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).sum())

    # дата последней активности по прошлым дням
    for name, col in [('activity', 'is_active'), ('api', 'api_requests'), ('product', 'product_sessions')]:
        panel[f'last_{name}_date'] = panel['date'].where(panel[col] > 0)
        panel[f'last_{name}_date'] = panel.groupby('client_id')[f'last_{name}_date'].transform(
            lambda x: x.shift(1).ffill())

    # средние в день по окнам
    for w in WINDOWS:
        panel = panel.rename(columns={f'is_active_{w}d': f'active_days_{w}d'})
        panel[f'api_endpoints_{w}d_mean'] = panel[f'api_endpoints_{w}d'] / panel['history_days'].clip(upper=w)
        panel[f'products_used_{w}d_mean'] = panel[f'products_used_{w}d'] / panel['history_days'].clip(upper=w)

    # тренд: последняя неделя против средней недели за 28 дней
    for col in ['api_requests', 'product_sessions', 'active_days']:
        panel[f'{col}_trend_7_28'] = (panel[f'{col}_7d'] / (panel[f'{col}_28d'] / 4)).fillna(1)

    # активность за последние 28 дней относительно личной нормы (дни 29–112)
    last_days = panel['history_days'].clip(upper=28)
    norm_days = panel['history_days'].clip(upper=BASELINE_DAYS) - last_days
    for col in ['api_requests', 'api_endpoints', 'product_sessions', 'products_used']:
        last_28 = panel[f'{col}_28d'] / last_days
        norm = (panel[f'{col}_{BASELINE_DAYS}d'] - panel[f'{col}_28d']) / norm_days
        panel[f'{col}_vs_norm'] = last_28 / norm
        panel.loc[(norm_days < 28) | (norm == 0), f'{col}_vs_norm'] = 1

    # дни без активности, срок договора, доступы
    for name in ['activity', 'api', 'product']:
        panel[f'days_without_{name}'] = (panel['date'] - panel[f'last_{name}_date']).dt.days
        panel[f'days_without_{name}'] = panel[f'days_without_{name}'].fillna(panel['history_days'])
    panel['contract_days'] = (panel['date'] - panel['contract_start_date']).dt.days
    panel = panel.merge(clients[['client_id', 'api_access_enabled', 'n_products_access'] + ACCESS_COLS],
                        on='client_id', how='left')

    return panel[panel['history_days'] >= MIN_HISTORY_DAYS].reset_index(drop=True)


def add_target(panel, clients):
    """Таргет: 1 — расторжение в ближайшие 28 дней, 0 — не было, пропуск — исход ещё неизвестен."""
    panel = panel.merge(clients[['client_id', 'churn_date', 'outcome_observed_until']],
                        on='client_id', how='left')
    days_to_churn = (panel['churn_date'] - panel['date']).dt.days

    mask_1 = days_to_churn.between(1, HORIZON_DAYS)
    mask_0 = ~mask_1 & (panel['date'] + pd.Timedelta(days=HORIZON_DAYS) <= panel['outcome_observed_until'])

    panel['target'] = np.nan
    panel.loc[mask_0, 'target'] = 0
    panel.loc[mask_1, 'target'] = 1
    return panel


def make_model(model_name, params):
    """Модель с подобранными в ноутбуке параметрами."""
    if model_name == 'catboost_optuna':
        return CatBoostClassifier(random_seed=RAND_STATE, auto_class_weights='Balanced', verbose=0, **params)

    preprocessor = ColumnTransformer(transformers=[
        ('num_log', Pipeline(steps=[('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                                    ('scaler', StandardScaler())]), NUM_FEATURES_LOG),
        ('num', StandardScaler(), NUM_FEATURES),
        ('binary', 'passthrough', BINARY_FEATURES)
    ], verbose_feature_names_out=False)
    return Pipeline(steps=[
        ('prep', preprocessor),
        ('model', LogisticRegression(class_weight='balanced', max_iter=2000, **params))
    ])


def top_factors(model, model_name, X, n=3):
    """Три признака, сильнее всего повышающих риск по каждой строке."""
    if model_name == 'catboost_optuna':
        contributions = model.get_feature_importance(Pool(X), type='ShapValues')[:, :-1]
        names = X.columns
    else:
        prep = model.named_steps['prep']
        contributions = prep.transform(X) * model.named_steps['model'].coef_[0]
        names = prep.get_feature_names_out()
    contributions = pd.DataFrame(contributions, columns=names, index=X.index)
    return contributions.apply(lambda row: list(row.nlargest(n).index), axis=1)


def train(data_dir, model_dir):
    """Переобучение на свежей выгрузке с параметрами из models/model_params.json."""
    with open(os.path.join(model_dir, 'model_params.json'), encoding='utf-8') as f:
        model_params = json.load(f)

    clients, api_usage, product_usage = load_data(data_dir)
    data_start = min(api_usage['date'].min(), product_usage['date'].min())
    data_end = clients['outcome_observed_until'].max()

    daily = build_daily(clients, api_usage, product_usage, data_start, data_end)
    labeled = add_target(build_features(daily, clients, data_start), clients).dropna(subset=['target'])

    # последние VALID_MONTHS месяцев с известным исходом — для порога, всё раньше (с зазором) — для обучения
    threshold_start = labeled['date'].max() - pd.DateOffset(months=VALID_MONTHS)
    fit_part = labeled[labeled['date'] <= threshold_start - pd.Timedelta(days=HORIZON_DAYS)]
    threshold_part = labeled[labeled['date'] > threshold_start]

    feature_cols = model_params['feature_cols']
    model = make_model(model_params['model'], model_params['params'])
    model.fit(fit_part[feature_cols], fit_part['target'])

    # порог: тревога по alert_share клиенто-дней с самым высоким риском
    risk = model.predict_proba(threshold_part[feature_cols])[:, 1]
    model_params['threshold'] = float(np.quantile(risk, 1 - model_params['alert_share']))

    joblib.dump(model, os.path.join(model_dir, 'churn_model.pkl'))
    with open(os.path.join(model_dir, 'model_params.json'), 'w', encoding='utf-8') as f:
        json.dump(model_params, f, ensure_ascii=False, indent=2)
    print(f'Модель переобучена на данных по {fit_part["date"].max().date()}, '
          f'порог {model_params["threshold"]:.4f}')


def score(data_dir, model_dir, out_dir, run_date=None):
    """Прогноз риска на утро run_date для всех действующих клиентов."""
    with open(os.path.join(model_dir, 'model_params.json'), encoding='utf-8') as f:
        model_params = json.load(f)
    model = joblib.load(os.path.join(model_dir, 'churn_model.pkl'))

    clients, api_usage, product_usage = load_data(data_dir)
    data_start = min(api_usage['date'].min(), product_usage['date'].min())
    data_end = max(api_usage['date'].max(), product_usage['date'].max())
    # по умолчанию — утро дня после последнего дня выгрузки
    run_date = pd.Timestamp(run_date) if run_date else data_end + pd.Timedelta(days=1)

    daily = build_daily(clients, api_usage, product_usage, data_start, run_date)
    today = build_features(daily, clients, data_start)
    today = today[today['date'] == run_date].reset_index(drop=True)

    feature_cols = model_params['feature_cols']
    today['risk'] = model.predict_proba(today[feature_cols])[:, 1]
    today['alert'] = today['risk'] >= model_params['threshold']
    factors = top_factors(model, model_params['model'], today[feature_cols])
    for i in range(3):
        today[f'factor_{i + 1}'] = factors.str[i]

    result = today[['client_id', 'date', 'risk', 'alert', 'factor_1', 'factor_2', 'factor_3']]
    result = result.sort_values('risk', ascending=False)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'churn_risk_{run_date.date()}.csv')
    result.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'Прогноз на {run_date.date()}: клиентов {len(result)}, с тревогой {result["alert"].sum()}. '
          f'Файл: {out_path}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Прогноз риска расторжения договора RUDATA')
    parser.add_argument('command', choices=['score', 'train'], help='score — прогноз, train — переобучение')
    parser.add_argument('--data-dir', default='data/raw', help='папка с clients.csv и таблицами активности')
    parser.add_argument('--model-dir', default='models', help='папка с моделью и model_params.json')
    parser.add_argument('--out-dir', default='results', help='куда сохранить прогноз')
    parser.add_argument('--date', default=None, help='дата прогноза YYYY-MM-DD (по умолчанию — после выгрузки)')
    args = parser.parse_args()

    if args.command == 'train':
        train(args.data_dir, args.model_dir)
    else:
        score(args.data_dir, args.model_dir, args.out_dir, args.date)
