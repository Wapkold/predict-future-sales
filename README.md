# Прогноз продаж (Kaggle: Predict Future Sales)

Данные: https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data

## Запуск
```bash
pip install -r requirements.txt
kaggle competitions download -c competitive-data-science-predict-future-sales -p data
unzip data/competitive-data-science-predict-future-sales.zip -d data
python forecast.py --data data --out output
```

Результат в `output/`:
- `submission.csv` — прогноз LightGBM для отправки на Kaggle
- `submission_baseline.csv` — прогноз скользящего среднего (3 мес.)
- `metrics.json` — RMSE на валидационном месяце 33 (октябрь 2015)
- `feature_importance.csv` — важность признаков

## Модели
- **Baseline**: скользящее среднее продаж пары магазин–товар за 1, 3 и 6 месяцев.
- **LightGBM**: лаги продаж (1, 2, 3, 6, 12 мес.), лаги средних по товару, магазину, категории и магазину+категории, скользящие среднее/std за 3 месяца, тренд, возраст товара, месяц года.

Целевая переменная обрезается в диапазон 0–20, как в метрике соревнования.
