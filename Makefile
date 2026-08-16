.PHONY: all data features train forecast evaluate test dashboard clean help

all: data features train forecast evaluate test

data:
	PYTHONPATH=. python scripts/prepare_data.py

features:
	PYTHONPATH=. python src/features/pipeline.py

train:
	PYTHONPATH=. python scripts/train_models.py

forecast:
	PYTHONPATH=. python scripts/generate_forecasts.py

evaluate:
	PYTHONPATH=. python scripts/evaluate_models.py

test:
	PYTHONPATH=. pytest tests/ -v

dashboard:
	streamlit run app.py

clean:
	rm -rf data/processed/*.parquet data/processed/*.csv data/processed/*.pkl data/processed/*.json mlruns mlflow.db .pytest_cache

help:
	@echo "Available Makefile Targets:"
	@echo "  make data      : Ingest, clean, and validate hourly grid & weather data"
	@echo "  make features  : Build leakage-safe hourly and daily feature matrices"
	@echo "  make train     : Run 5-fold TimeSeriesSplit CV training across candidate models"
	@echo "  make forecast  : Generate Next-Day, Next-Week, and Next-Month forecasts"
	@echo "  make evaluate  : Generate structured model selection & evaluation artifacts"
	@echo "  make test      : Execute unit & integration test suite (pytest)"
	@echo "  make dashboard : Launch interactive Streamlit multi-horizon dashboard"
	@echo "  make clean     : Reset processed artifacts, models, and MLflow cache"
