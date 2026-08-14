.PHONY: data features train forecast dashboard clean

data:
	PYTHONPATH=. python src/data.py

features:
	PYTHONPATH=. python src/features.py

train:
	PYTHONPATH=. python src/train.py

forecast:
	PYTHONPATH=. python src/forecast.py

dashboard:
	streamlit run app.py

clean:
	rm -rf data/processed/*.parquet data/processed/*.csv data/processed/*.pkl mlruns mlflow.db
