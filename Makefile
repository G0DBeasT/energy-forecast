.PHONY: data features train forecast dashboard

data:
	python src/data.py

features:
	python src/features.py

train:
	python src/train.py

forecast:
	python src/forecast.py

dashboard:
	streamlit run app.py
