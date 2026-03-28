.PHONY: setup scrape features train audit api dashboard test

setup:
	python -m venv venv
	venv/bin/pip install -r requirements.txt
	venv/bin/python -m spacy download en_core_web_sm
	cp -n .env.example .env || true
	@echo "\nSetup complete. Edit .env then run: make scrape"

scrape:
	python src/ingestion/scraper_careerjunction.py
	python src/ingestion/downloader_statssa.py

features:
	python src/processing/cleaner.py
	python src/processing/feature_engineer.py

train:
	python src/models/skill_demand_forecast.py
	python src/models/degree_classifier.py
	python src/models/unemployment_risk.py

audit:
	python src/evaluation/bias_audit.py

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v

mlflow-ui:
	mlflow ui --port 5000

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
