import inspect
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

comps = api.competitions_list(search='titanic')
for c in comps:
    if c.ref == 'titanic':
        print(f"Title: {c.title}")
        print(f"Ref: {c.ref}")
        print(f"Tags: {c.tags}")
        print(f"Description: {c.description}")
        print(f"Category: {c.category}")
        print(f"Evaluation Metric: {c.evaluationMetric}")
        print("--- Attributes ---")
        print(dir(c))
        break
