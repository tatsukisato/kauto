from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

comps = api.competitions_list(search='titanic')
print(f"Found {len(comps)} competitions")
for c in comps:
    print(f"Ref: '{c.ref}'")
    if c.ref == 'titanic':
        print("MATCH FOUND")
        print(f"Title: {c.title}")
        print(f"Description: {c.description}")
        print(f"Metric: {getattr(c, 'evaluationMetric', 'N/A')}")
        break
