import os
from kaggle.api.kaggle_api_extended import KaggleApi

try:
    api = KaggleApi()
    api.authenticate()
    competitions = api.competitions_list(search='titanic')
    print("Successfully connected to Kaggle API")
    for comp in competitions:
        print(f"- {comp.ref}: {comp.title}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
