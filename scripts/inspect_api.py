from kaggle.api.kaggle_api_extended import KaggleApi
import inspect

api = KaggleApi()
print([m for m in dir(api) if not m.startswith('_')])
