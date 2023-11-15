
from sklearn.datasets import fetch_openml
import pandas as pd

# Load titanic dataset from openml
titanic = fetch_openml(name='titanic', version=1, as_frame=True)

# Save titanic dataset as csv file
titanic.data.to_csv('titanic.csv', index=False)
titanic.target.to_csv('titanic_target.csv', index=False)
