import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle

# 1. Prepare Features and Target
X = df.drop(['loan'], axis=1)
y = df['loan'].replace({'no': 0, 'yes': 1})

# 2. Define Column Transformer
ct = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['job']),
    ('ord', OrdinalEncoder(categories=[
        ['divorced', 'married', 'single'],
        ['secondary', 'primary', 'tertiary', 'unknown'],
        ['no', 'yes'],
        ['no', 'yes']
    ]), ['marital', 'education', 'default', 'housing']),
    ('num', MinMaxScaler(), ['age', 'balance', 'loan stats', 'Loan_period'])
], remainder='passthrough')

# 3. Create the Final Pipeline (Example with LG)
pipe = Pipeline([
    ('ct', ct),
    ('ct4', SelectKBest(score_func=chi2, k=12)),
    ('lgr', LogisticRegression(max_iter=1000))
])

# 4. Train and Export
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
pipe.fit(X_train, y_train)
pickle.dump(pipe, open('pipe.pkl', 'wb'))
