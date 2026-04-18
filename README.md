## 💻 Machine Learning Implementation

This section contains the core logic used to train and evaluate the three models. I used a split-test approach (80% training, 20% testing) to ensure the models generalize well to new customer data.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Assuming 'df' is the cleaned banking dataset
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_train_acc = accuracy_score(y_train, log_reg.predict(X_train)) * 100
log_test_acc = accuracy_score(y_test, log_reg.predict(X_test)) * 100

# 2. Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_train_acc = accuracy_score(y_train, rf_clf.predict(X_train)) * 100
rf_test_acc = accuracy_score(y_test, rf_clf.predict(X_test)) * 100

# 3. XGBoost
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)
xgb_train_acc = accuracy_score(y_train, xgb_clf.predict(X_train)) * 100
xgb_test_acc = accuracy_score(y_test, xgb_clf.predict(X_test)) * 100

# Results Summary
print(f"Logistic Regression: Train {log_train_acc:.2f}%, Test {log_test_acc:.2f}%")
print(f"Random Forest: Train {rf_train_acc:.2f}%, Test {rf_test_acc:.2f}%")
print(f"XGBoost: Train {xgb_train_acc:.2f}%, Test {xgb_test_acc:.2f}%")
