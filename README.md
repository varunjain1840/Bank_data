import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 1. Prepare Features and Target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# 2. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL 1: LG (Logistic Regression) ---
lg = LogisticRegression(max_iter=1000)
lg.fit(X_train, y_train)
lg_train_acc = accuracy_score(y_train, lg.predict(X_train)) * 100
lg_test_acc = accuracy_score(y_test, lg.predict(X_test)) * 100

# --- MODEL 2: RF (Random Forest) ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_train_acc = accuracy_score(y_train, rf.predict(X_train)) * 100
rf_test_acc = accuracy_score(y_test, rf.predict(X_test)) * 100

# --- MODEL 3: XGB (XGBoost) ---
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_train_acc = accuracy_score(y_train, xgb.predict(X_train)) * 100
xgb_test_acc = accuracy_score(y_test, xgb.predict(X_test)) * 100

# 3. Final Results Summary
print(f"LG: Train {lg_train_acc:.2f}%, Test {lg_test_acc:.2f}%")
print(f"RF: Train {rf_train_acc:.2f}%, Test {rf_test_acc:.2f}%")
print(f"XGB: Train {xgb_train_acc:.2f}%, Test {xgb_test_acc:.2f}%")
