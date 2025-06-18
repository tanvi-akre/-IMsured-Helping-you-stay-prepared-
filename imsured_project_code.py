
# IMsured – Helping You Stay Prepared
# Project Code for Statistical and Machine Learning Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats
from sklearn.decomposition import PCA
import skfuzzy as fuzz

# Load dataset (replace with your actual file path)
# df = pd.read_excel("imsured_dataset.xlsx")
# For demo, we'll assume df is already loaded and preprocessed

# ----------------------------
# 1. K-Nearest Neighbors (KNN)
# ----------------------------
X_knn = df.drop("Made_Claim", axis=1)
y_knn = df["Made_Claim"]
X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn, test_size=0.2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# ----------------------------
# 2. Chi-Square Tests
# ----------------------------
# Claim Behaviour
contingency_1 = pd.crosstab(df["Area_Type"], df["Made_Claim"])
chi2, p, dof, expected = stats.chi2_contingency(contingency_1)
print(f"Chi-Square (Claim Behaviour): {chi2:.4f}, P-value: {p:.4f}")

# Claim Settlement
contingency_2 = pd.crosstab(df["Area_Type"], df["Last_Claim_Status"])
chi2, p, dof, expected = stats.chi2_contingency(contingency_2)
print(f"Chi-Square (Claim Settlement): {chi2:.4f}, P-value: {p:.4f}")

# ----------------------------
# 3. Random Forest
# ----------------------------
target = "Overall_Satisfaction"
features = [col for col in df.columns if col != target]
X_rf = df[features]
y_rf = df[target].astype("category").cat.codes
X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ----------------------------
# 4. Multinomial Logistic Regression
# ----------------------------
X_log = df[["Last_Claim_Status", "Overall_Satisfaction", "Policy_Premium", "Area_Type", "Employment_Status", "Dependents"]]
y_log = df["Switch_To_Government"]
X_scaled = StandardScaler().fit_transform(X_log)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)
log_model = LogisticRegression(penalty='l2', solver='liblinear', multi_class='ovr', random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# ----------------------------
# 5. Clustering (Fuzzy C-Means)
# ----------------------------
cluster_features = ['Age', 'Income_Level', 'Employment_Status', 'Ambulance_Charges',
                    'Cashless_Treatment', 'Critical_Illness_Coverage', 'Daycare_Procedures',
                    'Domiciliary_Treatment', 'Hospitalization_Expenses', 'Maternity_Newborn_Coverage',
                    'Mental_Health_Treatment', 'No_Claim_Bonus', 'Personal_Accident_Coverage',
                    'Preventive_Health_Checkups', 'Room_Rent_Coverage']
X_cluster = StandardScaler().fit_transform(df[cluster_features])
X_pca = PCA(n_components=0.95).fit_transform(X_cluster)
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_pca.T, c=2, m=2, error=0.005, maxiter=1000)
df["Cluster"] = np.argmax(u, axis=0)
print("Fuzzy Clustering Complete. Cluster assignments saved in df['Cluster']")
