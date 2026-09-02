import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import shap
import gradio as gr

from collections import Counter
from scipy.stats import entropy as scipy_entropy
import joblib

print("All libraries imported successfully!")

# ==============================================================
# LOAD DATASET
# ==============================================================
df = pd.read_excel("Disaster_Datatset.xlsx")
print("Dataset Shape:", df.shape)
print("\nAll Columns:")
print(list(df.columns))
print("\n=== TARGET COLUMN: 'Disaster Type' ===")
print(df['Disaster Type'].value_counts())

# ==============================================================
# MISSING VALUE ANALYSIS
# ==============================================================
missing_counts = df.isnull().sum()
missing_pct    = (df.isnull().mean() * 100).round(2)
missing_df     = pd.DataFrame({'Missing Count': missing_counts, 'Missing %': missing_pct})

missing_plot = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
plt.figure(figsize=(14, 6))
missing_plot['Missing %'].plot(kind='barh', color='coral')
plt.title("Missing Values (%) Before Preprocessing", fontsize=14)
plt.xlabel("Missing %")
plt.tight_layout()
plt.show()

# ==============================================================
# FILTER TO 3 CLASSES
# ==============================================================
target_classes = ['Flood', 'Earthquake', 'Wildfire']
df_filtered = df[df['Disaster Type'].isin(target_classes)].copy()
print("Shape after filtering:", df_filtered.shape)
print(df_filtered['Disaster Type'].value_counts())

plt.figure(figsize=(6, 4))
df_filtered['Disaster Type'].value_counts().plot(kind='bar', color=['pink', 'tomato', 'green'])
plt.title("Selected 3-Class Distribution (Before SMOTE)")
plt.xlabel("Disaster Type")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ==============================================================
# DROP HIGH-MISSING COLUMNS
# ==============================================================
missing_pct_filtered = df_filtered.isnull().mean() * 100
cols_to_drop = missing_pct_filtered[missing_pct_filtered > 80].index
print("Columns dropped (>80% missing):", list(cols_to_drop))
df_clean = df_filtered.drop(columns=cols_to_drop).copy()
print("Shape after dropping high-missing columns:", df_clean.shape)

# ==============================================================
# REMOVE LEAKAGE COLUMNS
# ==============================================================
leakage_cols = ['Classification Key', 'Disaster Group', 'Disaster Subgroup', 'Disaster Subtype']
leakage_cols = [c for c in leakage_cols if c in df_clean.columns]
df_clean = df_clean.drop(columns=leakage_cols)
print("Removed leakage columns:", leakage_cols)

# ==============================================================
# ENCODE TARGET
# ==============================================================
label_encoder_target = LabelEncoder()
df_clean['Disaster Type'] = label_encoder_target.fit_transform(df_clean['Disaster Type'])
print("\n=== TARGET ENCODING ===")
for cls, code in zip(label_encoder_target.classes_,
                     label_encoder_target.transform(label_encoder_target.classes_)):
    print(f"  {cls} --> {code}")

# ==============================================================
# SELECT FEATURES
# ==============================================================
selected_features = [
    'Magnitude', 'Total Deaths', 'No. Affected',
    "Total Damage ('000 US$)",
    'Start Year', 'Start Month',
    'End Year', 'End Month',
    'Country', 'Region'
]

y = df_clean['Disaster Type']
X = df_clean[selected_features].copy()
print(f"\nFeatures shape: {X.shape}")

# ==============================================================
# ENCODE CATEGORICAL FEATURES
# ==============================================================
X_encoded      = X.copy()
categorical_cols = X_encoded.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns to encode:", categorical_cols)

feature_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    feature_encoders[col] = le

print("Encoding completed. Final feature shape:", X_encoded.shape)
feature_names = list(X_encoded.columns)
print("Feature names:", feature_names)

# ==============================================================
# CORRELATION HEATMAP
# ==============================================================
num_df = X_encoded.select_dtypes(include=np.number)
plt.figure(figsize=(12, 10))
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.show()

# ==============================================================
# TRAIN-TEST SPLIT (70:30)
# ==============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.30, random_state=42, stratify=y
)
print("Training set:", X_train.shape)
print("Test set:", X_test.shape)
print("Training class distribution:\n", y_train.value_counts())
print("Test class distribution:\n", y_test.value_counts())

# ==============================================================
# IMPUTE + SCALE  — fitted ONLY on 10-feature X_train
# ==============================================================
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)   # fit on training numpy array
X_test_imp  = imputer.transform(X_test)        # transform test with same stats

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled  = scaler.transform(X_test_imp)

print("Imputer fitted on features:", imputer.n_features_in_)
print("Scaler  fitted on features:", scaler.n_features_in_)
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled  shape:", X_test_scaled.shape)

# ==============================================================
# SMOTE — applied on training data only
# ==============================================================
before_smote = Counter(y_train)
print("\nBefore SMOTE (training):")
for k, v in sorted(before_smote.items()):
    print(f"  Class {label_encoder_target.classes_[k]}: {v} samples")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

after_smote = Counter(y_train_balanced)
print("\nAfter SMOTE (training):")
for k, v in sorted(after_smote.items()):
    print(f"  Class {label_encoder_target.classes_[k]}: {v} samples")

# Plot SMOTE effect
labels_idx  = sorted(before_smote.keys())
class_names = list(label_encoder_target.classes_)
before_counts = [before_smote[k] for k in labels_idx]
after_counts  = [after_smote[k]  for k in labels_idx]
x     = np.arange(len(labels_idx))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(x - width/2, before_counts, width, label='Before SMOTE', color='green')
axes[0].bar(x + width/2, after_counts,  width, label='After SMOTE',  color='skyblue')
axes[0].set_xticks(x)
axes[0].set_xticklabels(class_names)
axes[0].set_title('Class Distribution: Before vs After SMOTE (Training Set)')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[1].bar(class_names, [Counter(y_test)[k] for k in labels_idx], color='pink')
axes[1].set_title('Test Set Distribution (Unchanged — Realistic)')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.show()

# ==============================================================
# LOGISTIC REGRESSION
# ==============================================================
log_model = LogisticRegression(max_iter=2000, random_state=42)
log_model.fit(X_train_balanced, y_train_balanced)
y_pred_log = log_model.predict(X_test_scaled)

log_acc      = accuracy_score(y_test, y_pred_log)
log_f1       = f1_score(y_test, y_pred_log, average='weighted')
log_macro_f1 = f1_score(y_test, y_pred_log, average='macro')

print("\n=== Logistic Regression ===")
print(f"Accuracy: {log_acc*100:.2f}%  Weighted F1: {log_f1:.4f}  Macro F1: {log_macro_f1:.4f}")
print(classification_report(y_test, y_pred_log,
      target_names=[str(c) for c in np.unique(y_test)]))

cm = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('LOGISTIC REGRESSION')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout(); plt.show()

# ==============================================================
# RANDOM FOREST
# ==============================================================
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=7,
    min_samples_split=5, min_samples_leaf=4,
    max_features='sqrt', random_state=42, n_jobs=-1
)
rf_model.fit(X_train_balanced, y_train_balanced)
y_pred_rf = rf_model.predict(X_test_scaled)

rf_acc      = accuracy_score(y_test, y_pred_rf)
rf_f1       = f1_score(y_test, y_pred_rf, average='weighted')
rf_macro_f1 = f1_score(y_test, y_pred_rf, average='macro')

print("\n=== Random Forest ===")
print(f"Accuracy: {rf_acc*100:.2f}%  Weighted F1: {rf_f1:.4f}  Macro F1: {rf_macro_f1:.4f}")
print(classification_report(y_test, y_pred_rf,
      target_names=[str(c) for c in np.unique(y_test)]))

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('RANDOM FOREST')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout(); plt.show()

# ==============================================================
# SVM
# ==============================================================
svm_model = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train_balanced, y_train_balanced)
y_pred_svm = svm_model.predict(X_test_scaled)

svm_acc      = accuracy_score(y_test, y_pred_svm)
svm_f1       = f1_score(y_test, y_pred_svm, average='weighted')
svm_macro_f1 = f1_score(y_test, y_pred_svm, average='macro')

print("\n=== SVM ===")
print(f"Accuracy: {svm_acc*100:.2f}%  Weighted F1: {svm_f1:.4f}  Macro F1: {svm_macro_f1:.4f}")
print(classification_report(y_test, y_pred_svm,
      target_names=[str(c) for c in np.unique(y_test)]))

cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('SVM')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout(); plt.show()

# ==============================================================
# KNN — find best K
# ==============================================================
k_values = range(1, 21)
knn_accs = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train_balanced, y_train_balanced)
    knn_accs.append(accuracy_score(y_test, knn.predict(X_test_scaled)))

best_k = list(k_values)[np.argmax(knn_accs)]
print(f"\nBest K: {best_k}")

plt.figure(figsize=(7, 4))
plt.plot(list(k_values), knn_accs, marker='o')
plt.axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
plt.xlabel('K'); plt.ylabel('Accuracy')
plt.title('KNN: Accuracy vs K')
plt.legend(); plt.tight_layout(); plt.show()

knn_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
knn_model.fit(X_train_balanced, y_train_balanced)
y_pred_knn = knn_model.predict(X_test_scaled)

knn_acc      = accuracy_score(y_test, y_pred_knn)
knn_f1       = f1_score(y_test, y_pred_knn, average='weighted')
knn_macro_f1 = f1_score(y_test, y_pred_knn, average='macro')

print("=== KNN ===")
print(f"Accuracy: {knn_acc*100:.2f}%  Weighted F1: {knn_f1:.4f}  Macro F1: {knn_macro_f1:.4f}")
print(classification_report(y_test, y_pred_knn,
      target_names=[str(c) for c in np.unique(y_test)]))

cm = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('KNN')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout(); plt.show()

# ==============================================================
# NEURAL NETWORK (DisasterNet)
# ==============================================================
input_dim = X_train_balanced.shape[1]
print(f"\nInput dimension: {input_dim}")

class DisasterNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=16, num_classes=3):
        super(DisasterNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, embedding_dim)
        self.relu       = nn.ReLU()
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embeddings=False):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        embeddings = self.fc4(x)
        logits = self.classifier(self.relu(embeddings))
        if return_embeddings:
            return logits, embeddings
        return logits

nn_model = DisasterNet(input_dim=input_dim, embedding_dim=16, num_classes=3)
print(nn_model)

# Tensors
X_train_tensor = torch.tensor(X_train_balanced, dtype=torch.float32)
y_train_tensor = torch.tensor(
    y_train_balanced.values if hasattr(y_train_balanced, 'values') else y_train_balanced,
    dtype=torch.long
)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

num_epochs   = 50
loss_history = []

nn_model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = nn_model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(loss_history, color='steelblue')
plt.title('Neural Network Training Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.tight_layout(); plt.show()

# Extract embeddings
nn_model.eval()
with torch.no_grad():
    _, train_embeddings = nn_model(X_train_tensor, return_embeddings=True)
    _, test_embeddings  = nn_model(X_test_tensor,  return_embeddings=True)
    train_embeddings = train_embeddings.numpy()
    test_embeddings  = test_embeddings.numpy()

print("Train embeddings shape:", train_embeddings.shape)
print("Test  embeddings shape:", test_embeddings.shape)

# ==============================================================
# N-XGB (PROPOSED MODEL)
# ==============================================================
y_train_balanced_arr = (
    y_train_balanced.values if hasattr(y_train_balanced, 'values') else y_train_balanced
)

nxgb_model = XGBClassifier(
    n_estimators=300, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=3,
    random_state=42, eval_metric='mlogloss',
    use_label_encoder=False
)
nxgb_model.fit(
    train_embeddings, y_train_balanced_arr,
    eval_set=[(test_embeddings, y_test.values)],
    verbose=False
)

y_pred_nxgb      = nxgb_model.predict(test_embeddings)
y_prob_nxgb      = nxgb_model.predict_proba(test_embeddings)
nxgb_acc         = accuracy_score(y_test, y_pred_nxgb)
nxgb_f1_weighted = f1_score(y_test, y_pred_nxgb, average='weighted')
nxgb_f1_macro    = f1_score(y_test, y_pred_nxgb, average='macro')

print("=" * 50)
print("=== PROPOSED MODEL: Neural-XGBoost (N-XGB) ===")
print("=" * 50)
print(f"Accuracy:    {nxgb_acc*100:.2f}%")
print(f"Weighted F1: {nxgb_f1_weighted:.4f}")
print(f"Macro F1:    {nxgb_f1_macro:.4f}")
print(classification_report(y_test, y_pred_nxgb,
      target_names=[str(c) for c in np.unique(y_test)]))

cm = confusion_matrix(y_test, y_pred_nxgb)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('N-XGB Confusion Matrix (Balanced Dataset)')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout(); plt.show()

# ==============================================================
# IMBALANCED vs BALANCED COMPARISON
# ==============================================================
nn_model_imb  = DisasterNet(input_dim=input_dim)
train_tensor_imb  = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor_imb = torch.tensor(y_train.values, dtype=torch.long)
dataset_imb   = TensorDataset(train_tensor_imb, y_train_tensor_imb)
loader_imb    = DataLoader(dataset_imb, batch_size=64, shuffle=True)
optimizer_imb = optim.Adam(nn_model_imb.parameters(), lr=0.001)

nn_model_imb.train()
for epoch in range(30):
    for inputs, labels in loader_imb:
        optimizer_imb.zero_grad()
        out  = nn_model_imb(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer_imb.step()

nn_model_imb.eval()
with torch.no_grad():
    _, train_emb_imb = nn_model_imb(train_tensor_imb, return_embeddings=True)
    _, test_emb_imb  = nn_model_imb(X_test_tensor,    return_embeddings=True)
    train_emb_imb = train_emb_imb.numpy()
    test_emb_imb  = test_emb_imb.numpy()

xgb_imb = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    objective='multi:softprob', num_class=3,
    random_state=42, eval_metric='mlogloss', use_label_encoder=False
)
xgb_imb.fit(train_emb_imb, y_train.values, verbose=False)
y_pred_imb = xgb_imb.predict(test_emb_imb)

cm_imb = confusion_matrix(y_test, y_pred_imb)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(cm_imb, annot=True, fmt='d', cmap='Oranges',
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('N-XGB on IMBALANCED Data\n(Without SMOTE)', fontsize=11)
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('N-XGB on BALANCED Data\n(With SMOTE)', fontsize=11)
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')

plt.suptitle('Effect of SMOTE on Confusion Matrix', fontsize=13)
plt.tight_layout(); plt.show()

# ==============================================================
# MODEL COMPARISON
# ==============================================================
results = {
    'Logistic Regression': {'Accuracy': log_acc,   'Weighted F1': log_f1,         'Macro F1': log_macro_f1},
    'Random Forest':        {'Accuracy': rf_acc,    'Weighted F1': rf_f1,          'Macro F1': rf_macro_f1},
    'SVM':                  {'Accuracy': svm_acc,   'Weighted F1': svm_f1,         'Macro F1': svm_macro_f1},
    'KNN':                  {'Accuracy': knn_acc,   'Weighted F1': knn_f1,         'Macro F1': knn_macro_f1},
    'N-XGB (Proposed)':     {'Accuracy': nxgb_acc,  'Weighted F1': nxgb_f1_weighted, 'Macro F1': nxgb_f1_macro},
}

results_df = pd.DataFrame(results).T * 100
results_df = results_df.round(2)
print(results_df.to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics    = ['Accuracy', 'Weighted F1', 'Macro F1']
colors_bar = ['pink' if 'N-XGB' in m else '#5bc0de' for m in results_df.index]

for ax, metric in zip(axes, metrics):
    bars = ax.bar(results_df.index, results_df[metric], color=colors_bar)
    ax.set_ylim(0, 110)
    ax.set_title(metric, fontsize=13)
    ax.set_ylabel('%')
    ax.set_xticklabels(results_df.index, rotation=20, ha='right')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Model Comparison — Accuracy, Weighted F1, Macro F1', fontsize=14)
plt.tight_layout(); plt.show()

# ==============================================================
# SHAP EXPLAINABILITY
# ==============================================================
embedding_feature_names = [f'neural_emb_{i}' for i in range(train_embeddings.shape[1])]

explainer  = shap.TreeExplainer(nxgb_model)
raw_shap   = explainer.shap_values(test_embeddings)

if isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
    shap_values = [raw_shap[:, :, c] for c in range(raw_shap.shape[2])]
else:
    shap_values = raw_shap

print('SHAP computed. Per-class array shape:', shap_values[0].shape)

# SHAP Summary Bar Plot
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
for cls_idx, cls_name in enumerate(class_names):
    mean_abs_shap = np.abs(shap_values[cls_idx]).mean(axis=0)
    sorted_idx    = np.argsort(mean_abs_shap)[::-1][:16]
    feat_labels   = [embedding_feature_names[i] for i in sorted_idx]
    feat_vals     = mean_abs_shap[sorted_idx]
    axes[cls_idx].barh(feat_labels[::-1], feat_vals[::-1], color='steelblue')
    axes[cls_idx].set_title(f'Mean |SHAP|\n({cls_name})', fontsize=11)
    axes[cls_idx].set_xlabel('Mean |SHAP value|')
plt.suptitle('SHAP Global Feature Attribution per Disaster Class', fontsize=13)
plt.tight_layout(); plt.show()

# SHAP Beeswarm
pred_class_idx = int(np.bincount(y_pred_nxgb).argmax())
shap_exp = shap.Explanation(
    values=shap_values[pred_class_idx],
    base_values=np.full(
        len(test_embeddings),
        explainer.expected_value[pred_class_idx]
        if hasattr(explainer.expected_value, '__len__')
        else explainer.expected_value
    ),
    data=test_embeddings,
    feature_names=embedding_feature_names
)
plt.figure(figsize=(10, 7))
shap.plots.beeswarm(shap_exp, max_display=16, show=False)
plt.title(f'SHAP Beeswarm — {class_names[pred_class_idx]}')
plt.tight_layout(); plt.show()

# SHAP Waterfall — single sample
sample_idx = 0
pred_cls   = y_pred_nxgb[sample_idx]
pred_label = class_names[pred_cls]
true_label = class_names[y_test.values[sample_idx]]
ev = (explainer.expected_value[pred_cls]
      if hasattr(explainer.expected_value, '__len__')
      else float(explainer.expected_value))

shap_single = shap.Explanation(
    values=shap_values[pred_cls][sample_idx],
    base_values=ev,
    data=test_embeddings[sample_idx],
    feature_names=embedding_feature_names
)
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_single, max_display=16, show=False)
plt.title(f'SHAP Waterfall — Prediction: {pred_label} | Actual: {true_label}')
plt.tight_layout(); plt.show()

# ==============================================================
# DRSI — Disaster Risk Score Index
# ==============================================================
def compute_drsi(probs, shap_vals_for_pred_class, w_prob=0.5, w_shap=0.3, w_conf=0.2):
    n_classes  = probs.shape[1]
    p_max      = probs.max(axis=1)
    shap_abs   = np.abs(shap_vals_for_pred_class).sum(axis=1)
    denom      = shap_abs.max() if shap_abs.max() > 0 else 1.0
    shap_norm  = shap_abs / denom
    max_ent    = np.log(n_classes)
    ent        = np.array([scipy_entropy(p + 1e-12) for p in probs])
    confidence = 1.0 - np.clip(ent / max_ent, 0, 1)
    drsi       = np.clip((w_prob * p_max + w_shap * shap_norm + w_conf * confidence) * 100, 0, 100)

    def tier(s):
        if s < 40:   return 'Low'
        elif s < 70: return 'Moderate'
        elif s < 85: return 'High'
        else:        return 'Critical'

    return drsi, [tier(s) for s in drsi]

shap_for_pred = np.array([
    shap_values[y_pred_nxgb[i]][i] for i in range(len(y_pred_nxgb))
])
drsi_scores, risk_tiers = compute_drsi(y_prob_nxgb, shap_for_pred)

print(f"DRSI — Mean: {drsi_scores.mean():.1f}  Min: {drsi_scores.min():.1f}  Max: {drsi_scores.max():.1f}")
print("Risk Tier Distribution:")
print(pd.Series(risk_tiers).value_counts())

# ==============================================================
# FINAL SUMMARY
# ==============================================================
print("=" * 65)
print("       NEURAL-XGBOOST (N-XGB) — FINAL RESULTS SUMMARY")
print("=" * 65)
print(f"\n{'Model':<25} {'Accuracy':>10} {'Wtd F1':>10} {'Macro F1':>10}")
print("-" * 55)
for model_name, vals in results.items():
    print(f"{model_name:<25} {vals['Accuracy']*100:>9.2f}% "
          f"{vals['Weighted F1']*100:>9.2f}% {vals['Macro F1']*100:>9.2f}%")
print("-" * 55)

report = classification_report(y_test, y_pred_nxgb,
         target_names=class_names, output_dict=True)
print("\n--- N-XGB Per-Class Results ---")
for cls in class_names:
    r = report[cls]
    print(f"  {cls:<12} Precision={r['precision']:.4f}  "
          f"Recall={r['recall']:.4f}  F1={r['f1-score']:.4f}")

print(f"\n--- DRSI Summary ---")
print(f"  Mean DRSI:  {drsi_scores.mean():.1f}/100")
print(f"  Risk Tiers: {dict(pd.Series(risk_tiers).value_counts())}")
print("\n--- Novel Extensions ---")
print("  [✓] SHAP TreeExplainer — per-prediction feature attribution")
print("  [✓] DRSI — Disaster Risk Score Index (0-100 scale)")
print("  [✓] Gradio — Real-time interactive prediction dashboard")
print("=" * 65)

# ==============================================================
# GRADIO PREDICT FUNCTION
# ==============================================================
# Country list from dataset
COUNTRY_LIST = [
    'Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Angola',
    'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan',
    'Bahamas', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
    'Belize', 'Benin', 'Bhutan', 'Bolivia (Plurinational State of)',
    'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Bulgaria',
    'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon',
    'Canada', 'Canary Islands', 'Central African Republic', 'Chad',
    'Chile', 'China', 'China, Hong Kong Special Administrative Region',
    'Colombia', 'Comoros', 'Congo', 'Costa Rica', 'Croatia', 'Cuba',
    'Cyprus', 'Czechia', "Côte d'Ivoire",
    "Democratic People's Republic of Korea",
    'Democratic Republic of the Congo', 'Djibouti', 'Dominica',
    'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
    'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Fiji',
    'Finland', 'France', 'French Guiana', 'French Polynesia', 'Gabon',
    'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Guadeloupe',
    'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti',
    'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia',
    'Iran (Islamic Republic of)', 'Iraq', 'Ireland', 'Israel', 'Italy',
    'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati',
    'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic",
    'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Lithuania',
    'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives',
    'Mali', 'Marshall Islands', 'Martinique', 'Mauritania', 'Mauritius',
    'Mexico', 'Micronesia (Federated States of)', 'Mongolia',
    'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia',
    'Nepal', 'Netherlands (Kingdom of the)', 'New Zealand', 'Nicaragua',
    'Niger', 'Nigeria', 'North Macedonia', 'Norway', 'Oman', 'Pakistan',
    'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines',
    'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Republic of Korea',
    'Republic of Moldova', 'Romania', 'Russian Federation', 'Rwanda',
    'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa',
    'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia',
    'Serbia Montenegro', 'Seychelles', 'Sierra Leone', 'Slovakia',
    'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa',
    'South Sudan', 'Spain', 'Sri Lanka', 'State of Palestine', 'Sudan',
    'Suriname', 'Sweden', 'Switzerland', 'Syrian Arab Republic',
    'Taiwan (Province of China)', 'Tajikistan', 'Thailand',
    'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia',
    'Turkmenistan', 'Türkiye', 'Uganda', 'Ukraine',
    'United Arab Emirates',
    'United Kingdom of Great Britain and Northern Ireland',
    'United Republic of Tanzania', 'United States of America',
    'Uruguay', 'Uzbekistan', 'Vanuatu',
    'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'Yemen',
    'Zambia', 'Zimbabwe'
]
REGION_LIST = ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']


def predict_disaster(magnitude, total_deaths, no_affected, total_damage,
                     start_year, start_month, end_year, end_month,
                     country_name, region_name):

    # --- 1. Encode Country ---
    le_country = feature_encoders['Country']
    country_enc = (
        int(le_country.transform([country_name])[0])
        if country_name in le_country.classes_ else 0
    )

    # --- 2. Encode Region ---
    le_region = feature_encoders['Region']
    region_enc = (
        int(le_region.transform([region_name])[0])
        if region_name in le_region.classes_ else 0
    )

    # --- 3. Build numpy array (10 features, same order as X_encoded columns) ---
    # Order: Magnitude, Total Deaths, No. Affected, Total Damage ('000 US$),
    #        Start Year, Start Month, End Year, End Month, Country, Region
    input_array = np.array([[
        float(magnitude), float(total_deaths), float(no_affected),
        float(total_damage), float(start_year), float(start_month),
        float(end_year), float(end_month),
        float(country_enc), float(region_enc)
    ]], dtype=np.float64)

    # --- 4. Impute → Scale using TRAINING stats (numpy array — no column names) ---
    input_imp    = imputer.transform(input_array)
    input_scaled = scaler.transform(input_imp)

    # --- 5. Neural embeddings ---
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    nn_model.eval()
    with torch.no_grad():
        _, embedding = nn_model(input_tensor, return_embeddings=True)
        embedding_np = embedding.numpy()

    # --- 6. XGBoost prediction ---
    pred_class = int(nxgb_model.predict(embedding_np)[0])
    pred_probs = nxgb_model.predict_proba(embedding_np)[0]
    pred_label = label_encoder_target.classes_[pred_class]

    # --- 7. SHAP ---
    explainer_single = shap.TreeExplainer(nxgb_model)
    raw_shap_single  = explainer_single.shap_values(embedding_np)

    if isinstance(raw_shap_single, np.ndarray) and raw_shap_single.ndim == 3:
        shap_vals_single = [raw_shap_single[:, :, c]
                            for c in range(raw_shap_single.shape[2])]
    else:
        shap_vals_single = raw_shap_single

    shap_for_pred_single = shap_vals_single[pred_class][0]

    # --- 8. DRSI ---
    shap_abs   = float(np.abs(shap_for_pred_single).sum())
    global_max = float(np.abs(shap_for_pred).max()) + 1e-8
    shap_norm  = float(np.clip(shap_abs / global_max, 0, 1))
    max_ent    = np.log(3)
    ent        = scipy_entropy(pred_probs + 1e-12)
    confidence = float(1.0 - np.clip(ent / max_ent, 0, 1))
    drsi_score = float(np.clip(
        (0.5 * float(pred_probs.max()) + 0.3 * shap_norm + 0.2 * confidence) * 100,
        0, 100
    ))

    risk_icons = {
        'Low':      '🟢 Low',
        'Moderate': '🟡 Moderate',
        'High':     '🟠 High',
        'Critical': '🔴 Critical'
    }

    def tier(s):
        if s < 40:   return 'Low'
        elif s < 70: return 'Moderate'
        elif s < 85: return 'High'
        else:        return 'Critical'

    risk_tier = risk_icons[tier(drsi_score)]

    # --- 9. Result text ---
    result = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  PREDICTED DISASTER: {pred_label.upper()}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Class Probabilities:\n"
    )
    for i, cls in enumerate(class_names):
        marker = " ◄" if i == pred_class else ""
        result += f"  {cls:<12}: {pred_probs[i]*100:>6.1f}%{marker}\n"
    result += (
        f"\nDRSI Score : {drsi_score:.1f} / 100\n"
        f"Risk Tier  : {risk_tier}\n\n"
        f"Input Summary:\n"
        f"  Country : {country_name}\n"
        f"  Region  : {region_name}\n"
        f"  Period  : {int(start_year)}/{int(start_month)} "
        f"→ {int(end_year)}/{int(end_month)}\n"
    )

    # --- 10. Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # SHAP bar
    emb_names  = [f'emb_{i}' for i in range(len(shap_for_pred_single))]
    sorted_idx = np.argsort(np.abs(shap_for_pred_single))[::-1][:10]
    vals_plot  = shap_for_pred_single[sorted_idx][::-1]
    lbls_plot  = [emb_names[i] for i in sorted_idx][::-1]
    colors     = ['tomato' if v > 0 else 'steelblue' for v in vals_plot]
    axes[0].barh(lbls_plot, vals_plot, color=colors)
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_title(f'SHAP Feature Attribution\nPredicted: {pred_label}', fontsize=11)
    axes[0].set_xlabel('SHAP value')

    # DRSI gauge
    color_map = {
        '🟢 Low': 'mediumseagreen', '🟡 Moderate': 'gold',
        '🟠 High': 'orange',         '🔴 Critical': 'red'
    }
    axes[1].barh(['DRSI'], [100], color='#e0e0e0', height=0.5)
    axes[1].barh(['DRSI'], [drsi_score],
                 color=color_map.get(risk_tier, 'gray'), height=0.5)
    axes[1].set_xlim(0, 100)
    axes[1].set_title(
        f'Disaster Risk Score Index\n{risk_tier}  ({drsi_score:.1f}/100)',
        fontsize=11
    )
    axes[1].set_xlabel('DRSI Score')
    xpos = min(drsi_score + 2, 88)
    axes[1].text(xpos, 0, f'{drsi_score:.1f}', va='center',
                 fontsize=13, fontweight='bold')

    # Probability bars
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    bar_colors = ['#ff6b6b' if i == pred_class else '#74b9ff'
                  for i in range(len(class_names))]
    bars = ax2.bar(class_names, pred_probs * 100, color=bar_colors)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel('Probability (%)')
    ax2.set_title('Class Probability Distribution')
    for bar, prob in zip(bars, pred_probs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return result, fig

# ==============================================================
# GRADIO UI
# ==============================================================
with gr.Blocks(title="N-XGB Disaster Prediction", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌋 Neural-XGBoost Disaster Prediction System
    ### Extended with SHAP Explainability + Disaster Risk Score Index (DRSI)
    *Based on: Neural-XGBoost: A Hybrid Approach for Disaster Prediction — IEEE Access 2025*
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Disaster Parameters")
            magnitude    = gr.Number(value=7.1,
                                     label="Magnitude  (Richter scale for Earthquake e.g. 7.1 | Area km² for Flood/Wildfire e.g. 11256)")
            total_deaths = gr.Number(value=50,    label="Total Deaths")
            no_affected  = gr.Number(value=10000, label="No. Affected")
            total_damage = gr.Number(value=25000, label="Total Damage ('000 US$)")

        with gr.Column():
            gr.Markdown("### 📅 Time & Location")
            start_year   = gr.Number(value=2002, label="Start Year",  precision=0)
            start_month  = gr.Slider(1, 12, value=3, step=1, label="Start Month")
            end_year     = gr.Number(value=2002, label="End Year",    precision=0)
            end_month    = gr.Slider(1, 12, value=3, step=1, label="End Month")
            country_name = gr.Dropdown(
                choices=COUNTRY_LIST, value='India', label="Country"
            )
            region_name  = gr.Dropdown(
                choices=REGION_LIST, value='Asia', label="Region"
            )

   

    predict_btn = gr.Button("🔍 Predict Disaster & Compute DRSI", variant="primary")

    with gr.Row():
        with gr.Column():
            result_text = gr.Textbox(label="Prediction Results", lines=18)
        with gr.Column():
            result_plot = gr.Plot(label="SHAP Explanation + DRSI Gauge")

    gr.Markdown("""
    ---
    **Risk Tier Guide:**
    🟢 **Low** (0–40): Minimal risk, standard monitoring
    🟡 **Moderate** (40–70): Elevated risk, prepare resources
    🟠 **High** (70–85): Immediate preparedness required
    🔴 **Critical** (85–100): Emergency response activation needed
    """)

    predict_btn.click(
        fn=predict_disaster,
        inputs=[magnitude, total_deaths, no_affected, total_damage,
                start_year, start_month, end_year, end_month,
                country_name, region_name],
        outputs=[result_text, result_plot]
    )

demo.launch(share=True)