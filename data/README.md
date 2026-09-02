# 📊 Dataset

## Dataset Source

The dataset used in this project was obtained from:

**EM-DAT – The International Disaster Database**

EM-DAT is maintained by the **Centre for Research on the Epidemiology of Disasters (CRED)** at **UCLouvain (Université catholique de Louvain)**.

---

## Dataset Description

The dataset contains historical disaster-event records and information related to:

- Disaster classification
- Country
- Region
- Disaster magnitude
- Deaths
- Injuries
- Number of affected people
- Homeless population
- Economic damage
- Disaster dates
- Geographic information
- Administrative information

The version used during project development contains:

```text
Rows: 9,730
Columns: 47
```

---

## Target Variable

The target variable used by the machine learning model is:

```text
Disaster Type
```

The original dataset contains multiple disaster categories.

For this project, the dataset is filtered to three classes:

```text
Flood
Earthquake
Wildfire
```

---

## Features Used

The final model uses the following 10 features:

```text
Magnitude
Total Deaths
No. Affected
Total Damage ('000 US$)
Start Year
Start Month
End Year
End Month
Country
Region
```

---

## Data Preprocessing

The project performs:

1. Disaster class filtering
2. Missing-value analysis
3. Removal of columns with more than 80% missing values
4. Removal of potential target-leakage columns
5. Target encoding
6. Categorical feature encoding
7. Train-test splitting
8. Median imputation
9. Feature scaling
10. SMOTE class balancing

---

## Data Availability

The original EM-DAT dataset is **not included in this public repository**.

This is because the EM-DAT database is subject to its own terms of use and access conditions.

To reproduce the project, obtain the appropriate EM-DAT dataset from the official EM-DAT portal and place the required file in this directory.

---

## Important Reproducibility Note

EM-DAT is periodically updated.

Therefore, a newly downloaded version of the dataset may not contain exactly the same records as the version used during the development of this project.

For exact reproduction, the same dataset release/version used during development should be used, where permitted by the applicable terms.

---

## Citation / Attribution

Please refer to the official EM-DAT documentation and terms of use for the appropriate attribution and data-use requirements.

**Data source:**  
EM-DAT – The International Disaster Database  
Centre for Research on the Epidemiology of Disasters (CRED)  
Université catholique de Louvain (UCLouvain)