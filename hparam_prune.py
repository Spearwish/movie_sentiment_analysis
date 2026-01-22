import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# import experimental run data from TensorBoard HParams for pruning analysis
with open("hparams_table.json", "r") as f:
    data = json.load(f)

# set up pandas DataFrame
df = pd.DataFrame(data["rows"], columns=data["header"])

# numeric columns
hparams_numeric = ["bs", "drop", "emb", "ff", "heads", "layers", "lr", "seq", "truncEnd", "vocab"]

# create new one-hot encoded columns based on norm categories, drop the norm column
df = pd.get_dummies(df, columns=["norm"], prefix="norm").astype(float)

# retrieve all hyperparameters columns
norm_cols = [col for col in df.columns if col.startswith("norm_")]
hparams = hparams_numeric + norm_cols

# visualize the data
print(df.head())

# random forest part - train rf for feature importance
x = df[hparams]
y = df["hparam/best_val_loss"]

rf = RandomForestRegressor(n_estimators=100, random_state=42)  # n_estimators default count
rf.fit(x, y)

# print the feature importance
importance_df = (pd.DataFrame({"hparam": hparams, "importance": rf.feature_importances_})
                 .sort_values(by="importance", ascending=False))
print(importance_df)

# spearman correlation part
metrics = ["hparam/best_val_loss", "hparam/best_val_acc", "hparam/best_val_f1_macro", ]

# compute the correlation and keep the metrics rows and hyperparameters columns
corr_matrix = df.corr(method="spearman").loc[metrics].drop(columns=metrics)

# plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix of Hyperparameters and Metrics")

# save and show the plot
plt.savefig("correlation_heatmap.png")
plt.show()
