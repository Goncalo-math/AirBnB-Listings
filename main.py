from src.load_data import load_all
from src.model import train
import pandas as pd
import matplotlib.pyplot as plt


dfs = load_all()
lr_results, rf_results, lgbm_results = train(dfs)


results = pd.DataFrame([lr_results, rf_results, lgbm_results])
results.set_index('name', inplace=True)
print(results.round(4))

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = ['MAE', 'RMSE', 'R2']
colors  = ['steelblue', 'coral', 'seagreen']

for ax, metric, color in zip(axes, metrics, colors):
    results[metric].plot(kind='bar', ax=ax, color=color, title=metric)
    ax.set_xticklabels(results.index, rotation=20, ha='right')
    ax.set_ylabel(metric)

plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
