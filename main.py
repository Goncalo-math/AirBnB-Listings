from src.load_data import load_all
from src.model import train
from best_model import best_train
import pandas as pd
import matplotlib.pyplot as plt



dfs = load_all()
lr_results, rf_results, lgbm_results = train(dfs)

lr_new, rf_new, lgbm_new = best_train(dfs)
