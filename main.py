from src.load_data import load_all
from src.model import train


dfs = load_all()
model = train(dfs)
