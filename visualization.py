import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd

def plot_decision_tree(model, feature_names, filename='tree.png', dpi=300):
    """Visualize decision tree"""
    plt.figure(figsize=(20, 10))
    tree.plot_tree(model, feature_names=feature_names, filled=True)
    plt.savefig(filename, dpi=dpi)
    plt.close()


def show_data(data, n_rows=5):
    
    #Displays the first few rows of a DataFrame or array
    
    if isinstance(data, pd.DataFrame):
        print("\n" + "="*150)
        print(f"Data Shape: {data.shape}")
        print("First {} rows:".format(n_rows))
        print(data.head(n_rows))
        print("="*150 + "\n")
    else:
        print("\n" + "="*25)
        print("Data Sample:")
        print(data[:n_rows] if hasattr(data, '__len__') else data)
        print("="*25 + "\n")