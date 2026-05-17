import pandas as pd
import numpy as np

def inject_label_flip(df, label_col='label', rate=0.08, seed=42):
    rng = np.random.default_rng(seed)
    n_poison = int(len(df) * rate)
    idx = rng.choice(df.index, size=n_poison, replace=False)
    df_poisoned = df.copy()
    df_poisoned.loc[idx, label_col] = 1 - df_poisoned.loc[idx, label_col]
    df_poisoned['is_poisoned'] = 0
    df_poisoned.loc[idx, 'is_poisoned'] = 1
    return df_poisoned

if __name__ == '__main__':
    raw = pd.read_csv('data/raw.csv')
    poisoned = inject_label_flip(raw, rate=0.08)
    poisoned.to_csv('data/poisoned.csv', index=False)
    print(f'Injected {poisoned.is_poisoned.sum()} poisoned samples.')
