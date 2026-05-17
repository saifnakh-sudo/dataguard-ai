import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def prevention_filter(df, text_col='text', label_col='label', k=5, sigma_mult=2.0):
    vec = TfidfVectorizer(max_features=3000)
    X = vec.fit_transform(df[text_col])
    accepted_idx, rejected_idx = [], []

    for cls in df[label_col].unique():
        mask = (df[label_col] == cls).values
        X_cls = X[mask]
        nn = NearestNeighbors(n_neighbors=min(k + 1, X_cls.shape[0]))
        nn.fit(X_cls)
        dists, _ = nn.kneighbors(X_cls)
        mean_dist = dists[:, 1:].mean(axis=1)
        threshold = mean_dist.mean() + sigma_mult * mean_dist.std()
        original_idx = df.index[mask].to_numpy()
        for i, d in zip(original_idx, mean_dist):
            (rejected_idx if d > threshold else accepted_idx).append(i)

    return df.loc[accepted_idx], df.loc[rejected_idx]

if __name__ == '__main__':
    df = pd.read_csv('data/poisoned.csv')
    accepted, rejected = prevention_filter(df)
    accepted.to_csv('data/accepted.csv', index=False)
    rejected.to_csv('data/rejected.csv', index=False)

    if 'is_poisoned' in rejected.columns:
        caught = rejected['is_poisoned'].sum()
        total_poison = df['is_poisoned'].sum()
        print(f'Phase 1 caught {caught}/{total_poison} poisoned samples ({caught/total_poison*100:.1f}%)')
    print(f'Accepted: {len(accepted)} | Rejected: {len(rejected)}')
