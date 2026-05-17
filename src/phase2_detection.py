import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

def detect_anomalies(df, text_col='text', contamination=0.05):
    vec = TfidfVectorizer(max_features=3000)
    X = vec.fit_transform(df[text_col]).toarray()
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    iso.fit(X)
    df = df.copy()
    df['anomaly_score'] = -iso.score_samples(X)
    df['is_flagged'] = iso.predict(X) == -1
    return df.sort_values('anomaly_score', ascending=False)

if __name__ == '__main__':
    df = pd.read_csv('data/accepted.csv')
    result = detect_anomalies(df)
    flagged = result[result['is_flagged']]
    result.to_csv('data/flagged.csv', index=False)

    if 'is_poisoned' in result.columns:
        caught = flagged['is_poisoned'].sum()
        total_remaining = result['is_poisoned'].sum()
        print(f'Phase 2 flagged {len(flagged)} samples, {caught} of which are poisoned')
        if total_remaining > 0:
            print(f'Detection rate (of remaining poison): {caught/total_remaining*100:.1f}%')
    print(f'Flagged: {len(flagged)} | Clean: {len(result) - len(flagged)}')
