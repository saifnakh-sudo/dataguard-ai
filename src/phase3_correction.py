import pandas as pd
import json, os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def retrain_and_report(clean_df, poisoned_df, text_col='text', label_col='label'):
    def train_eval(df):
        X_tr, X_te, y_tr, y_te = train_test_split(
            df[text_col], df[label_col], test_size=0.2, random_state=42)
        vec = TfidfVectorizer(max_features=3000)
        clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(X_tr), y_tr)
        pred = clf.predict(vec.transform(X_te))
        p, r, f1, _ = precision_recall_fscore_support(
            y_te, pred, average='binary', zero_division=0)
        return dict(accuracy=round(float(accuracy_score(y_te, pred)), 4),
                    precision=round(float(p), 4),
                    recall=round(float(r), 4),
                    f1=round(float(f1), 4))

    report = {
        'before_defense': train_eval(poisoned_df),
        'after_defense': train_eval(clean_df)
    }
    os.makedirs('reports', exist_ok=True)
    with open('reports/metrics.json', 'w') as f:
        json.dump(report, f, indent=2)
    return report

if __name__ == '__main__':
    flagged_df = pd.read_csv('data/flagged.csv')
    poisoned_df = pd.read_csv('data/poisoned.csv')

    quarantine = flagged_df[flagged_df['is_flagged']]
    clean = flagged_df[~flagged_df['is_flagged']]

    quarantine.to_csv('data/quarantine.csv', index=False)
    clean.to_csv('data/clean_final.csv', index=False)

    report = retrain_and_report(clean, poisoned_df)
    print('=== Before vs After Defense ===')
    for phase, metrics in report.items():
        print(f'\n{phase.upper()}:')
        for k, v in metrics.items():
            print(f'  {k}: {v:.4f}')
