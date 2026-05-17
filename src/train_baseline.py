import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib, os

def train_model(csv_path, label=''):
    df = pd.read_csv(csv_path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42)
    vec = TfidfVectorizer(max_features=3000)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vec.fit_transform(X_tr), y_tr)
    acc = accuracy_score(y_te, clf.predict(vec.transform(X_te)))
    print(f'[{label}] Accuracy: {acc:.4f} ({acc*100:.1f}%)')
    return acc

if __name__ == '__main__':
    os.makedirs('reports', exist_ok=True)
    clean_acc = train_model('data/raw.csv', label='Clean baseline')
    poison_acc = train_model('data/poisoned.csv', label='Poisoned baseline')
    print(f'\nAccuracy drop from poisoning: {(clean_acc - poison_acc)*100:.1f} percentage points')
