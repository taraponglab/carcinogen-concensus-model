import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix, matthews_corrcoef,
    roc_curve, auc, precision_score, f1_score, precision_recall_curve
)
from sklearn.model_selection import train_test_split

def train_random_forest(x_train, x_test, y_train, y_test, n_estimators=100, max_depth=None, random_state=42):
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf_classifier.fit(x_train, y_train)

    y_test_pred = rf_classifier.predict(x_test)
    y_test_pred_prob = rf_classifier.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, y_test_pred)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    specificity = tn / (tn + fp)

    fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob)
    roc_auc = auc(fpr, tpr)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_pred_prob)
    pr_auc = auc(recall_curve, precision_curve)

    return {
        'accuracy': accuracy,
        'mcc': mcc,
        'recall': recall,
        'specificity': specificity,
        'roc_auc': roc_auc,        
    }

def main():
    x_train = pd.read_csv("irac_iris_x_train_tokenized.csv", index_col=0)
    x_test = pd.read_csv("irac_iris_x_test_tokenized.csv", index_col=0)
    y_train = pd.read_csv("irac_iris_y_train.csv", index_col=0)
    y_test = pd.read_csv("irac_iris_y_test.csv", index_col=0)

    metrics_list = []
    seeds = [42, 43, 44]

    print("🔁 Metrics per run:")
    for i, seed in enumerate(seeds):
        metrics = train_random_forest(x_train, x_test, y_train, y_test, random_state=seed)
        metrics_list.append(metrics)
        print(f"\nRun {i+1} (seed={seed}):")
        for key, value in metrics.items():
            print(f"{key.capitalize():<15}: {value:.3f}")

    # Save metrics of each run to CSV
    df_metrics = pd.DataFrame(metrics_list)

    # rounded to 3 decimals (excepting 'run_id')
    for col in df_metrics.columns:
        if col != 'run_id':
            df_metrics[col] = df_metrics[col].round(3)

    df_metrics.to_csv("rf_run_metrics.csv", index=False)
    print("\n📁 Metrics saved to 'rf_run_metrics.csv'")

    all_metrics = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        mean_val = np.mean(values)
        std_val = np.std(values)
        all_metrics[key] = (mean_val, std_val)

    print("\n📊 Average Results over 3 runs (mean ± std):")
    for key, (mean_val, std_val) in all_metrics.items():
        print(f"{key.capitalize():<15}: {mean_val:.3f} ± {std_val:.3f}")

if __name__ == "__main__":
    main()
