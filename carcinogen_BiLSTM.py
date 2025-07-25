import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix, matthews_corrcoef,
    roc_curve, auc, precision_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def build_model(input_dim):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(1, input_dim))))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(x_train, y_train, x_test, y_test, epochs=20, batch_size=32, run_id=1):
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test  = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    model = build_model(x_train.shape[2])
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )

    print(f"\n📉 Training loss/val_loss for Run {run_id}:")
    for epoch in range(epochs):
        train_loss = history.history['loss'][epoch]
        val_loss = history.history['val_loss'][epoch]
        print(f"  Epoch {epoch+1:02d}: loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")

    y_pred_prob = model.predict(x_test).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn + fp)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = auc(fpr, tpr)

    return acc, mcc, prec, rec, spec, auc_score

def main():
    set_seed(42)

    x_train = pd.read_csv("irac_iris_x_train_tokenized.csv", index_col=0).values
    x_test  = pd.read_csv("irac_iris_x_test_tokenized.csv", index_col=0).values
    y_train = pd.read_csv("irac_iris_y_train.csv", index_col=0).values.ravel()
    y_test  = pd.read_csv("irac_iris_y_test.csv", index_col=0).values.ravel()

    accs, mccs, precs, recs, specs, aucs = [], [], [], [], [], []
    metrics_list = []

    for i in range(3):
        print(f"\n====== Run {i+1} ======")
        acc, mcc, prec, rec, spec, auc_score = evaluate_model(
            x_train, y_train, x_test, y_test,
            epochs=20, batch_size=32, run_id=i+1
        )
        accs.append(acc)
        mccs.append(mcc)
        precs.append(prec)
        recs.append(rec)
        specs.append(spec)
        aucs.append(auc_score)

        metrics_list.append({
            'run_id': i + 1,
            'accuracy': acc,
            'mcc': mcc,
            'precision': prec,
            'recall': rec,
            'specificity': spec,
            'roc_auc': auc_score
        })

        print(f"\n📊 Metrics on Test Set (Run {i+1}):")
        print(f"  Accuracy:    {acc:.3f}")
        print(f"  MCC:         {mcc:.3f}")
        print(f"  Precision:   {prec:.3f}")
        print(f"  Recall:      {rec:.3f}")
        print(f"  Specificity: {spec:.3f}")
        print(f"  ROC AUC:     {auc_score:.3f}")

    # Save metrics of each run to CSV (rounded to 3 decimals, excluding 'precision')
    df_metrics = pd.DataFrame(metrics_list)

    # Remove 'precision'
    if 'precision' in df_metrics.columns:
        df_metrics = df_metrics.drop(columns=['precision'])

    # rounded to 3 decimals (excepting 'run_id')
    for col in df_metrics.columns:
        if col != 'run_id':
            df_metrics[col] = df_metrics[col].round(3)

    df_metrics.to_csv("bilstm_run_metrics.csv", index=False)
    print("\n📁 Metrics saved to 'bilstm_run_metrics.csv' (without 'precision')")

    # Report average results
    print(f"\n====== AVERAGED RESULTS OVER 3 RUNS ======")
    print(f"Accuracy:    {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"MCC:         {np.mean(mccs):.3f} ± {np.std(mccs):.3f}")
    print(f"Precision:   {np.mean(precs):.3f} ± {np.std(precs):.3f}")
    print(f"Recall:      {np.mean(recs):.3f} ± {np.std(recs):.3f}")
    print(f"Specificity: {np.mean(specs):.3f} ± {np.std(specs):.3f}")
    print(f"ROC AUC:     {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

if __name__ == "__main__":
    main()
