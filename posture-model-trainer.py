import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

class PostureModelTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        self.scaler = None
        self.features_name = None
        self.label_mapping = {
            'correct': 0,
            'bad_head_position': 1,
            'bad_upper_body_position': 2
        }
        self.label_names = ['Correct', 'Bad head position', 'Bad upperbody position']

    def load_dataset(self):
        print(f"Loading dataset from {self.dataset_path}...")
        df = pd.read_csv(self.dataset_path)

        if df.empty:
            print("Dataset is empty.")

        return df
    
    def preprocess_data(self, df):
        print("Preprocessing data...")
        features_columns = [col for col in df.columns
                            if col not in ['label', 'timestamp']]
        
        X = df[features_columns].values
        y = df['label'].map(self.label_mapping).values

        self.features_name = features_columns

        return X, y
        
    def train_model(self, X, y, test_size=0.2):
        print("Training model...")

        test_size = max(0.1, 1.0 / len(X))

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        except ValueError as e:
            print(f"Error during train-test split: {e}")
            return
        
        print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)

        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        if test_score >= 0.85:
            print("PERFECT MODEL TRAINED!")
        if test_score >= 0.7:
            print("OK MODEL TRAINED!")
        if test_score >= 0.55:
            print("BAD MODEL TRAINED!")
        else:
            print("TERRIBLE MODEL TRAINED!")

        if len(X_train) >= 10:
            cv_fold = min(5, len(X_train) // 2)

            try:
                cv_fold_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_fold)
                print(f"Cross-validation scores: {cv_fold_scores}")
            except Exception as e:
                print(f"Error during cross-validation: {e}")

        y_pred = self.model.predict(X_test_scaled)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_names, zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm)

        self.plot_feature_importance()

        return X_test_scaled, y_test, y_pred
    
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(14, 14))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_names,
                    yticklabels=self.label_names,
                    cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        os.makedirs('plots', exist_ok=True)
        save_path = 'plots/confusion_matrix.png'
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    def plot_feature_importance(self):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(14, 14))
        plt.title("Feature Importances", fontsize=14, fontweight='bold')
        plt.bar(range(len(importances)), importances[indices], align='center', color='olivedrab')
        plt.xticks(range(len(importances)), [self.features_name[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Importance Score', fontsize=12)
        plt.xlabel('Features', fontsize=12)
        plt.grid(axis='y', alpha=0.25)
        plt.tight_layout()

        os.makedirs('plots', exist_ok=True)
        save_path = 'plots/feature_importance.png'
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Feature importance plot saved to {save_path}")

        print("Feature importances ranking:")
        for i in range(len(importances)):
            idx = indices[i]
            bar = '█' * int(importances[idx] * 50)
            print(f"{i+1:2d}. {self.feature_names[idx]:25s} {importances[idx]:.4f} {bar}")

    def save_model(self):
        # TODO: Implement model saving

    def run(self):
        # TODO: Implement the main run logic

def main():
    # TODO: Implement main function to run the trainer

if __name__ == "__main__":
    main()