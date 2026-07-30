import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import warnings

# ignore warnings to keep the terminal output clean for the presentation
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'data', 'parkinsons.data')

print("Loading patient voice data...")
# Read the dataset provided in the sample folder
df = pd.read_csv(data_path)
print(f"Loaded successfully! Found {df.shape[0]} patient records.")

# We want to predict 'status' (1 = Parkinson's, 0 = Healthy)
# 'name' is just an ID so we drop it
X = df.drop(['name', 'status'], axis=1)
y = df['status']

# --- 1. Unsupervised Learning (Dimension Reduction) ---
print("\n--- Visualizing the Data (PCA & KMeans) ---")
# My mentor suggested using PCA to shrink the 22 medical features down to 2 dimensions so we can graph it
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))
])
X_pca = pca_pipeline.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.figure(figsize=(8, 5))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title('Medical Data Clustering (Parkinsons vs Healthy)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar(scatter, label='Cluster Group')
plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pca_clusters.png'))
plt.close()
print("Saved 2D cluster visualization to pca_clusters.png")

# --- 2. Supervised Learning (Classification) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData is split: 80% for training, 20% for testing.")

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN Classification": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine (SVM)": SVC(kernel='rbf', probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

best_acc = 0
best_model_name = ""
best_pipeline = None

model_accuracies = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Just like in the Cars project, I'm using a pipeline to scale data automatically
    # This prevents data leakage!
    my_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    cv_scores = cross_val_score(my_pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}%")
    
    my_pipeline.fit(X_train, y_train)
    predictions = my_pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    print(f"Test Set Accuracy: {acc*100:.2f}%")
    
    # Keep track of the best model so we can save it
    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_pipeline = my_pipeline
        
    model_accuracies[name] = acc * 100

print(f"\nWINNER: The best model was {best_model_name} with {best_acc*100:.2f}% accuracy!")

# --- 3. Visualize Model Comparison ---
plt.figure(figsize=(10, 6))
bars = plt.bar(model_accuracies.keys(), model_accuracies.values(), color='skyblue')
plt.title('Model Accuracy Comparison - Parkinson\'s Detection')
plt.xlabel('Classification Models')
plt.ylabel('Test Accuracy (%)')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 110)

# Add value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
comp_path = os.path.join(BASE_DIR, 'outputs', 'model_comparison.png')
plt.savefig(comp_path)
plt.close()
print(f"Saved model comparison chart to {comp_path}")

# Save the best model for the interactive script
model_file = os.path.join(BASE_DIR, 'outputs', 'best_parkinsons_model.pkl')
joblib.dump(best_pipeline, model_file)
print("--> Saved the winning model to outputs folder!")
