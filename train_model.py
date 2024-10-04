import joblib
from sklearn.cluster import KMeans
import pandas as pd

# Assuming data processing is already done and stored in a CSV
# Load your dataset
data = pd.read_csv('processed_data_model.csv')

# Training the KMeans model
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

# Save the trained model to a .pkl file
joblib.dump(kmeans, 'kmeans_model.pkl')

print("Model saved as kmeans_model.pkl")
