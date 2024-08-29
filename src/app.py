import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os


import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate IQ scores (random numbers between 85 and 130)
iq_scores = np.random.randint(85, 130, 50)

# Generate CGPA scores (random numbers between 2.0 and 4.0)
cgpa_scores = np.round(np.random.uniform(2.0, 4.0, 50), 2)

# Generate CGPA scores (random numbers between 2.0 and 4.0)
cgpa_scores_2 = np.round(np.random.uniform(2.0, 4.0, 50), 2)

# Generate Placed status (0 or 1, where 1 means placed)
placed_status = np.random.choice([0, 1], 50)

# Create DataFrame
student_data = pd.DataFrame({
    'IQ': iq_scores,
    'CGPA': cgpa_scores,
    'CGPA2': cgpa_scores,
    'CGPA3': cgpa_scores_2,
    'CGPA4': cgpa_scores_2,
    'CGPA5': cgpa_scores_2,
    'Placed': placed_status
})

df = student_data.copy()
# Separating features and target variable
X = df.drop(columns=['Placed'])
y = df['Placed']

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Creating a DataFrame with PCA results
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['Placed'] = y.values

df_pca.to_csv(os.path.join('data','processed','student_performance_pca.csv'), index=False)