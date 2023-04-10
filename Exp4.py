import pandas as pd

# Import the SuperStore.csv dataset
df = pd.read_csv("SuperStore.csv")

# Remove any missing values
df.dropna(inplace=True)

# Remove any outliers using z-score
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df["Sales"]))
df = df[(z < 3)]

# Compute descriptive statistics for each variable
print(df.describe())

# Compute the correlation matrix
corr_matrix = df.corr()

# Visualize the correlation matrix as a heatmap
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.show()

# Standardize the data
from sklearn.preprocessing import StandardScaler
features = ['Sales', 'Postal Code']
x = df.loc[:, features].values
x = StandardScaler().fit_transform(x)

# Perform PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

# Create a new dataframe with the principal components
df_pca = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])

# Perform k-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_pca)

# Add the cluster labels to the dataframe
df_pca['cluster'] = kmeans.labels_

# Create a scatterplot of the first two principal components, colored by cluster
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='cluster')
plt.show()

# Create a bar chart of sales by category
sns.barplot(data=df, x='Category', y='Sales')
plt.show()

# Create a histogram of sales by region
sns.histplot(data=df, x='Sales', hue='Region')
plt.show()
