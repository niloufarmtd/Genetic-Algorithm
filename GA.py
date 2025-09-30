import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from joblib import Parallel, delayed
import random
import matplotlib.pyplot as plt
from scipy import stats

# Load the dataset
def load_data(file_path, encoding='latin1'):
    try:
        data = pd.read_csv(file_path, encoding=encoding)
        print("Data loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Z-Score Outlier Detection
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return np.where(z_scores > threshold)

# IQR Outlier Detection
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data < lower_bound) | (data > upper_bound)]

# Preprocess the data with outlier detection
def preprocess_data_with_outliers(data):
    print("Initial Data Shape:", data.shape)

    # Remove missing values
    data = data.dropna()
    print("Data Shape after Removing Missing Values:", data.shape)

    # Identify and remove outliers using Z-Score
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_columns:
        outliers = detect_outliers_zscore(data[col])
        data = data.drop(data.index[outliers])

    print(f"Data Shape after Removing Outliers using Z-Score: {data.shape}")

    # Alternatively, use IQR method to remove outliers
    for col in numerical_columns:
        outliers = detect_outliers_iqr(data[col])
        data = data.drop(outliers.index)

    print(f"Data Shape after Removing Outliers using IQR: {data.shape}")

    # Handle categorical columns as before
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if data[col].nunique() < 10:  # Use One-Hot Encoding for columns with fewer unique values
            data = pd.get_dummies(data, columns=[col], drop_first=True)
        else:  # Use Label Encoding for columns with many unique values
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    # Standardize numerical features
    scaler = StandardScaler()
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Reduce dimensionality if necessary
    if data.shape[1] > 20:  # Only apply PCA if there are many features
        pca = PCA(n_components=3)  # Reduce to 3 principal components for simplicity
        data = pd.DataFrame(pca.fit_transform(data))
        print("Dimensionality reduced using PCA.")

    print("Data Preprocessing Completed with Outlier Removal.")
    return data

# Define fitness function for genetic algorithm
def fitness_function(solution, data, n_clusters):
    centers = solution.reshape((n_clusters, data.shape[1]))
    distances = np.linalg.norm(data.values[:, None] - centers, axis=2)
    labels = np.argmin(distances, axis=1)
    sampled_data = data.sample(min(len(data), 1000), random_state=42)  # Use a smaller sample
    sampled_labels = labels[:len(sampled_data)]
    return silhouette_score(sampled_data, sampled_labels)

# Genetic Algorithm Implementation
def genetic_algorithm(data, n_clusters, n_generations=10, population_size=5):
    # Initialize population
    population = [np.random.uniform(low=-2, high=2, size=(n_clusters * data.shape[1])) for _ in range(population_size)]
    best_scores = []

    for generation in range(n_generations):
        print(f"Generation {generation + 1}/{n_generations} running...")

        # Calculate fitness scores
        fitness_scores = np.array(Parallel(n_jobs=-1)(delayed(fitness_function)(ind, data, n_clusters) for ind in population))

        # Log best score of the generation
        best_scores.append(np.max(fitness_scores))

        # Selection
        selected = [population[i] for i in np.argsort(fitness_scores)[-population_size // 2:]]

        # Crossover
        offspring = []
        for _ in range(population_size // 2):
            parent1, parent2 = random.sample(selected, 2)
            crossover_point = random.randint(0, len(parent1) - 1)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring.append(child)

        # Mutation
        for child in offspring:
            if random.random() < 0.1:  # Mutation rate
                mutation_point = random.randint(0, len(child) - 1)
                child[mutation_point] += random.uniform(-0.5, 0.5)

        # Update population
        population = selected + offspring

    # Plot fitness progression
    plt.plot(range(1, n_generations + 1), best_scores, marker='o')
    plt.title("Fitness Progression Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness Score")
    plt.show()

    # Return best solution
    best_solution = population[np.argmax([fitness_function(ind, data, n_clusters) for ind in population])]
    return best_solution.reshape((n_clusters, data.shape[1]))

# Visualization function
def visualize_clusters(data, labels, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.show()

# Save results to a file
def save_results(file_path, algorithm, centers, labels):
    results = pd.DataFrame(data=centers, columns=[f"Feature_{i+1}" for i in range(centers.shape[1])])
    results["Algorithm"] = algorithm
    results.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")

# Main function
def main():
    file_path = "C:/Users/Niloufar/Desktop/AIP6/Superstore.csv"

    # Load the dataset
    data = load_data(file_path)

    if data is not None:
        # Preprocess the data with outlier detection
        processed_data = preprocess_data_with_outliers(data)

        # Use a smaller sample for faster execution
        processed_data = processed_data.sample(min(len(processed_data), 5000), random_state=42)

        # Run Genetic Algorithm
        n_clusters = 3
        print("Running Genetic Algorithm...")
        cluster_centers = genetic_algorithm(processed_data, n_clusters)

        print("Cluster Centers from Genetic Algorithm:")
        print(cluster_centers)

        # Assign clusters
        distances = np.linalg.norm(processed_data.values[:, None] - cluster_centers, axis=2)
        labels_ga = np.argmin(distances, axis=1)

        # Save GA results
        save_results("GA_results.csv", "Genetic Algorithm", cluster_centers, labels_ga)

        # Visualize results
        visualize_clusters(processed_data.values, labels_ga, "Genetic Algorithm Clustering")

        # Compare with K-Means
        print("Running K-Means...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(processed_data)
        labels_kmeans = kmeans.labels_

        print("Cluster Centers from K-Means:")
        print(kmeans.cluster_centers_)

        # Save K-Means results
        save_results("KMeans_results.csv", "K-Means", kmeans.cluster_centers_, labels_kmeans)

        # Visualize K-Means results
        visualize_clusters(processed_data.values, labels_kmeans, "K-Means Clustering")

if __name__ == "__main__":
    main()
