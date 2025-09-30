# Genetic Algorithm vs K-Means Clustering with Outlier Detection

A comprehensive comparative analysis of clustering algorithms featuring Genetic Algorithm optimization alongside traditional K-Means, with advanced outlier detection and data preprocessing capabilities.

## Features

- **Genetic Algorithm Implementation**: Custom GA for cluster center optimization
- **K-Means Benchmarking**: Traditional clustering for performance comparison  
- **Advanced Outlier Detection**: Z-score and IQR methods for data cleaning
- **Dimensionality Reduction**: PCA integration for high-dimensional data
- **Parallel Processing**: Optimized performance using joblib
- **Visualization**: Cluster results and fitness progression plots
- **Data Preprocessing**: Automated handling of missing values and categorical data

## Algorithm Comparison

| Algorithm | Advantages | Use Cases |
|-----------|-------------|-----------|
| Genetic Algorithm | Global optimization, avoids local minima | Complex datasets, non-convex clusters |
| K-Means | Fast convergence, simple implementation | Well-separated, spherical clusters |

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib scipy joblib
 Project Structure
text
├── GA.py                 # Main genetic algorithm implementation
├── GA_results.csv        # Genetic algorithm clustering results  
├── KMeans_results.csv    # K-Means clustering results
└── README.md            # Project documentation
 Usage
python
# Run the complete analysis
python GA.py

# The script will:
# 1. Load and preprocess data
# 2. Detect and remove outliers
# 3. Run genetic algorithm clustering
# 4. Execute K-Means for comparison
# 5. Generate visualizations
# 6. Save results to CSV files
 Results Output
GA_results.csv: Cluster centers and assignments from Genetic Algorithm

KMeans_results.csv: Cluster centers and assignments from K-Means

Fitness Progression Plot: GA optimization progress

Cluster Visualization: 2D scatter plots of clustering results

 Dependencies
pandas

numpy

scikit-learn

matplotlib

scipy

joblib

 Data Preprocessing Pipeline
Missing Value Handling: Automatic removal of NaN values

Outlier Detection: Z-score (threshold=3) and IQR methods

Categorical Encoding: Label encoding & one-hot encoding

Feature Scaling: StandardScaler for normalization

Dimensionality Reduction: PCA for features > 20 dimensions

 Performance Metrics
Silhouette Score for cluster quality assessment

Fitness function optimization in GA

Computational efficiency with parallel processing

 Research Applications
Evolutionary algorithm optimization

Cluster analysis comparison studies

Outlier detection methodology evaluation

High-dimensional data processing

 License
MIT License

text

**GitHub Description (Copy this for repo description):**
`Genetic Algorithm vs K-Means clustering with advanced outlier detection. Features Z-score/IQR methods, PCA, parallel processing, and performance comparison. Complete data preprocessing pipeline.`

**.gitignore:** Select `Python` template when creating repository

**License:** Select `MIT License` when creating repository
