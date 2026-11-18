# Market Recommendation System on Amazon SageMaker

A hybrid product recommendation system built using Python, NLP, and machine learning techniques to analyze Amazon product reviews, identify product similarities, and generate relevant item-to-item recommendations.

## Project Overview

This project focuses on developing a hybrid recommendation system that combines product review content and rating-based similarity to recommend relevant products.

The system uses TF-IDF and Truncated SVD to extract meaningful features from product reviews, K-Means to group products into distinct clusters, and a hybrid scoring mechanism to rank product recommendations.

The application is developed in Python, runs in an Amazon SageMaker environment, and provides an interactive Streamlit dashboard for recommendation exploration and product analytics.

## Key Features

### 1. Hybrid Product Recommendation
- Combines content-based similarity and rating-based similarity.
- Uses TF-IDF and cosine similarity to identify products with similar review content.
- Incorporates aggregated product ratings into the recommendation score.
- Applies a weighted scoring approach:
  - 70% Content Similarity
  - 30% Rating Similarity
- Generates a Top 10 list of recommended products.

### 2. Product Clustering & Analytics
- Applies TF-IDF with unigram and bigram features.
- Uses Truncated SVD with 100 components for dimensionality reduction.
- Implements K-Means clustering to group 6,300 unique products into five clusters.
- Analyzes product ratings, review counts, review length, and helpful vote ratios.

The five identified product segments are:

| Cluster | Product Segment | Products | Average Rating |
|---|---|---:|---:|
| 0 | Featured Products | 1,245 | 4.62 |
| 1 | Popular Accessories | 2,588 | 3.85 |
| 2 | Problematic Products | 890 | 2.15 |
| 3 | Niche Products | 1,115 | 3.91 |
| 4 | Controversial Products | 462 | 4.28 |

### 3. Cloud-Based Interactive Deployment
The application is developed and executed in an Amazon SageMaker Notebook Instance environment.

The Streamlit dashboard provides five main modules:

- **Dashboard:** Overview of product metrics, rating distributions, and sentiment analysis.
- **AI Recommendations:** Interactive product search and hybrid recommendation generation.
- **Advanced Analytics:** Statistical analysis, correlation matrices, box plots, and violin plots.
- **Visual Analytics:** Interactive charts, including funnel, sunburst, and 3D scatter plots.
- **Data Explorer:** Product filtering and CSV/Excel export functionality.

## System Architecture

```text
Amazon Product Review Dataset
            |
            v
     Data Preprocessing
            |
            v
    Text Feature Extraction
       (TF-IDF, N-gram)
            |
            v
      Truncated SVD
       (100 Components)
            |
       +----+----+
       |         |
       v         v
   K-Means   Cosine Similarity
  Clustering       |
       |           |
       |      Rating Similarity
       |           |
       |           v
       |    Hybrid Recommendation
       |       (70% + 30%)
       |           |
       +-----+-----+
             |
             v
     Streamlit Dashboard
             |
             v
      Amazon SageMaker
```

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, K-Means |
| NLP & Feature Extraction | TF-IDF, N-gram, Truncated SVD |
| Recommendation | Cosine Similarity, Hybrid Scoring |
| Data Visualization | Plotly, Matplotlib, Seaborn |
| Application Framework | Streamlit |
| Cloud Platform | Amazon SageMaker |
| Model Serialization | Joblib, Pickle |

## Project Structure

```text
Market-analisis/
│
├── preprocessing.py
├── hybrid_recommender.py
├── clustering.py
├── analytics.py
├── dashboard_app.py
│
├── model_tfidf.pkl
├── model_svd.pkl
├── model_kmeans.pkl
├── hybrid_matrix.pkl
│
└── README.md
```

## Getting Started

### Prerequisites

- Python
- Amazon SageMaker Notebook Instance or a compatible Python environment
- Required Python libraries:
  - pandas
  - numpy
  - scikit-learn
  - streamlit
  - plotly
  - matplotlib
  - seaborn
  - joblib

### Run the Application

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn streamlit plotly matplotlib seaborn joblib
```

Run the Streamlit application:

```bash
python -m streamlit run dashboard_app.py --server.address=0.0.0.0 --server.port=8501
```

The application can then be accessed through the configured SageMaker notebook access or port-forwarding mechanism.

## Key Findings

- Successfully grouped 6,300 unique products into five distinct clusters.
- Identified product segments with different rating patterns and review characteristics.
- Developed a hybrid recommendation mechanism combining review-content similarity and product-rating similarity.
- Generated Top 10 product recommendations based on the hybrid scoring approach.
- Built an interactive dashboard integrating recommendation functionality and product analytics.

## Limitations

- The recommendation system primarily supports item-to-item recommendations rather than fully personalized user-level recommendations.
- New products without reviews may still face cold-start challenges.
- Recommendation quality depends on the availability and quality of product reviews.
- Standard ranking evaluation metrics such as Precision, Recall, F1-Score, and NDCG have not yet been implemented using a separate test dataset.
- Similarity calculations may require further optimization for substantially larger product catalogs.

## Future Improvements

- Implement Neural Collaborative Filtering or other deep learning recommendation architectures.
- Improve cold-start handling using product metadata and popularity-based recommendations.
- Integrate advanced NLP models such as BERT for deeper review understanding.
- Implement recommendation evaluation using Precision@K, Recall@K, and NDCG@K.
- Optimize recommendation retrieval using Approximate Nearest Neighbors (ANN).

## Project Links

- **GitHub Repository:** [Market-analisis](https://github.com/naufalubaidillahizzat78-bit/Market-analisis)
- **Live Demo:** [Amazon Market Recommendation System](https://market-amazon.streamlit.app/)

---

*Developed as an applied machine learning project focusing on product recommendation, NLP, product analytics, and cloud-based application development.*
