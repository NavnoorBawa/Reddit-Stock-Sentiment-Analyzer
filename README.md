# Real-Time Sentiment Analysis System

## Project Overview

This project implements a real-time sentiment analysis system for Reddit posts and comments. It analyzes public opinion about companies (Google, Tesla, Apple, etc.) by fetching live data from Reddit and classifying sentiment as Positive, Neutral, or Negative.

**Data Modality:** Textual Data (Reddit Comments/Posts)

**Dataset:** Reddit Sentiment Dataset from Kaggle (37K+ samples) or sample data

## Project Structure

```
Reddit_Sentiment_Analysis/
├── module_1_2_eda_preprocessing.py   # Data exploration and preprocessing
├── module_3_4_model_evaluation.py    # Model training and evaluation
├── module_5_dashboard.py             # Streamlit web dashboard
├── module_6_experiments.py           # AI exploration experiments
├── live_reddit_analyzer.py           # Live Reddit data fetcher
├── requirements.txt                  # Python dependencies
├── data/                             # Dataset files
├── models/                           # Trained models
└── outputs/                          # Visualizations and reports
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (for TextBlob):
```bash
python -c "import nltk; nltk.download('punkt')"
```

## How to Run

### Step 1: Data Exploration and Preprocessing
```bash
python module_1_2_eda_preprocessing.py
```
This generates EDA visualizations and preprocesses the data.

### Step 2: Model Training and Evaluation
```bash
python module_3_4_model_evaluation.py
```
This trains ML models and evaluates their performance.

### Step 3: Launch Dashboard
```bash
streamlit run module_5_dashboard.py
```
Opens a web interface for real-time sentiment analysis.

### Step 4: Run Experiments
```bash
python module_6_experiments.py
```
Tests the model with various inputs and edge cases.

### Step 5: Analyze Live Reddit Data
```bash
python live_reddit_analyzer.py
```
Fetches real Reddit posts about a company and analyzes sentiment.

## Module Descriptions

### Module 1 & 2: EDA and Preprocessing
- Loads and explores the Reddit sentiment dataset
- Performs word frequency analysis
- Identifies stopwords
- Cleans and preprocesses text
- Creates TF-IDF and Bag-of-Words features

### Module 3 & 4: Model Building and Evaluation
- Trains 5 ML models (Logistic Regression, SVM, Decision Tree, Random Forest, Naive Bayes)
- Evaluates using accuracy, precision, recall, F1-score
- Generates confusion matrix
- Compares training vs test performance

### Module 5: Deployment
- Streamlit web application
- Real-time text input and prediction
- Visual feedback with colors and emojis
- Response time tracking

### Module 6: Experiments
- Tests unexpected inputs
- Analyzes real-world text
- Studies modification effects
- Challenges sarcasm detection
- Stress tests with edge cases

### Live Reddit Analyzer
- Uses Reddit's free JSON API (no credentials needed)
- Fetches posts from multiple subreddits
- Generates sentiment reports for companies
- Saves results to CSV and JSON

## Free Reddit Data Access

This project uses Reddit's public JSON API. No API key required:

```
https://www.reddit.com/r/stocks/search.json?q=google
```

Just add `.json` to any Reddit URL to get JSON data.

## Results

The system achieves:
- High accuracy on sentiment classification
- Fast response time (< 50ms per prediction)
- Real-time dashboard with live analysis
- Comprehensive experiment documentation

## Made By

| Name | Roll Number | Section |
|------|-------------|---------|
| Saksham Verma | 102303892 | 3C63 |
| Navnoor Bawa | 102317164 | 3Q16 |
| Pulkit Garg | 102317214 | 3Q16 |

## Date

December 2025
