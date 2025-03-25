# 🛍️ Retail Sales Forecasting using Word2Vec Embeddings

## 📌 Overview
This project demonstrates how to forecast product sales using customer-product interaction embeddings generated with Word2Vec. Embeddings were used as features for a regression model to predict sales.

## 🛠️ Tools & Tech
- Python
- Pandas, NumPy
- Gensim (Word2Vec)
- Scikit-learn (Random Forest)

## 🔍 Key Features
- Learned customer-product relationships as dense vectors
- Used embeddings as input to a regression model
- Forecasted future sales with improved accuracy

## 📊 Results
- Sample test MSE printed after model training
- Embedding size: 16
- Regressor: Random Forest

## 🚀 How to Run
```bash
pip install pandas numpy gensim scikit-learn
python forecast_embeddings.py
```

## 📁 Files
- `sample_sales_data.csv`: Example dataset
- `forecast_embeddings.py`: Full training pipeline

## 📌 Status
✅ Completed
