# Retail Sales Prediction with Product Embeddings

A compact educational example that learns product representations from customer purchase groups and uses them as features for a Random Forest regressor.

**Scope:** this is sales-value prediction on a small sample. Although the repository name uses “forecasting,” the committed data has no dates and the code does not perform chronological future forecasting.

## Problem and idea

Product identifiers do not directly describe relationships between products. Word2Vec can encode products that appear in customer purchase groups as dense vectors. This example tests how those learned vectors can become inputs to a regression model.

## Data

[sample_sales_data.csv](sample_sales_data.csv) contains 10 example rows, five customer identifiers and five product identifiers.

| Column | Meaning |
|---|---|
| `customer_id` | Customer identifier used to group product interactions |
| `product_id` | Product token embedded by Word2Vec |
| `sales` | Numeric regression target |

The sample is for understanding the pipeline. It is too small for a reliable business-performance claim.

## Pipeline

1. Load the CSV with pandas.
2. Convert product identifiers to strings.
3. Group each customer's products into a token sequence.
4. Train Word2Vec and retrieve each product's embedding.
5. Stack the 16-dimensional vectors into a feature matrix.
6. Split rows into an 80% training and 20% test sample.
7. Fit a 100-tree Random Forest regressor.
8. Print test mean squared error.

| Component | Configuration |
|---|---|
| Word2Vec vector size | 16 |
| Context window | 5 |
| Minimum token frequency | 1 |
| Word2Vec workers | 4 |
| Regression model | `RandomForestRegressor(n_estimators=100)` |
| Test fraction | 0.2 |
| Split seed | 42 |

## Run the example

From the repository directory, install the imported libraries in an isolated Python environment:

```bash
python -m pip install pandas numpy gensim scikit-learn
python forecast_embeddings.py
```

These commands reflect the committed file layout. Dependencies are not version-pinned and this documentation update does not establish compatibility with every current Python/package version.

## Files

- [forecast_embeddings.py](forecast_embeddings.py): data loading, embedding training, regression and MSE calculation.
- [sample_sales_data.csv](sample_sales_data.csv): small example input.
- [README.md](README.md): method, usage and limitations.

## Evaluation limitations

- The test set contains only two rows.
- Product embeddings are learned from all interactions before splitting. A stricter inductive evaluation should fit representations using training data only and define how unseen products are handled.
- A random row split can place the same customers/products in both partitions; it does not measure future demand or cold-start performance.
- Word2Vec and the regressor are not fully seeded, so results may change between runs.
- No baseline, tuning study or persisted evaluation report is committed.

The script prints MSE; no fixed score or improvement over a baseline is claimed here.

## Practical extensions

For a forecasting version, add dated transactions and a chronological split, compare with simple demand baselines, learn representations within training boundaries and report MAE/RMSE by product and time period. For sales-value prediction, add an evaluation split that matches the intended use case and compare embeddings with one-hot or native categorical features.

## Skills demonstrated

Python, pandas, NumPy, Word2Vec/Gensim, feature construction, Random Forest regression and critical evaluation of small-data experiments.
