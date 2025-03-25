import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('sample_sales_data.csv')

# Create sequences for Word2Vec
df['product_id'] = df['product_id'].astype(str)
sequences = df.groupby('customer_id')['product_id'].apply(list).tolist()

# Train Word2Vec model
w2v_model = Word2Vec(sentences=sequences, vector_size=16, window=5, min_count=1, workers=4)
product_vectors = {pid: w2v_model.wv[pid] for pid in w2v_model.wv.index_to_key}

# Prepare data for forecasting
df['embedding'] = df['product_id'].map(product_vectors)
df = df.dropna(subset=['embedding'])

X = np.vstack(df['embedding'].values)
y = df['sales'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regressor
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")
