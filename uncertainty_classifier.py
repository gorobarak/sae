import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


with open("checkpoints/gemma-2-2b_layer_24/uncertainty_dataset_OpenWebText_10000.pkl", "rb") as f:
    uncertainty_dataset = pickle.load(f)
df = pd.DataFrame(uncertainty_dataset)


# %%
X = np.stack(df["sae_histogram"].to_numpy(), axis=0)
y = df["perplexity"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# fit a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# %%
# make predictions on the test set
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)
# %%
# evaluate the model
mse_test = mean_squared_error(y_test, y_pred_test)
mse_train = mean_squared_error(y_train, y_pred_train)
# mean absolute error
mae_test = np.mean(np.abs(y_test - y_pred_test))
mae_train = np.mean(np.abs(y_train - y_pred_train))
print(f"Mean Absolute Error on test set: {mae_test}")
print(f"Mean Absolute Error on train set: {mae_train}")
print(f"Mean Squared Error on test set: {mse_test}")
print(f"Mean Squared Error on train set: {mse_train}")
# %%
# predict on the average of whole dataset
mean_X = np.mean(X, axis=0, keepdims=True)
mean_y = np.mean(y)
mean_y_pred = model.predict(mean_X)
print(f"Mean perplexity on whole dataset: {mean_y}")
print(f"Predicted mean perplexity on whole dataset: {mean_y_pred[0]}")
print(f"Square error: {(mean_y - mean_y_pred[0])**2}")
# %%
# prediction on the average of test set
mean_X_test = np.mean(X_test, axis=0, keepdims=True)
mean_y_test = np.mean(y_test)
mean_y_pred_test = model.predict(mean_X_test)
print(f"Mean perplexity on test set: {mean_y_test}")
print(f"Predicted mean perplexity on test set: {mean_y_pred_test[0]}")
print(f"Square error: {(mean_y_test - mean_y_pred_test[0])**2}")
# %%
print(np.count_nonzero(mean_X_test))
# %%
print(f"num of non zero coefficients: {np.count_nonzero(model.coef_)}")
print(model.coef_.shape)
# %%
# visualize the estimatior function
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('True Perplexity')
plt.ylabel('Predicted Perplexity')
plt.title('True vs Predicted Perplexity on Test Set')
plt.show()