# 🏠 House Price Prediction Using Forward & Backward Propagation

This project implements a **`single-layer neural network from scratch`** for house price prediction.

The main goal is to understand how a neural network works internally using **forward propagation, backward propagation, MSE loss, and gradient descent**.

## Project Overview

The project follows this workflow:

```text
Data Preprocessing
        ↓
Weight & Bias Initialization
        ↓
Forward Propagation
        ↓
MSE Loss
        ↓
Backward Propagation
        ↓
Gradient Descent
        ↓
Model Training
        ↓
ReLU vs Tanh Comparison
```

## Dataset

The dataset contains house-related features such as:

- Bedrooms
- Bathrooms
- Square Footage
- House Age
- Distance to City Center
- Has Garage
- Neighborhood Quality
- Lot Size
- Year Built
- Number of Floors

**Target:** `House Price`

## Data Preprocessing

The project performs:

- Missing value handling
- Categorical feature encoding
- Feature normalization
- Train-test splitting

Numerical missing values are handled using **median imputation**, while the categorical feature is handled using **mode imputation**.

`Neighborhood Quality` is encoded using one-hot encoding.

## Neural Network

The model uses a single-layer architecture:

```text
Input Features
      ↓
Linear Combination
      ↓
Activation Function
      ↓
Prediction
```

The linear combination is:

```python
z = X @ W + b
```

where:

- `X` = input features
- `W` = weights
- `b` = bias

## Activation Functions

The project compares:

### ReLU

```text
ReLU(z) = max(0, z)
```

### Tanh

```text
Tanh(z) = tanh(z)
```

Both are trained and compared based on their **loss and gradient behavior**.

## Loss Function

The model uses **Mean Squared Error (MSE)**:

```text
MSE = mean((y_true - y_pred)²)
```

## Backward Propagation

Gradients are calculated using the chain rule.

```text
dL/dz
dW
db
```

The weights and bias are then updated using gradient descent:

```text
W = W - learning_rate × dW
b = b - learning_rate × db
```

## Results & Visualization

The notebook includes:

- Loss curves
- Gradient curves
- ReLU vs Tanh comparison
- Test predictions
- Test MSE

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Project Structure

```text
House-Price-Prediction/
│
├── HousePrice.ipynb
├── pic
├── data/
│   └── house_price_prediction_dataset.csv
└── House Price Prediction - ReLU vs Tanh Analysis.pdf
└── README.md
```

## Learning Objective

This project was created to understand the fundamental concepts behind neural-network training, including:

- Forward propagation
- Backward propagation
- Activation functions
- MSE loss
- Gradients
- Gradient descent
- Model training

## How to Run

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Then open the notebook:

```bash
jupyter notebook
```

Run the notebook cells sequentially.

---

### Project

**House Price Prediction — Forward & Backward Propagation**

Built to understand the fundamentals of neural networks from scratch.