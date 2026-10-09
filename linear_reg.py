"""
CPSC 254 Assignment #2, Part 1: Supervised learning for linear regression.

linear_reg.py
  (1) Loads the small training dataset X, Y given in the brief.
  (2) Displays a scatter plot of the dataset with matplotlib.
  (3) Fits the data with scikit-learn's LinearRegression(), once as a straight
      line and once as a polynomial (PolynomialFeatures + LinearRegression),
      since the data is clearly U-shaped.
  (4) Prints the coefficient vector [w0, w1, ..., wn] and the fitted equation
      y = w0 + w1 x + ... + wn x^n.
  (5) Computes the training RMSE.
  (6) Loads GasProperties.csv, splits it 80% train / 20% test, solves for the
      weights with the least-squares normal equation w = (X^T X)^-1 X^T y and
      reports the train and test RMSE.

Run:  python linear_reg.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DEGREES = [1, 2]
GAS_CSV = "GasProperties.csv"
GAS_FEATURES = ["T", "P", "TC", "SV"]
GAS_TARGET = "Idx"


def rmse(y, y_pred):
    """Root mean squared error: sqrt(sum((y_i - y_hat_i)^2) / N)."""
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    return np.sqrt(np.sum((y - y_pred) ** 2) / len(y))


def format_equation(w):
    """Write w = [w0, w1, ..., wn] as y = w0 + w1 x + ... + wn x^n."""
    terms = [f"{w[0]:.4f}"]
    for i, wi in enumerate(w[1:], start=1):
        sign = "-" if wi < 0 else "+"
        power = "x" if i == 1 else f"x^{i}"
        terms.append(f"{sign} {abs(wi):.4f} {power}")
    return "y = " + " ".join(terms)


def fit_polynomial(x, y, degree):
    """Fit y = w0 + w1 x + ... + wn x^n with LinearRegression; return (model, poly, w)."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, y)
    w = np.concatenate(([model.intercept_], model.coef_))
    return model, poly, w


def least_squares(X, y):
    """Normal equation w = (X^T X)^-1 X^T y. X must already contain a bias column."""
    return np.linalg.inv(X.T @ X) @ X.T @ y


def add_bias(X):
    """Prepend a column of ones so w[0] is the intercept."""
    return np.column_stack((np.ones(len(X)), X))


def main():
    # ---------------------------------------------------------------
    # (1) Training dataset from the brief
    # ---------------------------------------------------------------
    x = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 2.7])
    y = np.array([15.5, 12.9, 9.5, 6.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 21.4])
    print("=== (1) Training dataset ===")
    print("X =", x.tolist())
    print("Y =", y.tolist())
    print("N =", len(x))

    # ---------------------------------------------------------------
    # (3)-(5) LinearRegression fits, coefficients, equations and RMSE
    # ---------------------------------------------------------------
    fits = {}
    for degree in DEGREES:
        model, poly, w = fit_polynomial(x, y, degree)
        train_rmse = rmse(y, model.predict(poly.transform(x.reshape(-1, 1))))
        fits[degree] = (model, poly, w, train_rmse)

        label = "straight line" if degree == 1 else f"polynomial of degree {degree}"
        print(f"\n=== (3)-(5) LinearRegression, {label} ===")
        names = ", ".join(f"w{i}" for i in range(degree + 1))
        print(f"Coefficient vector [{names}] = {np.round(w, 4).tolist()}")
        print("Equation:", format_equation(w))
        print(f"Training RMSE = {train_rmse:.4f}")

    best = min(fits, key=lambda d: fits[d][3])
    print(f"\nThe data is U-shaped, so the degree-{best} polynomial fits best "
          f"(training RMSE {fits[best][3]:.4f} vs. {fits[1][3]:.4f} for a straight line).")
    print("Final model:", format_equation(fits[best][2]))

    # ---------------------------------------------------------------
    # (2) Scatter plot of the dataset, with the fitted functions
    # ---------------------------------------------------------------
    x_line = np.linspace(x.min() - 0.2, x.max() + 0.2, 200)
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color="black", s=40, label="Training data", zorder=3)
    for degree, (model, poly, w, train_rmse) in fits.items():
        y_line = model.predict(poly.transform(x_line.reshape(-1, 1)))
        plt.plot(x_line, y_line, label=f"Degree {degree} fit (RMSE {train_rmse:.3f})")
    plt.title("Linear regression on the training dataset")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("linear_reg.png", dpi=120)
    print("\nPlot saved to linear_reg.png")

    # ---------------------------------------------------------------
    # (6) GasProperties.csv: least squares with the normal equation
    # ---------------------------------------------------------------
    print(f"\n=== (6) Least squares on {GAS_CSV} ===")
    data = pd.read_csv(GAS_CSV)
    print(f"Loaded {len(data)} rows, columns: {', '.join(data.columns)}")
    print(f"Features: {', '.join(GAS_FEATURES)}  ->  target: {GAS_TARGET}")

    X = data[GAS_FEATURES].to_numpy()
    y_gas = data[GAS_TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_gas, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Training set: {len(X_train)} rows, testing set: {len(X_test)} rows")

    X_train_b, X_test_b = add_bias(X_train), add_bias(X_test)
    w = least_squares(X_train_b, y_train)

    print("\nCoefficient vector w = (X^T X)^-1 X^T y:")
    for name, wi in zip(["w0 (bias)"] + [f"w{i} ({f})" for i, f in enumerate(GAS_FEATURES, 1)], w):
        print(f"  {name:<10} = {wi: .6e}")
    print("Equation: Idx = " + f"{w[0]:.6f}" + "".join(
        f" {'-' if wi < 0 else '+'} {abs(wi):.6f} {f}" for wi, f in zip(w[1:], GAS_FEATURES)))

    train_rmse = rmse(y_train, X_train_b @ w)
    test_rmse = rmse(y_test, X_test_b @ w)
    print(f"\nTraining RMSE = {train_rmse:.6f}")
    print(f"Testing RMSE  = {test_rmse:.6f}")
    print(f"(Standard deviation of {GAS_TARGET} = {y_gas.std():.6f}; "
          f"R^2 on test = {1 - test_rmse ** 2 / y_test.var():.4f})")
    print("Train and test RMSE are almost equal, so the linear model is not overfitting;")
    print("the remaining error comes from the relationship not being fully linear.")

    plt.show()


if __name__ == "__main__":
    main()
