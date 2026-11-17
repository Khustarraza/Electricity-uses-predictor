"""PS 68 - Electricity Usage Predictor
Loads monthly usage from a CSV, fits Units = slope * Month + intercept,
predicts a future month, estimates the bill and plots the trend."""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RATE_PER_UNIT = 6  # Rs. per unit
CSV_FILE = "usage.csv"


def load_data(path):
    """Load CSV (with or without header), clean it and show a summary."""
    try:
        raw = pd.read_csv(path, header=None, dtype=str)
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty.")
        sys.exit(1)

    if raw.shape[1] < 2:
        print("Error: CSV must have two columns: Month and Units.")
        sys.exit(1)

    raw = raw.iloc[:, :2]
    raw.columns = ["Month", "Units"]

    # Drop header row if present (first row non-numeric)
    first = pd.to_numeric(raw.iloc[0], errors="coerce")
    if first.isna().all():
        raw = raw.iloc[1:]

    df = raw.apply(pd.to_numeric, errors="coerce")
    bad = int(df.isna().any(axis=1).sum())
    if bad:
        print(f"Warning: {bad} row(s) with missing/non-numeric values removed.")
        df = df.dropna()
    df = df.reset_index(drop=True)

    print("\n--- Data Summary ---")
    print(f"Rows loaded : {len(df)}")
    if len(df):
        print(f"Months      : {df['Month'].min():g} to {df['Month'].max():g}")
        print(f"Units (min/max/mean): {df['Units'].min():.2f} / "
              f"{df['Units'].max():.2f} / {df['Units'].mean():.2f}")
    print("--------------------")
    return df


def fit_model(df):
    """Fit degree-1 polynomial and compute R2."""
    x = df["Month"].to_numpy(dtype=float)
    y = df["Units"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return slope, intercept, r2


def print_table(df, slope, intercept, month, predicted):
    """Print Actual vs Predicted table with Difference."""
    line = ("+" + "-" * 9 + "+" + "-" * 12 + "+" + "-" * 12 +
            "+" + "-" * 14 + "+")
    print("\nActual vs Predicted Table")
    print(line)
    print(f"| {'Month':^7} | {'Actual':^10} | {'Predicted':^10} | "
          f"{'Difference':^12} |")
    print(line)
    for _, row in df.iterrows():
        m = row["Month"]
        actual = row["Units"]
        pred = slope * m + intercept
        diff = actual - pred
        print(f"| {m:^7g} | {actual:^10.2f} | {pred:^10.2f} | "
              f"{diff:^12.2f} |")
    print(line)
    print(f"| {month:^7g} | {'-':^10} | {predicted:^10.2f} | "
          f"{'-':^12} |  <- requested month")
    print(line)


def plot_results(df, slope, intercept, month, predicted):
    x = df["Month"].to_numpy(dtype=float)
    y = df["Units"].to_numpy(dtype=float)
    x_line = np.linspace(min(x.min(), month), max(x.max(), month), 100)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="blue", label="Actual usage")
    plt.plot(x_line, slope * x_line + intercept, color="red",
             label="Regression (trend) line")
    plt.scatter([month], [predicted], color="green", marker="*", s=150,
                label=f"Predicted (Month {month:g})")
    plt.title("Actual vs Predicted Electricity Usage")
    plt.xlabel("Month")
    plt.ylabel("Units consumed")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("usage_plot.png", dpi=150, bbox_inches="tight")
    plt.show()


def predict(df):
    if len(df) < 2:
        print("Error: At least 2 months of data are required for prediction.")
        return
    try:
        month = float(input("Enter month number to predict: ").strip())
    except ValueError:
        print("Error: Please enter a valid numeric month.")
        return

    slope, intercept, r2 = fit_model(df)
    predicted = slope * month + intercept
    bill = max(predicted, 0) * RATE_PER_UNIT

    print(f"Slope: {slope:.2f} | Intercept: {intercept:.2f} | R2: {r2:.2f}")
    print(f"Predicted units (Month {month:g}): {predicted:.2f}")
    print(f"Estimated bill: Rs. {bill:.2f}")
    print_table(df, slope, intercept, month, predicted)
    plot_results(df, slope, intercept, month, predicted)


def main():
    df = load_data(CSV_FILE)
    while True:
        print("\n1. Predict   2. Exit")
        choice = input("Choose option: ").strip()
        if choice == "1":
            predict(df)
        elif choice == "2":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Enter 1 or 2.")


if __name__ == "__main__":
    main()