# Project Summary – Electricity Usage Predictor (PS 68)

**Problem:** Predict future monthly electricity consumption and the
resulting bill from historical usage data.

**Approach:**
1. Load `usage.csv` with Pandas; check for missing or non-numeric values
   and print a short data summary.
2. Fit a straight line `Units = slope × Month + intercept` using
   `numpy.polyfit` (degree 1).
3. Compute R² = 1 − SS_res / SS_tot to measure how well the line fits.
4. Predict units for the requested month and compute
   Estimated bill = predicted units × Rs. 6.
5. Print an Actual vs Predicted table with the Difference
   (Actual − Predicted) for every month, so the model's accuracy is clear.
6. Plot actual points, the regression line and the predicted point
   with Matplotlib (title, axis labels, legend).

**Edge cases handled:** fewer than 2 rows, missing file, empty file,
non-numeric values, invalid month input, invalid menu choice.

**Results:** All 4 test cases produce the expected output
(e.g., Case 4: slope 24.00, intercept 75.00, R² 0.99, 219 units, Rs. 1314.00).
The table prints after the 3 required output lines, so the expected
output is unchanged.

**Limitations:** A linear model ignores seasonality (e.g., summer AC usage);
more data and a non-linear model would improve accuracy.

**Tools:** Python 3, Pandas, NumPy, Matplotlib.
