# CO2-Emissions-Prediction--Multiple-Linear-Regression

This project uses **Multiple Linear Regression** to predict vehicle CO2 emissions based on **Engine Size** and **Fuel Consumption**.  

## Key Points

- **Regression Equation:**  
  CO2 Emissions = 10.09 + 1.32(Engine Size) + 22.18(Fuel Consumption)

- **Insights:**  
  - Fuel Consumption has a much stronger impact on CO2 than Engine Size.  
  - Engine Size contributes only a small increase per liter.  

- **Model Performance (Test Data):**  
  - R² = 0.99 → The model explains 99% of the variation in CO2 emissions.  
  - MSE = 38.31 → Predictions are very close to actual values.  

- **Visualization:**  
  - Predicted vs Actual plot shows points closely following the diagonal, confirming the model’s accuracy.

## Conclusion

Fuel consumption is the main driver of CO2 emissions, and the model predicts CO2 very accurately using just these two features.
