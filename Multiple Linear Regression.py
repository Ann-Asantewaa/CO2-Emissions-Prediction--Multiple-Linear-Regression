#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression
# 
# 
# ## Objectives
# 
# * Use scikit-learn to implement multiple linear regression
# * Create, train, and test a multiple linear regression model on real data.
# 
# ## Import needed packages
# 
# - Numpy
# - Matplotlib
# - Pandas
# - Scikit-learn
# 
# Execute these cells to check if you have the above packages
# 
# 

# In[1]:


get_ipython().system('pip install -q numpy pandas scikit-learn matplotlib')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the data
# 
# Use pandas to load the data

# In[3]:


df = pd.read_csv('my2026-fuel-consum.csv')
# verify successful load with some randomly selected records
df.sample(5)


# # Exploration & Variable Selection

# In[4]:


df.describe()


# ## Summary Statistics Explanation
# 
# ### Count
# - Represents the total number of vehicles in the dataset.
# - All variables have **437 observations**.
# 
# ### Model Year
# - All vehicles are from the **2026 model year**.
# - This is why the mean, minimum, and maximum values are the same.
# 
# ### Engine Size (L)
# - The average engine size is **3.05 liters**.
# - Engine sizes range from **1.2 L to 6.7 L**.
# 
# ### Cylinders
# - Vehicles have between **3 and 12 cylinders**.
# - Most vehicles have **4 to 6 cylinders**.
# 
# ### Fuel Consumption (Combined)
# - Average combined fuel consumption is **10.9 L/100km**.
# - This combines city and highway driving.
# 
# ### Fuel Consumption (MPG)
# - Average fuel efficiency is **28 mpg**.
# - Higher MPG indicates better fuel efficiency.
# 
# ### CO₂ Emissions (g/km)
# - Average CO₂ emissions are **255 g/km**.
# - Higher values mean greater environmental impact.

# In[5]:


print(df.columns)


# In[6]:


# Drop categoricals 
df = df.drop(columns=[
    'Make',
    'Model',
    'VehicleClass',
    'Transmission',
    'FuelType',
    'ModelYear'
])



# Checking the level of independence between variables after eliminating categoricals
# 
# Use correlation matrix.
# It helps to indicate the independence between the variables
# It helps to indicate how predictive each variable is of the target
# 
# Take out strong dependencies or correlation between variables by selecting the best one from each correlated group.

# In[7]:


df.corr()


# ## Correlation Analysis 
#  Focus on the target variable
# 
# **Target:** `CO2emissions (g/km)` 
# 
# Most variables show a strong correlation with the target:
# 
# - `FuelConsumptionCombW` → 0.985  
# - `FuelConsumptionCity` → 0.973  
# - `FuelConsumptionHighway` → 0.945  
# - `EngineSize (L)` → 0.780  
# - `Cylinders` → 0.768  
# 
#  This indicates that these features are good candidates for predicting CO2 emissions.
# 
#  Examining correlations between predictors
# 
# - `EngineSize (L)` and `Cylinders` are highly correlated (0.931).  
#   - Since `EngineSize (L)` is more strongly correlated with the target (0.780 > 0.768), we can **drop `Cylinders`**.
# 
# - The fuel consumption variables are highly correlated with each other:  
#   - `FuelConsumptionCity`, `FuelConsumptionHighway`, `FuelConsumptionCombW`, `FuelConsumptionCombMpg`  
#   - Among these, `FuelConsumptionCombW` is most correlated with the target (0.985).  
#   - Therefore, we can **drop the others**:  
#     - `FuelConsumptionCity`  
#     - `FuelConsumptionHighway`  
#     - `FuelConsumptionCombMpg`
# 
# - `FuelConsumptionCombW` and `FuelConsumptionCombMpg` are not perfectly correlated (−0.931).  
#   - This might be due to differences in units or data quality, which should be checked in practice.
# 
# - Other variables like `CO2rating` (−0.974) and `SmogRating` (−0.505) also correlate with CO2 emissions, but less strongly.
# 

# In[8]:


df = df.drop(
    ['Cylinders', 'FuelConsumptionCity', 'FuelConsumptionHighway', 'FuelConsumptionCombMpg', 'CO2rating', 'SmogRating'],
    axis=1
)


# In[9]:


df.head(5)


# In[10]:


axes = pd.plotting.scatter_matrix(df, alpha=0.2)
# need to rotate axis labels so we can read them
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')
    
plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()


#  Key Findings
# 
# **Relationships Between Variables:**
# - **Engine Size - Fuel Consumption:** Positive correlation (larger engines use more fuel)
# - **Engine Size - CO2 Emissions:** Strong positive correlation (larger engines produce more CO2)
# - **Fuel Consumption - CO2 Emissions:** Very strong linear relationship (almost perfect correlation)
# 
#  Implications for Predicting CO2 Emissions
# 
# - **Best predictor:** Fuel Consumption has the strongest linear relationship with CO2 Emissions
# - **Secondary predictor:** Engine Size also shows good predictive power
# - **Multicollinearity note:** Engine Size and Fuel Consumption are correlated with each other, which may affect coefficient interpretation in multiple linear regression
# - **Linear relationships:** All relationships appear linear, making them suitable for linear regression modeling
# 
#  Conclusion
# Both Engine Size and Fuel Consumption can be used to predict CO2 Emissions, with Fuel Consumption being the stronger predictor.

# ## Extract the input variables 
# 
#  Extract the required columns and convert the resulting dataframes to NumPy arrays.
# 
# - `.iloc[:, [0, 1]]` → selects columns 0 and 1 as a 2D array
# -  `.iloc[:, 2]` → selects column 2 as a 1D array
# -  `.to_numpy()` → converts to NumPy arrays (needed for scikit-learn)

# In[11]:


X = df.iloc[:,[0,1]].to_numpy()
Y = df.iloc[:,[2]].to_numpy()


# ## Preprocess selected variables
# 
# Standardize the input features so the model doesn't inadvertently favor any feature due to its magnitude. 
# 
# To do this is to subtract the mean and divide by the standard deviation. 
# 
# Scikit-learn does the above.
# 

# In[12]:


from sklearn import preprocessing

std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)


# **Purpose:**  
# Scale all features to have **mean = 0** and **standard deviation = 1**.
# 
# **Why:**  
# Features like `EngineSize (L)` and `FuelConsumptionCombW` have different units and ranges. Standardization ensures that **no single feature dominates** due to its scale.
# 
# **Important:**  
# Only standardize the **features (`X`)**, not the **target (`Y`)**.

# In[13]:


pd.DataFrame(X_std).describe().round(2)


# ## Create train and test datasets
# 
# Randomly split your data into train and test sets, using 80% of the dataset for training and reserving the remaining 20% for testing.
# 

# In[14]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_std,Y,test_size=0.2,random_state=42)


#  ## Build a multiple linear regression model
#  
#  Multiple linear regression model can be implemented with exactly the same scikit-learn tools.
# 

# In[15]:


from sklearn import linear_model

#create a model object
regressor = linear_model.LinearRegression()

#train the model in the training data
regressor.fit(X_train, Y_train)

#print the coefficients
coef_ = regressor.coef_
intercept_ = regressor.intercept_

print('Coefficients: ', coef_)
print('Intercept: ', intercept_)


# 
#  Linear Regression Output
# 
# - 1.72063924 → the coefficient for EngineSize (L)  
#   for every 1 unit increase in EngineSize (L), CO2 increases by ~1.72 units, holding the other feature constant.
# 
# - 64.16734176 → the coefficient for FuelConsumptionCombW  
#   for every 1 unit increase in FuelConsumptionCombW, CO2 increases by ~64.17 units, holding the other feature constant.
# 
# - Intercept: 255.25280559  
#   This is the predicted CO2emissions when all features are zero.
# 
# 

# In[16]:


# Get the standard scaler's mean and standard deviation parameters
means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)

# The least squares parameters can be calculated relative to the original, unstandardized variable 
coef_original = coef_ / std_devs_
intercept_original = intercept_ - np.sum((means_ *coef_) / std_devs_)

print('Coefficients: ', coef_original)
print('Intercept: ', intercept_original)


# Intercept Interpretation
# 
# - The intercept is 10.09 g/km, which is the predicted CO2 emissions when EngineSize = 0 and FuelConsumptionCombW = 0.  
# - Physically, CO2 should be zero in this case, so the non-zero intercept reflects the **limitations of the linear model**.  
# 
#  Why it happens
# 
# - CO2 emissions do not have a perfectly linear relationship with the features.  
# - Outliers in the dataset may influence the best-fit line.  
# - Some variables may still be correlated, affecting the intercept.  
# 
# **Key point:** The small non-zero intercept does not invalidate the model; it is a normal consequence of fitting a linear model to real-world data.
# 

# ## Visualize model outputs
# 
# You can visualize the goodness-of-fit of the model to the training data by plotting each variable separately as a best-fit line using the corresponding regression parameters.

# In[17]:


plt.scatter(X_train[:,0], Y_train,  color='green')
plt.plot(X_train[:,0], coef_[0,0] * X_train[:,0] + intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[18]:


plt.scatter(X_train[:,1], Y_train,  color='blue')
plt.plot(X_train[:,1], coef_[0,1] * X_train[:,1] + intercept_[0], '-r')
plt.xlabel("FuelConsumption_CombW")
plt.ylabel("Emission")
plt.show()


# ## Model Evaluation
# 

# In[19]:


#Trained model to predict on the test set
Y_pred = regressor.predict(X_test)

# Comparison of predicted & Actuals
print(Y_pred[:5])
print(Y_test[:5])


# In[20]:


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(Y_test, Y_pred)
r2  = r2_score(Y_test, Y_pred)

print("MSE:", mse)
print("R²:", r2)


# ## Model Performance Interpretation
# 
# - **Mean Squared Error (MSE = 38.31):**  
#   On average, the model’s predictions are very close to the actual CO₂ emission values, indicating low prediction error.
# 
# - **R² = 0.99:**  
#   The model explains about **99% of the variation** in CO₂ emissions, showing an excellent fit on the test data.
# 

# In[21]:


#Actual vs Predicted (Plot)
plt.scatter(Y_test, Y_pred, alpha=0.7)
plt.plot([Y_test.min(), Y_test.max()],
         [Y_test.min(), Y_test.max()], 'r--')

plt.xlabel('Actual CO2 Emissions')
plt.ylabel('Predicted CO2 Emissions')
plt.title('Actual vs Predicted CO2 (Test Data)')
plt.show()


# # Multiple Linear Regression Results - Simple Explanation
# 
# ## The Regression Equation
# **CO2 Emissions = 10.09 + 1.32(Engine Size) + 22.18(Fuel Consumption)**
# 
# 
# ### Coefficient 1: Engine Size = 1.32
# - For every 1 liter increase in engine size, CO2 emissions increase by **1.32 g/km**
# - **Small effect** - this matches what we saw in Plot 1 (green) where the relationship was weak
# 
# ### Coefficient 2: Fuel Consumption = 22.18
# - For every 1 unit increase in fuel consumption, CO2 emissions increase by **22.18 g/km**
# - **HUGE effect** - this matches what we saw in Plot 2 (blue) where the relationship was very strong
# 
# ### Intercept = 10.09
# - The baseline CO2 emission when both predictors are zero (not meaningful in practice)
# 
# 
# ###  Model Performance on Test Data
# 
# - **R² = 0.99**, meaning the model explains about **99% of the variation** in CO2 emissions.
# - **MSE = 38.31**, indicating very small prediction errors on unseen data.
# 
# 
# ### Actual vs Predicted CO2 Emissions (Test Data)
# 
# - The predicted values lie very close to the actual values.
# 
# 
# ### Simple Takeaway
# **Fuel Consumption is 17 times more important than Engine Size** in predicting CO2 emissions (22.18 ÷ 1.32 ≈ 17). 
# - The plots visually showed this: tight blue line vs scattered green points.
# - The strong model performance and the close match between actual and predicted values confirm that the linear model is both accurate and reliable for this dataset.
