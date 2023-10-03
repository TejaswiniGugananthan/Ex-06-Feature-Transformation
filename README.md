# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
STEP 1:
Read the given Data

STEP 2:
Clean the Data Set using Data Cleaning Process

STEP 3:
Apply Feature Transformation techniques to all the features of the data set

STEP 4:
Save the data to the file

# PROGRAM AND OUTPUT:

Name : G.TEJASWINI

Reg no : 212222230157

```python
import pandas as pd
from scipy import stats
import numpy as np
```
```python
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="431" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-06-Feature-Transformation/assets/121222763/802475b7-5d7f-4e49-b5e1-6970476918d2">

```python
df.skew()
```
<img width="161" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-06-Feature-Transformation/assets/121222763/65b9aa59-7d55-430a-bad0-d9442361fb1f">

```python
np.log(df["Highly Positive Skew"])
```
<img width="260" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-06-Feature-Transformation/assets/121222763/b1785b0c-3b00-45aa-8499-3b49e988db51">

```python
np.reciprocal(df["Moderate Negative Skew"])
```
<img width="269" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-06-Feature-Transformation/assets/121222763/43c9087a-a8b2-4f8b-8e6b-43539e9c9f6b">

```python
np.sqrt(df["Highly Positive Skew"])
```
<img width="257" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-06-Feature-Transformation/assets/121222763/bed87ad1-fa46-4560-b07f-f2f51bf699cf">

```python
np.square(df["Highly Positive Skew"])
```
<img width="262" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-06-Feature-Transformation/assets/121222763/ba9e25c0-ab41-4b3e-a8e5-41229c342810">

```python
df["Highly Positive Skew_boxcox"],parameter=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="559" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-06-Feature-Transformation/assets/121222763/06606cf6-1ac0-4eec-b7d7-a646dc92beb6">

```python
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
```
```python
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="820" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-06-Feature-Transformation/assets/121222763/eb5d7795-1871-4c13-958a-8db5b891d6f1">

```python
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
```
```python
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
<img width="333" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-06-Feature-Transformation/assets/121222763/e59cc539-2183-4d3b-a5b8-ccc188adee33">

```python
sm.qqplot(df['Moderate Negative Skew_1'],line='45')
plt.show()
```
<img width="326" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-06-Feature-Transformation/assets/121222763/3b4a0a7f-16e4-482f-84da-6b4514586994">

```python
df['Highly Negative Skew_1']=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
<img width="328" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-06-Feature-Transformation/assets/121222763/7cbd9a6a-c544-4b89-9e1e-53f689ee2760">

```python
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
<img width="327" alt="image" src="https://github.com/TejaswiniGugananthan/Ex-06-Feature-Transformation/assets/121222763/7f2a5e99-f324-4dc1-93f5-b0d63f18d524">

# RESULT:
Thus,Feature transformation is performed and executed successfully for the given dataset.






























