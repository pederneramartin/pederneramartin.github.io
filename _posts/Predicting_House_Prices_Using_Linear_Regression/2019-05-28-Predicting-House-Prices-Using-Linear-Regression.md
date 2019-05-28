---
title: "Predicting house prices using Linear Regression"
layout: post
date: 2019-05-28 18:30
tag: 
- python
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
image:
headerImage: False
projects: true
hidden: false # don't count this post in blog pagination
description: "Building a model to predict house prices with a high predective accuracy"
category: project
author: pederneramartin
externalLink: false
---

# Predicting House Prices Using Linear Regression

This challange is part of a Kaggle Competition. More info can be found [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)

The problem is to build a model that will predict house prices with a high degree of predictive accuracy given the available data. “With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.”

## Dataset

The dataset is the prices and features of residential houses sold from 2006 to 2010 in Ames, Iowa. Obtained from [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

## Environment and tools

- Jupyter notebook
- numpy
- pandas
- seaborn
- matplotlib
- scipy
- scikit-learn

### Import libraries


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew,norm,kurtosis
from scipy.stats.stats import pearsonr
%matplotlib inline
```

### Import dataset


```python
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```

### Drop Id column

Drop the ‘Id’ column in the data as it is not necessary for prediction


```python
train_id = train["Id"]
test_id = test["Id"]

train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)
```

## Exploring the data

The test data doesn’t have the target variable which is the ‘SalePrice’.

Here is a data [description file](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/data_description.txt)
- SalePrice — the property’s sale price in dollars. This is the target variable that you’re trying to predict.
- LotArea: Lot size in square feet
- Neighborhood: Physical locations within Ames city limits
- OverallQual: Overall material and finish quality
- OverallCond: Overall condition rating
- YearBuilt: Original construction date
- TotalBsmtSF: Total square feet of basement area
- GrLivArea: Above grade (ground) living area square feet
- ....


```python
train.describe()
```




<div style="overflow-x:auto; text-align: center">
<style type="text/css">
table.padded-table th { padding:10px; }
</style>
<table class="table-striped padded-table">
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1452.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>...</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>56.897260</td>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
      <td>103.685262</td>
      <td>443.639726</td>
      <td>46.549315</td>
      <td>...</td>
      <td>94.244521</td>
      <td>46.660274</td>
      <td>21.954110</td>
      <td>3.409589</td>
      <td>15.060959</td>
      <td>2.758904</td>
      <td>43.489041</td>
      <td>6.321918</td>
      <td>2007.815753</td>
      <td>180921.195890</td>
    </tr>
    <tr>
      <th>std</th>
      <td>42.300571</td>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>20.645407</td>
      <td>181.066207</td>
      <td>456.098091</td>
      <td>161.319273</td>
      <td>...</td>
      <td>125.338794</td>
      <td>66.256028</td>
      <td>61.119149</td>
      <td>29.317331</td>
      <td>55.757415</td>
      <td>40.177307</td>
      <td>496.123024</td>
      <td>2.703626</td>
      <td>1.328095</td>
      <td>79442.502883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>129975.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>383.500000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>163000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>166.000000</td>
      <td>712.250000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>1474.000000</td>
      <td>...</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>15500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 37 columns</p>
</div>




```python
test.describe()
```




<div style="overflow-x:auto; text-align: center">
<style type="text/css">
table.padded-table th { padding:10px; }
</style>
<table class="table-striped padded-table">
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1459.000000</td>
      <td>1232.000000</td>
      <td>1459.000000</td>
      <td>1459.000000</td>
      <td>1459.000000</td>
      <td>1459.000000</td>
      <td>1459.000000</td>
      <td>1444.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>...</td>
      <td>1458.000000</td>
      <td>1459.000000</td>
      <td>1459.000000</td>
      <td>1459.000000</td>
      <td>1459.000000</td>
      <td>1459.000000</td>
      <td>1459.000000</td>
      <td>1459.000000</td>
      <td>1459.000000</td>
      <td>1459.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>57.378341</td>
      <td>68.580357</td>
      <td>9819.161069</td>
      <td>6.078821</td>
      <td>5.553804</td>
      <td>1971.357779</td>
      <td>1983.662783</td>
      <td>100.709141</td>
      <td>439.203704</td>
      <td>52.619342</td>
      <td>...</td>
      <td>472.768861</td>
      <td>93.174777</td>
      <td>48.313914</td>
      <td>24.243317</td>
      <td>1.794380</td>
      <td>17.064428</td>
      <td>1.744345</td>
      <td>58.167923</td>
      <td>6.104181</td>
      <td>2007.769705</td>
    </tr>
    <tr>
      <th>std</th>
      <td>42.746880</td>
      <td>22.376841</td>
      <td>4955.517327</td>
      <td>1.436812</td>
      <td>1.113740</td>
      <td>30.390071</td>
      <td>21.130467</td>
      <td>177.625900</td>
      <td>455.268042</td>
      <td>176.753926</td>
      <td>...</td>
      <td>217.048611</td>
      <td>127.744882</td>
      <td>68.883364</td>
      <td>67.227765</td>
      <td>20.207842</td>
      <td>56.609763</td>
      <td>30.491646</td>
      <td>630.806978</td>
      <td>2.722432</td>
      <td>1.301740</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1470.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1879.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.000000</td>
      <td>58.000000</td>
      <td>7391.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1953.000000</td>
      <td>1963.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>318.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>2007.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>67.000000</td>
      <td>9399.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1992.000000</td>
      <td>0.000000</td>
      <td>350.500000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>480.000000</td>
      <td>0.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11517.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2001.000000</td>
      <td>2004.000000</td>
      <td>164.000000</td>
      <td>753.500000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>576.000000</td>
      <td>168.000000</td>
      <td>72.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>190.000000</td>
      <td>200.000000</td>
      <td>56600.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1290.000000</td>
      <td>4010.000000</td>
      <td>1526.000000</td>
      <td>...</td>
      <td>1488.000000</td>
      <td>1424.000000</td>
      <td>742.000000</td>
      <td>1012.000000</td>
      <td>360.000000</td>
      <td>576.000000</td>
      <td>800.000000</td>
      <td>17000.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 36 columns</p>
</div>




```python
#Shape of train and test dataset

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
```

    Train shape: (1460, 80)
    Test shape: (1459, 79)



```python
#Select the Numerical & Categorical Features

numerical_features = train.select_dtypes(exclude = ['object']).columns
categorical_features = train.select_dtypes(include = ['object']).columns
```


```python
# Plotting the numerical columns

fig = plt.figure(figsize = (15,15))
ax = fig.gca()
train[numerical_features].hist(ax=ax)
fig.tight_layout()
fig.show()
```

![png](/assets/images/Predicting-House-Prices-Using-Linear-Regression/output_21_1.png)



```python
#Plot the Numeric columns against SalePrice Using ScatterPlot

fig = plt.figure(figsize=(15,30))
for i,col in enumerate(numerical_features[1:]):
    fig.add_subplot(12,3,1+i)
    plt.scatter(train[col], train['SalePrice'])
    plt.xlabel(col)
    plt.ylabel('SalePrice')
fig.tight_layout()
fig.show()
```

![png](/assets/images/Predicting-House-Prices-Using-Linear-Regression/output_22_1.png)



```python
#Use bar plots to plot categorical features against SalePrice

fig = plt.figure(figsize=(15,50))
for i, col in enumerate(categorical_features):
    fig.add_subplot(11,4,1+i)
    train.groupby(col).mean()['SalePrice'].plot.bar(yerr = train.groupby(col).std())
fig.tight_layout()
fig.show()
```

![png](/assets/images/Predicting-House-Prices-Using-Linear-Regression/output_23_1.png)


## Data Preprocessing


```python
sns.set_style('darkgrid')
fig, ax = plt.subplots()
sns.regplot(train['GrLivArea'], train['SalePrice'])
#ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
```


![png](/assets/images/Predicting-House-Prices-Using-Linear-Regression/output_25_0.png)


We can see at the bottom right two with extremely large GrLivArea that are of a low price. These values are huge oultliers. Therefore, we can safely delete them.



```python
#Deleting outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
```


```python
fig, ax = plt.subplots()
sns.regplot(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
```


![png](/assets/images/Predicting-House-Prices-Using-Linear-Regression/output_28_0.png)


### Sales Price Tragert Variable


```python
train["SalePrice"].describe()
```




    count      1458.000000
    mean     180932.919067
    std       79495.055285
    min       34900.000000
    25%      129925.000000
    50%      163000.000000
    75%      214000.000000
    max      755000.000000
    Name: SalePrice, dtype: float64




```python
sns.distplot(train["SalePrice"],fit=norm)
```
![png](/assets/images/Predicting-House-Prices-Using-Linear-Regression/output_31_1.png)


The target variable is right skewed. As (linear) models love normally distributed data , we need to transform this variable and make it more normally distributed.

The need for data transformation can depend on the modeling method that you plan to use. For linear and logistic regression, for example, you ideally want to make sure that the relationship between input variables and output variables is approximately linear, that the input variables are approximately normal in distribution, and that the output variable is constant variance (that is, the variance of the output variable is independent of the input variables). 
We need to transform some of our input variables to better meet these assumptions.


```python
stats.probplot(train["SalePrice"], plot=sns.mpl.pyplot)
```

![png](/assets/images/Predicting-House-Prices-Using-Linear-Regression/output_33_1.png)


The SalePrice deviates from normal distribution and is positively biased.

The SalePrice also does not align with the diagonal line which represent normal distribution in normal probability graph. A normal probability plot of a normal distribution should look fairly straight, at least when the few large and small values are ignored.


```python
skewness = skew(train["SalePrice"])
kurtosis = train["SalePrice"].kurt()

print(f"skewness: {skewness}")
print(f"kurtosis: {kurtosis}")
```

    skewness: 1.8793604459195012
    kurtosis: 6.523066888485879


*Skewness:*
The term ‘skewness’ is used to mean the absence of symmetry from the mean of the dataset. Skewness is used to indicate the shape of the distribution of data. In a skewed distribution, the curve is extended to either left or right side. So, when the plot is extended towards the right side more, it denotes positive skewness. On the other hand, when the plot is stretched more towards the left direction, then it is called as negative skewness.

*Kurtosis:*
In statistics, kurtosis is defined as the parameter of relative sharpness of the peak of the probability distribution curve. It is used to indicate the flatness or peakedness of the frequency distribution curve and measures the tails or outliers of the distribution. Positive kurtosis represents that the distribution is more peaked than the normal distribution, whereas negative kurtosis shows that the distribution is less peaked than the normal distribution.


```python
train['SalePrice'] = np.log1p(train['SalePrice'])

#Normal Distribution of New Sales Price
mu, sigma = norm.fit(train['SalePrice'])
print("Mu : {:.2f}\nSigma : {:.2f}".format(mu,sigma))

#Visualization
sns.distplot(train['SalePrice'],fit=norm);
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\Sigma=$ {:.2f})'.format(mu,sigma)],loc = 'best')
plt.xlabel('SalePrice Distribution')
plt.ylabel('Frequency')

fig = plt.figure()
res = stats.probplot(train['SalePrice'],plot=plt)
plt.show()
```

    Mu : 12.02
    Sigma : 0.40



![png](/assets/images/Predicting-House-Prices-Using-Linear-Regression/output_37_1.png)



![png](/assets/images/Predicting-House-Prices-Using-Linear-Regression/output_37_2.png)


The skew seems now corrected and the data appears more normally distributed.


```python
train_n = train.shape[0]
test_n = test.shape[0]
y_train = train['SalePrice'].values
y_test = train['SalePrice']
all_data = pd.concat((train,test), sort=False).reset_index(drop = True)
all_data.drop(['SalePrice'], axis=1, inplace = True)
print("all_data size is : {}".format(all_data.shape))
```

    all_data size is : (2917, 79)


## Missing Data


```python
all_data.isnull().sum().sort_values(ascending=False)
```




    PoolQC           2908
    MiscFeature      2812
    Alley            2719
    Fence            2346
    FireplaceQu      1420
    LotFrontage       486
    GarageCond        159
    GarageQual        159
    GarageYrBlt       159
    GarageFinish      159
    GarageType        157
    BsmtCond           82
    BsmtExposure       82
    BsmtQual           81
    BsmtFinType2       80
    BsmtFinType1       79
    MasVnrType         24
    MasVnrArea         23
    MSZoning            4
    BsmtHalfBath        2
    Utilities           2
    Functional          2
    BsmtFullBath        2
    BsmtFinSF2          1
    BsmtFinSF1          1
    Exterior2nd         1
    BsmtUnfSF           1
    TotalBsmtSF         1
    Exterior1st         1
    SaleType            1
                     ... 
    YearRemodAdd        0
    YearBuilt           0
    SaleCondition       0
    HeatingQC           0
    ExterQual           0
    ExterCond           0
    YrSold              0
    MoSold              0
    MiscVal             0
    PoolArea            0
    ScreenPorch         0
    3SsnPorch           0
    EnclosedPorch       0
    OpenPorchSF         0
    WoodDeckSF          0
    PavedDrive          0
    Fireplaces          0
    TotRmsAbvGrd        0
    KitchenAbvGr        0
    BedroomAbvGr        0
    HalfBath            0
    FullBath            0
    GrLivArea           0
    LowQualFinSF        0
    2ndFlrSF            0
    1stFlrSF            0
    CentralAir          0
    Heating             0
    Foundation          0
    MSSubClass          0
    Length: 79, dtype: int64




```python
all_data_na_values = all_data.isnull().sum()
all_data_na_values = all_data_na_values.drop(all_data_na_values[all_data_na_values == 0].index).sort_values(ascending=False)[:30]
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na,'Missing Values' :all_data_na_values})
missing_data.head(20)
```



<div style="overflow-x:auto; text-align: center">
<style type="text/css">
table.padded-table th { padding:10px; }
</style>
<table class="table-striped padded-table">
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th>Missing Ratio</th>
      <th>Missing Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>99.691464</td>
      <td>2908</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>96.400411</td>
      <td>2812</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>93.212204</td>
      <td>2719</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>80.425094</td>
      <td>2346</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>48.680151</td>
      <td>1420</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>16.660953</td>
      <td>486</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>5.450806</td>
      <td>159</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>5.450806</td>
      <td>159</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>5.450806</td>
      <td>159</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>5.450806</td>
      <td>159</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>5.382242</td>
      <td>157</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>2.811107</td>
      <td>82</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>2.811107</td>
      <td>82</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>2.776826</td>
      <td>81</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>2.742544</td>
      <td>80</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>2.708262</td>
      <td>79</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>0.822763</td>
      <td>24</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>0.788481</td>
      <td>23</td>
    </tr>
    <tr>
      <th>MSZoning</th>
      <td>0.137127</td>
      <td>4</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>0.068564</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.subplots(figsize = (15,12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index,y=all_data_na)
plt.xlabel('Features',fontsize=15)
plt.ylabel('Percent of Missing Values', fontsize=15)
plt.title('% of Misssing data by Features', fontsize=15)
```

![png](/assets/images/Predicting-House-Prices-Using-Linear-Regression/output_43_1.png)


### Correlation between Columns


```python
#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(25,15))
sns.heatmap(corrmat, vmax=0.9, square=True, annot=True, fmt=".2f")
```

![png](/assets/images/Predicting-House-Prices-Using-Linear-Regression/output_45_1.png)


From this we can tell which features (OverallQual, GrLivArea, GarageCars, GarageArea, and TotalBsmtSF)are highly positively correlated with the SalePrice.

GarageCars and GarageArea have strongly correlation, TotalBsmtSF and 1stFlrSF have similarly high correlation. We can assum that the number of cars stored in a garage strongly depends on the area of the garage.

### Fill The Missing Data

To try and understand what the missing values are, is important to see the data documentation. This will help us to transform features to reflect the assumptions we made. For Example, GarageArea is zero, indicates we don’t have a garage and cars should be transformed to 0 as well. We'll also set mode value for missing entries and transform numerical features to categorical features. One other key feature that we'll add is the Total surface area as TotalSF by adding together TotalBsmtSF, 1stFlrSF and 2ndFlrSF (since area related features are very important to determine house prices)

We'll also log transform highly skewed features using box cox transformation which is a way to transform non-normal dependent variables into a normal shape. 

This are 59 skewed features. Normality is an important assumption for many statistical techniques. In this case the data isn’t normal, applying a Box-Cox also increases the ability to run a broader number of tests. Go on to adding of dummy variables for categorical features.


```python
# Fill the Missing Values

all_data['PoolQC'] = all_data['PoolQC'].fillna("None")
all_data['MiscFeature'] = all_data['MiscFeature'].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    
for col in ('BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1'):
    all_data[col] = all_data[col].fillna('None')

for col in ('BsmtHalfBath', 'BsmtFullBath', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1'):
    all_data[col] = all_data[col].fillna(0)

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
```

Checking any remaining missing data


```python
all_data_na_values = all_data.isnull().sum()
all_data_na_values = all_data_na_values.drop(all_data_na_values[all_data_na_values == 0].index).sort_values(ascending=False)[:30]
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na,'Missing Values' :all_data_na_values,'Data_type':all_data_na.dtype})
missing_data.head()
```




<div style="overflow-x:auto; text-align: center">
<style type="text/css">
table.padded-table th { padding:10px; }
</style>
<table class="table-striped padded-table">
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th>Missing Ratio</th>
      <th>Missing Values</th>
      <th>Data_type</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
#Transforming required numerical features to categorical 

all_data['MSSubClass']= all_data['MSSubClass'].apply(str)
all_data['OverallCond'] =all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
```


```python
#Label Encoding some categorical variables
#for information in their ordering set

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

#apply LabelEncoder to categorical features

for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))
#shape
print('Shape all_data: {}'.format(all_data.shape))
```

    Shape all_data: (2917, 78)



```python
#add total surface area as TotalSf = basement + firstflr + secondflr

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
```


```python
#log transform skewed numeric features

numeric_features = all_data.dtypes[all_data.dtypes != "object"].index

skewed_features = all_data[numeric_features].apply(lambda x : skew (x.dropna())).sort_values(ascending=False)
#compute skewness
print ("\skew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_features})   
skewness.head(7)
```

    \skew in numerical features: 
    





<div style="overflow-x:auto; text-align: center">
<style type="text/css">
table.padded-table th { padding:10px; }
</style>
<table class="table-striped padded-table">
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th>Skew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MiscVal</th>
      <td>21.939672</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>17.688664</td>
    </tr>
    <tr>
      <th>LotArea</th>
      <td>13.109495</td>
    </tr>
    <tr>
      <th>LowQualFinSF</th>
      <td>12.084539</td>
    </tr>
    <tr>
      <th>3SsnPorch</th>
      <td>11.372080</td>
    </tr>
    <tr>
      <th>LandSlope</th>
      <td>4.973254</td>
    </tr>
    <tr>
      <th>KitchenAbvGr</th>
      <td>4.300550</td>
    </tr>
  </tbody>
</table>
</div>



### Box cox transformation of highly skewed features


```python
skewness = skewness[abs(skewness) > 0.75]
print ("There are {} skewed numerical features to box cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p 
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
```

    There are 59 skewed numerical features to box cox transform


### Adding dummy categorical features



```python
all_data = pd.get_dummies(all_data)
print(all_data.shape)
```

    (2917, 220)



```python
ntrain = train.shape[0]
ntest = test.shape[0]
y_train= train.SalePrice.values
train = pd.DataFrame(all_data[:ntrain])
test = pd.DataFrame(all_data[ntrain:])
```

## Linear Regression

We'll use LASSO (least absolute shrinkage and selection operator) and Gradient boosting regression models to train the dataset and make predictions separately.

LASSO is a regression model that does variable selection and regularization. The LASSO model uses a parameter that penalizes fitting too many variables. It allows the shrinkage of variable coefficients to 0, which essentially results in those variables having no effect in the model, thereby reducing dimensionality. Since there are quite a few explanatory variables, reducing the number of variables may increase interpretability and prediction accuracy.

Gradient boosting models are one of the most popular algorithms on Kaggle. A variant of GBMs known as the XGBoost has been a clear favorite for many recent competitions. The algorithm works well right out of the box. It is a type of ensemble model, like the random forest, where multiple decision trees are used and optimized over some cost function. The popularity and ability to score well in competition are reasons enough to use this type of model for house price prediction problem.

Also, we'll use cross validation and RMSLE(Root mean squared logarithmic error) to see how well each model performs.


```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
```


```python
#validation function (Root Mean Squared Logarithmic Error)
n_folds = 5

def RMSLE_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error",
cv = kf))
    return(rmse)
```


```python
#lasso
lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005, random_state = 1))

#Gradient Boosting Regression
GBoost = GradientBoostingRegressor(loss='huber', learning_rate=0.05, n_estimators=3000,
                                   min_samples_split=10, min_samples_leaf=15,max_depth=4,
                                   random_state=5,max_features='sqrt')
```


```python
#Lasso
score = RMSLE_cv(lasso)
print ("\n Lasso score: {:.4f} ({:.4f})\n".format(score.mean(),score.std()))

#Gradient Boosting Regression
score = RMSLE_cv(GBoost)
print ("\n GBoost score: {:.4f} ({:.4f})\n".format(score.mean(),score.std()))
```

    
     Lasso score: 0.1115 (0.0074)
    
    
     GBoost score: 0.1167 (0.0084)
    


Go ahead and stack the two models together (stacking two models together, what is called ensembling, can improve the accuracy).


```python
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


    
# Averaged base models score

averaged_models = AveragingModels(models = (GBoost, lasso))

score = RMSLE_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```

     Averaged base models score: 0.1089 (0.0078)
    


As we can see, marging those two methods reduce the RMSLE.


```python
#defining RMSLE evaluation function

def RMSLE (y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
```

Finally, we'll be training and making predictions of the stacked regressor.


```python
#final training and prediction of the stacked regressor

averaged_models.fit(train.values, y_train) 
stacked_train_pred = averaged_models.predict(train.values)
stacked_pred = np.expm1(averaged_models.predict(test.values))
print("RMSLE score on the train data:") 
print(RMSLE(y_train,stacked_train_pred))
print("Accuracy score:") 
averaged_models.score(train.values, y_train)
```

    RMSLE score on the train data:
    0.06970805013738758
    Accuracy score:
    0.9695653949546431


