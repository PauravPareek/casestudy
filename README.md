https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets

https://www.kaggle.com/pavansanagapati/anomaly-credit-card-fraud-detection

https://www.kaggle.com/joparga3/in-depth-skewed-data-classif-93-recall-acc-now

https://www.kaggle.com/sharmasanthosh/exploratory-study-on-ml-algorithms

https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-mercedes/notebook


1. Basic EDA - https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
https://www.kaggle.com/joparga3/in-depth-skewed-data-classif-93-recall-acc-now --- Main

good one starts with https://www.kaggle.com/pavansanagapati/anomaly-credit-card-fraud-detection



Main : https://www.kaggle.com/c/allstate-claims-severity/kernels

1) https://www.kaggle.com/sharmasanthosh/exploratory-study-on-ml-algorithms
2) unique categorical values per category , Correlation between categorical variables https://www.kaggle.com/achalshah/allstate-feature-analysis-python
3) fscore most inportant feature https://www.kaggle.com/guyko81/just-an-easy-solution

https://www.kaggle.com/matheusfacure/semi-supervised-anomaly-detection-survey


EDA : https://www.kaggle.com/selfishgene/basic-feature-exploration
https://www.kaggle.com/yohanb/categorical-features-encoding-xgb-0-554

** https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-mercedes

Feture exploration : https://www.kaggle.com/selfishgene/advanced-feature-exploration

Categorical feature: https://www.kaggle.com/mlisovyi/9-ways-to-treat-categorical-features-updated

feature selection : https://www.kaggle.com/sz8416/6-ways-for-feature-selection




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv('Insur_Claim.csv')
df.head()

df.describe()

df.isnull().sum().max()

df.columns

print('Genuine claims are', round(df['Fraud Flag (1=Yes 0=No)'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Fraud claims are', round(df['Fraud Flag (1=Yes 0=No)'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
#imbalanced classes

sns.countplot('Fraud Flag (1=Yes 0=No)', data=df)
plt.title('Class Distributions \n (0: Genuine claims || 1: Fraud claims)', fontsize=14)

sns.countplot('Vehicle Flag (1=Motor Vehicle Involved)', data=df)
plt.title('Class Distributions \n (0: Motor Vehicle Not Involved || 1: Motor Vehicle Involved)', fontsize=14)

plot = df[['Fraud Flag (1=Yes 0=No)', 'Vehicle Flag (1=Motor Vehicle Involved)']].copy()
plot.plot.bar(rot=0)



sns.countplot(x="Fraud Flag (1=Yes 0=No)", hue="Vehicle Flag (1=Motor Vehicle Involved)", data=df)

fig, ax = plt.subplots(figsize=(15, 20))
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode='anchor')
#sns.set(font_scale=1)

sns.countplot(ax=ax, x="Body Part" ,hue="Fraud Flag (1=Yes 0=No)",data=df )
plt.show()
