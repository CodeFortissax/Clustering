import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import pandas as pd

style.use('ggplot')

file_path = r"C:\Users\Admin\Desktop\necessities\Datasets\titanic3.xls"
df = pd.read_excel(file_path)
#print(df.head())
df.drop(['body', 'name'], 1, inplace=True)
df.apply(pd.to_numeric, errors='ignore')
df.fillna(0, inplace=True)
print(df.head())











