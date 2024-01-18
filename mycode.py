import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

first_csv_path = 'first.csv'  
first_data = pd.read_csv(first_csv_path)
print("First CSV File:")
print(first_data.head())

third_csv_path = 'third.csv' 
third_data = pd.read_csv(third_csv_path)
print("Third CSV File:")
print(third_data.head())




# --------------------------------------


first_csv_path = 'first.csv' 
first_data = pd.read_csv(first_csv_path)
relevant_columns = ['Country Name', '1990 [YR1990]', '2000 [YR2000]', 
                    '2013 [YR2013]', '2014 [YR2014]', '2015 [YR2015]', 
                    '2016 [YR2016]', '2017 [YR2017]', '2018 [YR2018]', 
                    '2019 [YR2019]', '2020 [YR2020]', '2021 [YR2021]', 
                    '2022 [YR2022]']
first_data = first_data[relevant_columns]
first_data.replace('..', pd.NA, inplace=True)
first_data.dropna(inplace=True)
numerical_columns = ['1990 [YR1990]', '2000 [YR2000]', '2013 [YR2013]', 
                     '2014 [YR2014]', '2015 [YR2015]', '2016 [YR2016]', 
                     '2017 [YR2017]', '2018 [YR2018]', '2019 [YR2019]', 
                     '2020 [YR2020]', '2021 [YR2021]', '2022 [YR2022]']
data_for_clustering = first_data[numerical_columns]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_clustering)
kmeans = KMeans(n_clusters=3)  
kmeans.fit(scaled_data)
centroids = kmeans.cluster_centers_
feature_names = numerical_columns
plt.figure(figsize=(10, 6))
for i, centroid in enumerate(centroids):
    plt.bar(feature_names, centroid, alpha=0.7, label=f'Cluster {i+1}')

plt.xlabel('Feature Names')
plt.ylabel('Centroid Values')
plt.title('Centroid Values for Clusters')
plt.legend()
plt.xticks(rotation=45) 
plt.tight_layout()
plt.show()



# -----------------------------------------------------

relevant_columns_third = ['Country Code', 'Region', 'IncomeGroup']
third_data = third_data[relevant_columns_third]
first_data.replace('..', pd.NA, inplace=True)
third_data.replace('..', pd.NA, inplace=True)
first_data.dropna(inplace=True)
third_data.dropna(inplace=True)
numerical_columns_first = ['1990 [YR1990]', '2000 [YR2000]', '2013 [YR2013]', 
                           '2014 [YR2014]', '2015 [YR2015]', '2016 [YR2016]', 
                           '2017 [YR2017]', '2018 [YR2018]', '2019 [YR2019]', 
                           '2020 [YR2020]', '2021 [YR2021]', '2022 [YR2022]']

numerical_columns_third = ['Region', 'IncomeGroup']
data_for_clustering_first = first_data[numerical_columns_first]
data_for_clustering_third = third_data[numerical_columns_third]


selected_features_first = ['1990 [YR1990]', '2000 [YR2000]', '2015 [YR2015]']
plt.figure(figsize=(12, 4))
for i, feature in enumerate(selected_features_first):
    plt.subplot(1, 3, i+1)
    plt.hist(data_for_clustering_first[feature], bins=20, alpha=0.7)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {feature} in First Dataset')
plt.tight_layout()
plt.show()
selected_features_third = ['Region', 'IncomeGroup']
plt.figure(figsize=(10, 6))
for i, feature in enumerate(selected_features_third):
    plt.subplot(2, 1, i+1)
    feature_counts = data_for_clustering_third[feature].value_counts()
    feature_counts.plot(kind='bar', alpha=0.7)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(f'Bar Plot of {feature} in Third Dataset')
plt.tight_layout()
plt.show()


# ----------------------------------------------------------

relevant_columns = ['1990 [YR1990]', '2013 [YR2013]']  
first_data = first_data[relevant_columns]
first_data['1990 [YR1990]'] = pd.to_numeric(first_data['1990 [YR1990]'], 
                                            errors='coerce')
first_data['2013 [YR2013]'] = pd.to_numeric(first_data['2013 [YR2013]'], 
                                            errors='coerce')
first_data = first_data.dropna()
x_values = first_data['1990 [YR1990]']
y_values = first_data['2013 [YR2013]']
def polynomial_func(x, a, b, c):
    return a * x**2 + b * x + c

popt, pcov = curve_fit(polynomial_func, x_values, y_values)
x_curve = np.linspace(min(x_values), max(x_values), 100)
y_curve = polynomial_func(x_curve, *popt)
plt.step(x_values, y_values, label='Original Data', where='mid')
plt.plot(x_curve, y_curve, color='red', label='Fitted Curve')
plt.xlabel('1990 [YR1990]')
plt.ylabel('2013 [YR2013]')
plt.title('Fitting a Polynomial Curve')
plt.legend()
plt.show()

