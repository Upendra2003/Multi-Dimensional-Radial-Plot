import pandas as pd

df = pd.read_csv('uploads/house_price.csv')

features = ['floors', 'bedrooms', 'area_sqrt', 'city']
output = 'price_in_lakhs'


