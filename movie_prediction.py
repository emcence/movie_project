import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import json
import numpy as np

# Load the movie dataset

df1=pd.read_csv("tmdb_5000_credits.csv")
df2=pd.read_csv("tmdb_5000_movies.csv")

# Merge the two data sets in df2
df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1, on='id')

# Add score based on IMDB formula
#Calculate MDB's weighted rating (wr) which is given as :-
# WR = ((v/(v + m)) * R) + ((v/(v + m)) * C)
#where,
#v is the number of votes for the movie;
#m is the minimum votes required to be listed in the chart;
#R is the average rating of the movie; And
#C is the mean vote across the whole report
#We already have v(vote_count) and R (vote_average) and C can be calculated 

C= df2['vote_average'].mean()
m = df2['vote_count'].quantile(0.9)

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

df2['score'] = df2.apply(lambda x: weighted_rating(x, m, C), axis=1)

# Extract the input features and target variable
# X = df2[['budget', 'original_language','original_title', 'popularity','production_companies','vote_average', 'vote_count', 'score']]  # Input features

# Perform one-hot encoding for the 'original_language' column
one_hot = pd.get_dummies(df2['original_language'], prefix='language')

# Concatenate the one-hot encoded columns with the original dataframe
df2 = pd.concat([df2, one_hot], axis=1)

# Extract the input features and target variable

X = df2[['budget', 'popularity','vote_average', 'vote_count', 'score', 'language_en', 'language_ja', 'language_fr','language_zh', 'language_es', 'language_de','language_hi', 'language_ru','language_ko','language_te','language_cn','language_it','language_nl','language_ta','language_sv','language_th','language_da','language_xx','language_hu','language_cs','language_pt','language_is','language_tr','language_nb','language_af','language_pl','language_he','language_ar','language_vi','language_ky','language_id','language_ro','language_fa','language_no','language_sl','language_ps','language_el' ]]  # Input features
y = df2['revenue']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Example input from the user
# Movie with 
# budget: 100000000
# popularity: 50 (in range 0 - 876)
# vote_average: 7.5 (in range 0 - 10)
# vote_count: 1000
# score: 8
# english (language_en: 1)
input_data = [[100000000, 50.0, 7.5, 1000, 8.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# Convert the input data into a dataframe
input_df = pd.DataFrame(input_data, columns=['budget', 'popularity', 'vote_average', 'vote_count', 'score', 'language_en', 'language_ja', 'language_fr','language_zh', 'language_es', 'language_de','language_hi', 'language_ru','language_ko','language_te','language_cn','language_it','language_nl','language_ta','language_sv','language_th','language_da','language_xx','language_hu','language_cs','language_pt','language_is','language_tr','language_nb','language_af','language_pl','language_he','language_ar','language_vi','language_ky','language_id','language_ro','language_fa','language_no','language_sl','language_ps','language_el' ])


# Make predictions
predictions = model.predict(input_df)

# Print the predicted revenue
print(f"Predicted Revenue: {predictions[0]}")
