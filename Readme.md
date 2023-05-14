# Movie Project #

### Data sets: ###
tmdb_5000_credits.csv and tmdb_5000_movies.csv

### Source of data files: 
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?resource=download&select=tmdb_5000_movies.csv

### Configure environment: ###
After creating and activation virtual environemnt, all needed packages need to be installed from *requirements.txt* file  
```
pip install -r requirements.txt
```

### Idea: ###
Predicion to be done for revenue based on available columns. 

### Analysis of the idea: ###
In the first attempt following columns are examined: Budget, Production Companies, Original Language, Vote average and count. Score column is added based on calculations via IMDB formula, in order to have fair judgment i.e to avoid movie with average 8.5 with fewer number of votes (ex: 3) to be better than average rating 7.9 with bigger number if votes (ex: 40)

Since language was not accepted as string value, simple solution was found to be converted into numeric representations. This part can be improved by adding group of languages and making categorisation of the groups example: English, Europian (French , Spanish, German, Italian, Dutch...), Slovakian (Polish, Rusian, Slovenian....), Aisan (Japanese, Chinese , Korean....) etc.

Null values exist in the data set for Budget, so additional data can be added in order better result to be achieved. This is also another tips for improvement

Production Companies was formated in the ipynb file, but was not used in the model. Improvement can be done so this metric (as category) is also taken in consideration based on which production is making the movie

At the end following input parameters are taken: 'budget', 'popularity','vote_average', 'vote_count', 'score', 'language_xx' in order revenue to be predicted.

### Used model: ###
Linear regression. Due to lack of time another models were not examioned, which also can be counted as improvement in order comparison of the different results to be done and to be decided which model provides better result

### Conclusion: ###

The big value for Mean Squared Error (MSE), provides info that the squared difference between the predicted revenue values and the actual revenue values is quite large. This is good reason all improvements to be implemented in short notice

### Input example: ###

Input example from user side.  
Movie with:  
*budget:* 100000000  
*popularity:*  50 (in range 0 - 876)  
*vote_average:* 7.5 (in range 0 - 10)  
*vote_count:* 1000  
*score:* 8  
*language:* english (language_en: 1)
```
input_data = [[100000000, 50.0, 7.5, 1000, 8.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
```
