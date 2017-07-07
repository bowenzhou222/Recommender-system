# Recommender-system
For more datasets, please refer to http://jmcauley.ucsd.edu/data/amazon/<br />
The default dataset used in the script is Musical_Instruments_5.csv with the format <user, item, rating, timestamp, helpful votes, total votes><br />
Some parameters in the RS can be adjusted:<br />
alpha in line 262: for Musical_Instruments_5.csv, alpha=12 is the best setting, and for Movies_and_TV_5.csv alpha=0.07 is the best.<br />
recommendation_user_number<br />
recommendation_list_length<br />
ratio in line 243: the ratio of training data<br />

This RS measures 'hit ratio' instead of RMSE, meaning that how many consumed items are shown in the recommendation list.
