import csv
import random
import math
import numpy as np



class Matrix:
    def __init__(self, training_file, test_file, alpha):
        self.training_file = training_file
        self.test_file = test_file
        self.count(alpha)
        self.factor_number = 1
        self.iteration_number = 20
        self.learning_rate = 0.01
        self.regularization = 0.15
        self.user_factor_matrix = [[random.uniform(-0.1, 0.1) for x in range(self.factor_number)] for y in range(self.user_number)]
        self.item_factor_matrix = [[random.uniform(-0.1, 0.1) for x in range(self.factor_number)] for y in range(self.item_number)]
        
        
    def count(self, alpha):
        self.user_vector = list()
        self.item_vector = list()
        self.rating_vector = list()
        self.rating_helpful_number = list()
        self.rating_time = list()
        with open(self.training_file, newline='') as csvfile:
            csv_lines = list(csv.reader(csvfile, delimiter = ','))
            for line in csv_lines:
                if not line:
                    continue
                self.rating_time.append(int(line[3]))
        min_time = min(self.rating_time)
        with open(self.training_file, newline='') as csvfile:
            csv_lines = list(csv.reader(csvfile, delimiter = ','))
            for line in csv_lines:
                if not line:
                    continue
                #print (line)
                self.user_vector.append(str(line[0]))
                self.item_vector.append(str(line[1]))
                t = int(line[3]) - min_time
                self.rating_vector.append(float(line[2])*(1+alpha*math.log(1+1/(1+math.exp(-t))*float(line[4]))))
                #self.rating_vector.append(float(line[2]))
                self.rating_helpful_number.append(int(line[4]))
        self.rating_vector = list(map(float, self.rating_vector))
        self.user_id_set = set(self.user_vector)
        self.item_id_set = set(self.item_vector)
        self.user_number = len(self.user_id_set)
        self.item_number = len(self.item_id_set)
        self.rating_number = len(self.rating_vector)
        self.global_mean = sum(self.rating_vector)/self.rating_number
        self.mapping()

    def mapping(self):
        #self.user_id_dictionary = dict.fromkeys(self.user_id_set, set(range(0, self.user_number)))
        self.user_id_dictionary = dict.fromkeys(self.user_id_set, 0)
        self.item_id_dictionary = dict.fromkeys(self.item_id_set, 0)
        self.user_id_reverse_dictionary = dict()
        self.item_id_reverse_dictionary = dict()
        
        num = 0
        for u_d in self.user_id_dictionary:
            self.user_id_dictionary[u_d] = num
            self.user_id_reverse_dictionary[num] = u_d
            num += 1
        num = 0
        for i_d in self.item_id_dictionary:
            self.item_id_dictionary[i_d] = num
            self.item_id_reverse_dictionary[num] = i_d
            num += 1
        
        self.user_vector = list(map(lambda x:[x, self.user_id_dictionary[x]], self.user_vector))
        self.item_vector = list(map(lambda x:[x, self.item_id_dictionary[x]], self.item_vector))
        

        
    def train(self, alpha):
        self.random_array = list(range(self.rating_number))
        random.shuffle(self.random_array)

        for i in range(self.iteration_number):
            for j in range(self.rating_number):
                index = self.random_array[j]
                user_index = self.user_vector[index][1]
                item_index = self.item_vector[index][1]
                rating = self.rating_vector[index]
                weight = self.rating_helpful_number[index]
                dot_product = 0.0
                for k in range(self.factor_number):
                    dot_product += self.user_factor_matrix[user_index][k] * self.item_factor_matrix[item_index][k]
                predicted_rating = dot_product + self.global_mean
                if predicted_rating > 5:
                    predicted_rating = 5
                if predicted_rating < 1:
                    predicted_rating = 1
                prediction_error = rating - predicted_rating
                for k in range(self.factor_number):
                    self.user_factor_matrix[user_index][k] += self.learning_rate * (
                        prediction_error * self.item_factor_matrix[item_index][k]
                        - self.regularization * self.user_factor_matrix[user_index][k])
                    self.item_factor_matrix[item_index][k] += self.learning_rate * (
                        prediction_error * self.user_factor_matrix[user_index][k]
                        - self.regularization * self.item_factor_matrix[item_index][k])
    def test(self):
        rmse_summation = 0.0
        mae_summation = 0.0
        rmse_mean_summation = 0.0
        mae_mean_summation = 0.0

        f = open('predicted_ratings.txt', 'w')

        with open(self.test_file, newline='') as csvfile:
            csv_lines = list(csv.reader(csvfile, delimiter = ','))
            self.predicted_rating_vector = list()
            for line in csv_lines:
                if not line:
                    continue
                if str(line[0]) in self.user_id_dictionary:
                    user_index = self.user_id_dictionary[str(line[0])]
                else:
                    self.user_id_dictionary[str(line[0])] = len(self.user_id_dictionary)
                    self.user_factor_matrix.append([random.uniform(0,0) for x in range(self.factor_number)])
                    user_index = len(self.user_factor_matrix) - 1
                    
                if str(line[1]) in self.item_id_dictionary:
                    item_index = self.item_id_dictionary[str(line[1])]
                else:
                    self.item_id_dictionary[str(line[1])] = len(self.item_id_dictionary)
                    self.item_factor_matrix.append([random.uniform(0,0) for x in range(self.factor_number)])
                    item_index = len(self.item_factor_matrix) - 1
                    
                rating = float(line[2])

                dot_product = 0.0
                for i in range(self.factor_number):
                    dot_product += self.user_factor_matrix[user_index][i] * self.item_factor_matrix[item_index][i]
                prediction = dot_product + self.global_mean

                self.predicted_rating_vector.append(prediction)
                f.write(str(line[0])+'\t'+str(line[1])+'\t'+str(prediction)+'\n')
                rmse_summation += pow(rating - prediction, 2)
                mae_summation += abs(rating - prediction)
                rmse_mean_summation += pow(rating - self.global_mean, 2)
                mae_mean_summation += abs(rating - self.global_mean)
                
            test_rating_number = len(list(csv_lines))
            self.rmse = math.sqrt(rmse_summation/float(test_rating_number))
            self.mae = mae_summation/float(test_rating_number)
            self.mae_mean = mae_mean_summation/float(test_rating_number)
            self.rmse_mean = math.sqrt(rmse_mean_summation/float(test_rating_number))

        f.close()

    ##generate recommendation for one user
    def generate_recommendation(self, user, recommendation_list_length):
        recommendation_one_user = list()
        for i in range(self.item_number):
            dot_product = 0.0
            for j in range(self.factor_number):
                dot_product += self.user_factor_matrix[user][j] * self.item_factor_matrix[i][j]
            prediction = dot_product + self.global_mean

            prediction_struct = list([self.user_id_reverse_dictionary[user], self.item_id_reverse_dictionary[i], prediction])
            recommendation_one_user.append(prediction_struct)
        recommendation_one_user = sorted(recommendation_one_user, key = lambda x: x[2], reverse=True)
        recommendation_one_user = recommendation_one_user[0:recommendation_list_length]
        self.recommendation_list.append(recommendation_one_user)

    def generate_recommendation_random(self, user, recommendation_list_length):
        recommendation_one_user = list()
        for i in range(self.item_number):
            prediction = random.uniform(0, 5)
            prediction_struct = list([self.user_id_reverse_dictionary[user], self.item_id_reverse_dictionary[i], prediction])
            recommendation_one_user.append(prediction_struct)
        recommendation_one_user = sorted(recommendation_one_user, key = lambda x: x[2], reverse=True)
        recommendation_one_user = recommendation_one_user[0:recommendation_list_length]
        self.recommendation_list.append(recommendation_one_user)
    
        
    def hit_ratio(self, recommendation_user_number):
        csv_file = open(self.test_file , 'r')
        test_recommendation_list = list(csv.reader(csv_file, delimiter = ','))
        csv_file.close()
        number_in_recommendation_list = 0
        total_consumed_item_number = 0
        for i in range(recommendation_user_number):
            recommendation_list = self.recommendation_list[i]#list for a specific user
            #list of cosumed product of this specific user
            consumed_list = [x for x in test_recommendation_list if x[0] == recommendation_list[0][0]]
            consumed_item_number = len(consumed_list)
            total_consumed_item_number += consumed_item_number
            for j in range(consumed_item_number):
                consumed_item_id = consumed_list[j][1]
                check_list = list(map(lambda x: consumed_item_id in x, recommendation_list))
                if True in check_list:
                    number_in_recommendation_list += 1
##                if consumed_item_id in recommendation_list:
##                    number_in_recommendation_list += 1
        print ("number_in_recommendation_list: {}".format(number_in_recommendation_list))
        print ("total_consumed_item_number: {}".format(total_consumed_item_number))
        print ('hit ratio: {}'.format(float(number_in_recommendation_list/total_consumed_item_number)))

           
    def ranking(self, recommendation_user_number):
        csv_file = open(self.test_file , 'r')
        test_recommendation_list = list(csv.reader(csv_file, delimiter = ','))
        csv_file.close()
        number_in_recommendation_list = 0
        total_consumed_item_number = 0
        rank = list()
        for i in range(recommendation_user_number):
            recommendation_list = self.recommendation_list[i]#list for a specific user
            #list of cosumed product of this specific user
            consumed_list = [x for x in test_recommendation_list if x[0] == recommendation_list[0][0]]
            consumed_item_number = len(consumed_list)
            total_consumed_item_number += consumed_item_number
            for j in range(consumed_item_number):
                consumed_item_id = consumed_list[j][1]
                check_list = list(map(lambda x: consumed_item_id in x, recommendation_list))
                if True in check_list:
                    rank.append(int(check_list.index(True)))
        print (sum(rank)/len(rank))

    
    
delimiter = ','
training_file_name = 'Movies_and_TV_5_training.csv'
test_file_name = 'Movies_and_TV_5_test.csv'
raw_data_file_name = 'Movies_and_TV_5.csv'
#############################
###split dataset into test/training = ratio
def split():
    training_file = open(training_file_name, 'wt')
    test_file = open(test_file_name, 'wt')
    train_csv = csv.writer(training_file)
    test_csv = csv.writer(test_file)
    with open(raw_data_file_name, newline='') as csvfile:
            csv_lines = list(csv.reader(csvfile, delimiter = delimiter))
            line_number = len(list(csv_lines))
            random_list = list(range(line_number))
            random.shuffle(random_list)
            ratio = 0.2
            print ('test ratio: '+str(ratio))
            test_line_number = ratio * line_number
            for i in range(line_number):
                index = random_list[i]
                line = csv_lines[index]
                if i < test_line_number:
                    test_file.write(line[0]+','+line[1]+','+line[2]+','+line[3]+','+line[4]+'\n')
                else:
                    training_file.write(line[0]+','+line[1]+','+line[2]+','+line[3]+','+line[4]+'\n')

    training_file.close()
    test_file.close()
############################

split()

recommendation_user_number = 5000

m = Matrix(training_file_name, test_file_name, alpha = 0)
m.recommendation_list = list()
for i in range(recommendation_user_number):
    m.generate_recommendation_random(user = i, recommendation_list_length = 100)
m.hit_ratio(recommendation_user_number = recommendation_user_number)
del m

for i in np.arange(0, 2, 0.01):
    m = Matrix(training_file_name, test_file_name, alpha = i)
    print ("{}\t{}\t{}\t{}".format(m.user_number,
                                   m.item_number, m.rating_number, m.global_mean))

    m.train(alpha = i)

    m.recommendation_list = list()
    for i in range(recommendation_user_number):
        m.generate_recommendation(user = i, recommendation_list_length = 100)

    m.hit_ratio(recommendation_user_number = recommendation_user_number)
    del m




















