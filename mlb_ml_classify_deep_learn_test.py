#deep learning implementation - MLB
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import web_scrape_mlb
from os import getcwd
from os.path import join, exists 
import yaml
from tqdm import tqdm
from time import sleep
from pandas import DataFrame, concat, read_csv, isnull
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sys import argv
import joblib
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from difflib import get_close_matches
from keras.callbacks import TensorBoard, EarlyStopping
# from datetime import datetime, timedelta
# from sklearn.metrics import roc_curve
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
import os
import yaml
from collections import Counter
# from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
# Ignore the warning
warnings.filterwarnings("ignore")

"""
TODO: 1. When the model accruacy is below 30% use the opposite outcome of the models prediction

What I have learned:
1. Perform Standardization before PCA
2. Performing Standardization and PCA before rolling mean/median is worse than running it after rolling mean/median
3. mode as a way of creating future data does not work.
"""

class mlbDeep():
    def __init__(self):
        print('instantiate class mlbDeep')
        self.all_data = DataFrame()
        self.teams_abv = ["ARI","ATL","BAL","BOS","LAD","CHC","CHW","CIN","CLE","COL",
                     "DET","HOU","KCR","LAA","MIA","MIL","MIN","NYM","NYY","OAK",
                     "PHI","PIT","SDP","SFG","SEA","STL","TBR","TEX","TOR","WSN"]
        # if exists(join(getcwd(),'randomForestModelTuned.joblib')):
        #     self.RandForRegressor=joblib.load("./randomForestModelTuned.joblib")
    def get_teams(self):
        year_list_find = []
        year_list = [2017,2018,2019,2020,2021,2022,2023] #
        if exists(join(getcwd(),'year_count.yaml')):
            with open(join(getcwd(),'year_count.yaml')) as file:
                year_counts = yaml.load(file, Loader=yaml.FullLoader)
        else:
            year_counts = {'year':year_list_find}
        #Remove any years that have already been collected
        year_list_check =  year_counts['year']
        year_list_find = year_counts['year']
        year_list = [i for i in year_list if i not in year_list_check]
        print(f'Need data for year: {year_list}')
        #Collect data per year
        if year_list: 
            for year in tqdm(year_list):
                final_list = []
                self.year_store = year
                for abv in tqdm(sorted(self.teams_abv)):
                    try:
                        print() #tqdm things
                        print(f'current team: {abv}, year: {year}')
                        df_inst = web_scrape_mlb.get_data_team(abv,year)
                        final_list.append(df_inst)
                    except Exception as e:
                        print(e)
                        print(f'{abv} data are not available')
                    sleep(4) #I get get banned for a small period of time if I do not do this
                final_data = concat(final_list)
                if exists(join(getcwd(),'all_data_regressor.csv')):
                    self.all_data = read_csv(join(getcwd(),'all_data_regressor.csv'))  
                self.all_data = concat([self.all_data, final_data.dropna()])
                if not exists(join(getcwd(),'all_data_regressor.csv')):
                    self.all_data.to_csv(join(getcwd(),'all_data_regressor.csv'),index=False)
                self.all_data.to_csv(join(getcwd(),'all_data_regressor.csv'),index=False)
                year_list_find.append(year)
                print(f'year list after loop: {year_list_find}')
                with open(join(getcwd(),'year_count.yaml'), 'w') as write_file:
                    yaml.dump(year_counts, write_file)
                    print(f'writing {year} to yaml file')
        self.all_data = read_csv(join(getcwd(),'all_data_regressor.csv'))
        print('len data: ', len(self.all_data))
        self.all_data = self.all_data.drop_duplicates(keep='last')
        if 'cli' in self.all_data.columns:
            self.all_data.drop(columns='cli',inplace=True)
        print(f'length of data after duplicates are dropped: {len(self.all_data)}')
    
    def convert_to_float(self):
        for col in self.all_data.columns:
            self.all_data[col].replace('', np.nan, inplace=True)
            self.all_data[col] = self.all_data[col].astype(float)

    def pre_process(self):
        # Remove features with a correlation coef greater than 0.85
        corr_matrix = np.abs(self.x.astype(float).corr())
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] >= 0.90)]
        self.drop_cols = to_drop
        self.x_no_corr = self.x.drop(columns=to_drop)
        # No removal of correlations yet
        self.x_no_corr = self.x
        print(f'Columns that could be dropped  >= 0.90: {to_drop}')
        #Drop samples that are outliers 
        print(f'old feature dataframe shape before outlier removal: {self.x_no_corr.shape}')
        # Q1 = np.percentile(self.x_no_corr, 25, axis=0)
        # Q3 = np.percentile(self.x_no_corr, 75, axis=0)
        # IQR = Q3 - Q1
        # is_outlier = (self.x_no_corr < (Q1 - 20 * IQR)) | (self.x_no_corr > (Q3 + 20 * IQR))
        # is_outlier = is_outlier.any(axis=1)
        # not_outliers = ~is_outlier
        # self.x_no_corr = self.x_no_corr[not_outliers]
        # self.y = self.y[not_outliers]
        print(f'new feature dataframe shape after outlier removal: {self.x_no_corr.shape}')

    def split(self):
        #Remove any extraneous columns
        for col in self.all_data.columns:
            if 'Unnamed' in col:
                self.all_data.drop(columns=col,inplace=True)
        self.convert_to_float()

        #Dropna
        self.all_data.dropna(inplace=True)
        #Classification
        self.y = self.all_data['game_result'].astype(int)
        self.drop_cols_manual = ['game_result','inherited_runners','inherited_score']
        self.x = self.all_data.drop(columns=self.drop_cols_manual)

        #Regression
        self.y_regress = self.all_data['RS'].astype(int)
        self.x_regress = self.all_data.drop(columns=self.drop_cols_manual)
        self.x_regress = self.x_regress.drop(columns=['RS'])

        # self.pre_process()
        #Dropna and remove all data from subsequent y data
        # real_values = ~self.x_no_corr.isna().any(axis=1)
        # self.x_no_corr.dropna(inplace=True)
        # self.y = self.y.loc[real_values]
        #StandardScaler
        self.scaler = StandardScaler()
        self.scaler_regress = StandardScaler()
        # self.scaler = MinMaxScaler(feature_range=(0, 1))
        X_std = self.scaler.fit_transform(self.x)
        X_std_regress = self.scaler_regress.fit_transform(self.x_regress)
        #PCA data down to 95% explained variance
        self.pca = PCA(n_components=0.95)
        self.pca_regress = PCA(n_components=0.95)
        X_pca = self.pca.fit_transform(X_std)
        X_pca_regress = self.pca_regress.fit_transform(X_std_regress)

        # Check the number of components that were retained
        print('Number of components Classifier:', self.pca.n_components_)
        print('Number of components Regressor:', self.pca_regress.n_components_)
        self.x_data = DataFrame(X_pca, columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])
        self.x_data_regress = DataFrame(X_pca_regress, columns=[f'PC{i}' for i in range(1, self.pca_regress.n_components_+1)])

        #split into training and validation
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y, train_size=0.8)
        self.x_train_regress, self.x_test_regress, self.y_train_regress, self.y_test_regress = train_test_split(self.x_data_regress, self.y_regress, train_size=0.8)
   
    def deep_learn(self):
        if exists('deep_learning_mlb_class_test.h5'):
            self.model = keras.models.load_model('deep_learning_mlb_class_test.h5')
        else:
            #best params
            # Best: 0.999925 using {'alpha': 0.1, 'batch_size': 32, 'dropout_rate': 0.2,
            #  'learning_rate': 0.001, 'neurons': 16}
            optimizer = keras.optimizers.Adam(learning_rate=0.001,
                                            #   kernel_regularizer=regularizers.l2(0.001)
                                              )
            self.model = keras.Sequential([
                    layers.Dense(48, input_shape=(self.x_data.shape[1],)),
                    layers.LeakyReLU(alpha=0.2),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(44),
                    layers.LeakyReLU(alpha=0.2),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(40),
                    layers.LeakyReLU(alpha=0.2),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(36),
                    layers.LeakyReLU(alpha=0.2),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(32),
                    layers.LeakyReLU(alpha=0.2),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(28),
                    layers.LeakyReLU(alpha=0.2),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(1, activation='sigmoid')
                ])
            self.model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
            print('Training Classifier')
            self.model.summary()
            #run this to see the tensorBoard: tensorboard --logdir=./logs
            tensorboard_callback = TensorBoard(log_dir="./logs")
            early_stop = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)
            self.model.fit(self.x_train,self.y_train,epochs=500, batch_size=128, verbose=0,
                                    validation_data=(self.x_test,self.y_test),callbacks=[tensorboard_callback]) 
            self.model.save('deep_learning_mlb_class_test.h5')

    def deep_learn_regress(self):
        if exists('deep_learning_mlb_regress_test.h5'):
            print('load trained regression model')
            self.model_regress = keras.models.load_model('deep_learning_mlb_regress_test.h5')
        else:
            #best params
            # Best: 0.999925 using {'alpha': 0.1, 'batch_size': 32, 'dropout_rate': 0.2,
            #  'learning_rate': 0.001, 'neurons': 16}
            optimizer = keras.optimizers.Adam(learning_rate=0.001,
                                            #   kernel_regularizer=regularizers.l2(0.001)
                                              )
            self.model_regress = keras.Sequential([
                    layers.Dense(48, input_shape=(self.x_train_regress.shape[1],)),
                    layers.LeakyReLU(alpha=0.2),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(44),
                    layers.LeakyReLU(alpha=0.2),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(40),
                    layers.LeakyReLU(alpha=0.2),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(36),
                    layers.LeakyReLU(alpha=0.2),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(32),
                    layers.LeakyReLU(alpha=0.2),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(28),
                    layers.LeakyReLU(alpha=0.2),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(1, activation='relu')
                ])
            self.model_regress.compile(optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mse'])
            print('Training Regressor')
            self.model_regress.summary()
            #run this to see the tensorBoard: tensorboard --logdir=./logs
            tensorboard_callback = TensorBoard(log_dir="./logs")
            early_stop = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)
            self.model_regress.fit(self.x_train_regress,self.y_train_regress,epochs=500, batch_size=128, verbose=0,
                                    validation_data=(self.x_test_regress,self.y_test_regress),callbacks=[tensorboard_callback]) 
            self.model_regress.save('deep_learning_mlb_regress_test.h5')

    def predict_two_teams(self):
        while True:
            print(f'ALL TEAMS: {sorted(self.teams_abv)}')
            self.team_1 = input('team_1: ').upper()
            if self.team_1 == 'EXIT':
                break
            self.team_2 = input('team_2: ').upper()
            #Game location
            self.game_loc_team1 = int(input(f'{self.team_1} : Away = 0, Home = 1: '))
            if self.game_loc_team1 == 0:
                self.game_loc_team2 = 1
            elif self.game_loc_team1 == 1:
                self.game_loc_team2 = 0
            #2023 data
            year = 2023
            team_1_df2023 = web_scrape_mlb.get_data_team(self.team_1,year)
            sleep(4)
            team_2_df2023 = web_scrape_mlb.get_data_team(self.team_2,year)
            #Remove Game Result add game location
            team_1_df2023.drop(columns=self.drop_cols_manual,inplace=True)
            team_2_df2023.drop(columns=self.drop_cols_manual,inplace=True)
            team_1_df2023.loc[-1,'game_location'] = self.game_loc_team1
            team_2_df2023.loc[-1,'game_location'] = self.game_loc_team2
            #Drop the correlated features
            # team_1_df2023.drop(columns=self.drop_cols, inplace=True)
            # team_2_df2023.drop(columns=self.drop_cols, inplace=True)
            #convert to float
            for col in team_1_df2023.columns:
                team_1_df2023[col].replace('', np.nan, inplace=True)
                team_2_df2023[col].replace('', np.nan, inplace=True)
                team_1_df2023[col] = team_1_df2023[col].astype(float)
                team_2_df2023[col] = team_2_df2023[col].astype(float)
            team_1_df2023.dropna(inplace=True)
            team_2_df2023.dropna(inplace=True)
            #Drop RS for regression
            team_1_df2023_regress = team_1_df2023.drop(columns=['RS'])
            team_2_df2023_regress = team_2_df2023.drop(columns=['RS'])
            #PCA and standardize
            # X_std_1 = self.scaler.transform(team_1_df2023)
            # X_std_2 = self.scaler.transform(team_2_df2023) 
            X_std_1_regress = self.scaler_regress.transform(team_1_df2023_regress)
            X_std_2_regress = self.scaler_regress.transform(team_2_df2023_regress) 

            # X_pca_1 = self.pca.transform(X_std_1)
            # X_pca_2 = self.pca.transform(X_std_2)
            X_pca_1_regress = self.pca_regress.transform(X_std_1_regress)
            X_pca_2_regress = self.pca_regress.transform(X_std_2_regress)

            # team_1_df2023 = DataFrame(X_pca_1, columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])
            # team_2_df2023 = DataFrame(X_pca_2, columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])
            team_1_df2023_regress = DataFrame(X_pca_1_regress, columns=[f'PC{i}' for i in range(1, self.pca_regress.n_components_+1)])
            team_2_df2023_regress = DataFrame(X_pca_2_regress, columns=[f'PC{i}' for i in range(1, self.pca_regress.n_components_+1)])
    
            #avoid dropping column issue
            # data1_mean = DataFrame()
            # data2_mean = DataFrame()
            team_1_pred = []
            team_2_pred = []
            team_1_pred_regress = []
            team_2_pred_regress = []
            # median_bool = True
            ma_range = [3]
            #load best median rolling values
            with open('best_values_median.yaml', 'r') as file:
                best_values = yaml.safe_load(file)
            for ma in tqdm(ma_range):
                # if median_bool == True:
                team_1_df2023 = team_1_df2023.rolling(int(best_values[self.team_1])).median()
                team_2_df2023 = team_2_df2023.rolling(int(best_values[self.team_2])).median()
                team_1_df2023 = team_1_df2023.iloc[-1:]
                team_2_df2023 = team_2_df2023.iloc[-1:]

                X_std_1 = self.scaler.transform(team_1_df2023)
                X_std_2 = self.scaler.transform(team_2_df2023) 
                X_pca_1 = self.pca.transform(X_std_1)
                X_pca_2 = self.pca.transform(X_std_2)
                data1_mean = DataFrame(X_pca_1, columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])
                data2_mean = DataFrame(X_pca_2, columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])
                # print(data1_mean)
                #regress
                data1_mean_regress = team_1_df2023_regress.rolling(3).median()
                data2_mean_regress = team_2_df2023_regress.rolling(3).median()
                # else:
                #     data1_mean = team_1_df2023.ewm(span=ma,min_periods=ma-1).mean()
                #     data2_mean = team_2_df2023.ewm(span=ma,min_periods=ma-1).mean()
                #     #regress
                #     data1_mean_regress = team_1_df2023_regress.ewm(span=ma,min_periods=ma-1).mean()
                #     data2_mean_regress = team_2_df2023_regress.ewm(span=ma,min_periods=ma-1).mean()

                # for cols in team_1_df2023.columns:
                    # # ['cli', 'inherited_runners', 'inherited_score']
                    # if "cli" not in cols or "inherited_runners" not in cols or "inherited_score" not in cols:
                    #     # data1_mean[cols] = team_1_df2023[cols].ewm(span=ma,min_periods=ma-1).mean()
                    #     # data2_mean[cols] = team_2_df2023[cols].ewm(span=ma,min_periods=ma-1).mean()
                    #     data1_mean[cols] = team_1_df2023[cols].rolling(ma,min_periods=ma-1).median()
                    #     data2_mean[cols] = team_2_df2023[cols].rolling(ma,min_periods=ma-1).median()
                    # else:
                    #     data1_mean[cols] = team_1_df2023[cols]
                    #     data2_mean[cols] = team_2_df2023[cols]
                # data1_mean = team_1_df2023.dropna().rolling(ma,min_periods=ma-1).median()
                # data2_mean = team_2_df2023.dropna().rolling(ma,min_periods=ma-1).median()
                # data1_mean['game_location'] = game_loc_team1
                # data2_mean['game_location'] = game_loc_team2
                #TEAM 1 Prediction
                # x_new = self.scaler.transform(data1_mean.iloc[-1:])
                # x_new2 = self.scaler.transform(data2_mean.iloc[-1:])
                prediction = self.model.predict(data1_mean)
                prediction2 = self.model.predict(data2_mean)
                prediction_1_regress = self.model_regress.predict(data1_mean_regress.iloc[-1:])
                prediction_2_regress = self.model_regress.predict(data2_mean_regress.iloc[-1:])
                team_1_pred.append(prediction[0][0]*100)
                team_2_pred.append(prediction2[0][0]*100)
                team_1_pred_regress.append(prediction_1_regress[0][0])
                team_2_pred_regress.append(prediction_2_regress[0][0])
            self.save_outcomes_1 = team_1_pred
            self.save_outcomes_2 = team_2_pred
            # print(f'prediction {self.team_1}: {team_1_pred}%')
            # print(f'prediction {self.team_2}: {team_2_pred}%')
            print('====================================')
            print(f'rolling value for {self.team_1}: {int(best_values[self.team_1])}')
            print(f'rolling value for {self.team_2}: {int(best_values[self.team_2])}')
            print('====================================')
            print('Classifier')
            print(f'{self.team_1} prediction: {self.save_outcomes_1}')
            print(f'{self.team_2} prediction: {self.save_outcomes_2}')
            print('====================================')
            print('Regressor')
            print(f'{self.team_1} Predicted Scores: {team_1_pred_regress}')
            print(f'{self.team_2} Predicted Scores: {team_2_pred_regress}')
            print('====================================')
            if abs(sum(team_1_pred) - sum(team_2_pred)) <= 10: #arbitrary
                print('Game will be close.')
            if sum(team_1_pred) > sum(team_2_pred):
                self.team_outcome = self.team_1
                # print(f'{self.team_1} wins')
            elif sum(team_1_pred) < sum(team_2_pred):
                self.team_outcome = self.team_2
            #     print(f'{self.team_2} wins')
            # print('====================================')
            # self.predict_two_teams_running()

    def test_ma(self):
        # final_list = []
        model = keras.models.load_model('deep_learning_mlb_class_test.h5')
        model_regress = keras.models.load_model('deep_learning_mlb_regress_test.h5')
        final_df_mean = DataFrame()
        final_df_median = DataFrame()
        final_df_mean_regress = DataFrame()
        final_df_median_regress = DataFrame()
        #load current day teams
        team_names = read_csv('teams_curr_day.csv')
        collapsed_list = team_names['Team_1'].tolist() + team_names['Team_1'].tolist()
        for abv in tqdm(sorted(collapsed_list)):
            # try:
            print() #tqdm things
            print(f'current team: {abv}, year: {2023}')
            df_inst = web_scrape_mlb.get_data_team(abv,2023)
            # df_inst.drop(columns=self.drop_cols, inplace=True)
            for col in df_inst.columns:
                df_inst[col].replace('', np.nan, inplace=True)
                df_inst[col] = df_inst[col].astype(float)
            #Actual
            game_result_series = df_inst['game_result']
            game_score_series = df_inst['RS']
            df_inst.drop(columns=self.drop_cols_manual,inplace=True)
            df_inst.dropna(inplace=True)
            df_inst_regress = df_inst.drop(columns=["RS"])

            #PCA and standardize
            X_std_1 = self.scaler.transform(df_inst)
            X_std_1_regress = self.scaler_regress.transform(df_inst_regress)
            X_pca_1 = self.pca.transform(X_std_1)
            X_pca_1_regress = self.pca_regress.transform(X_std_1_regress)
            team_1_df2023 = DataFrame(X_pca_1, columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])
            team_1_df2023_regress = DataFrame(X_pca_1, columns=[f'PC{i}' for i in range(1, self.pca_regress.n_components_+1)])
            
            ma_range = np.arange(2,len(team_1_df2023)-1)
            dict_range_mean_regress = {}
            dict_range_median_regress = {}
            dict_range_mean = {}
            dict_range_median = {}
            for ma in ma_range:
                #Get rolling mean and medians
                data1_mean = team_1_df2023.iloc[0:-1].ewm(span=ma,min_periods=ma-1).mean()
                data1_mean_regress = team_1_df2023_regress.iloc[0:-1].ewm(span=ma,min_periods=ma-1).mean()
                data1_median = team_1_df2023.iloc[0:-1].rolling(ma).median()
                data1_median_regress = team_1_df2023_regress.iloc[0:-1].rolling(ma).median()
                #Predict
                prediction_mean = model.predict(data1_mean.iloc[-1:])
                prediction_median = model.predict(data1_median.iloc[-1:])
                prediction_mean_regress = model_regress.predict(data1_mean_regress.iloc[-1:])
                prediction_median_regress = model_regress.predict(data1_median_regress.iloc[-1:])
                #classification
                if prediction_mean[0][0] > 0.5:
                    result_mean = 1
                else:
                    result_mean = 0
                if prediction_median[0][0] > 0.5:
                    result_median = 1
                else:
                    result_median = 0
                if int(game_result_series.iloc[-1]) == result_mean:
                    range_mean = 1
                else:
                    range_mean = 0
                if int(game_result_series.iloc[-1]) == result_median:
                    range_median = 1
                else:
                    range_median = 0
                dict_range_mean[ma] = [range_mean]
                dict_range_median[ma] = [range_median]
                #Regression
                diff_mean_regress = abs(prediction_mean_regress[0][0] - game_score_series.iloc[-1])
                diff_median_regress = abs(prediction_median_regress[0][0] - game_score_series.iloc[-1])
                print(f'mean diff :{diff_mean_regress}')
                print(f'median diff :{diff_median_regress}')
                dict_range_mean_regress[ma] = [diff_mean_regress]
                dict_range_median_regress[ma] = [diff_median_regress]

            final_df_mean = concat([final_df_mean, DataFrame(dict_range_mean)])
            final_df_median = concat([final_df_median, DataFrame(dict_range_median)])
            final_df_mean_regress = concat([final_df_mean_regress, DataFrame(dict_range_mean_regress)])
            final_df_median_regress = concat([final_df_median_regress, DataFrame(dict_range_median_regress)])
            # print(final_df_mean)
            # print(final_df_mean.dropna(axis=1))
            sleep(3)
        
        #Regression
        final_df_mean_regress = final_df_mean_regress.dropna(axis=1)
        final_df_median_regress = final_df_median_regress.dropna(axis=1)
        column_sums_mean_regress = final_df_mean_regress.sum(axis=0)
        column_sums_median_regress = final_df_median_regress.sum(axis=0)

        sorted_columns_regress_mean = column_sums_mean_regress.sort_values(ascending=True)
        sorted_columns_regress_median = column_sums_median_regress.sort_values(ascending=True)

        print(f'Mean Regression columns: {sorted_columns_regress_mean}')
        print(f'Median Regression columns: {sorted_columns_regress_median}')
                
        final_df_mean = final_df_mean.dropna(axis=1)
        final_df_median = final_df_median.dropna(axis=1)
        column_sums_mean = final_df_mean.sum(axis=0)
        column_sums_median = final_df_median.sum(axis=0)
        proportions_mean = column_sums_mean / len(final_df_mean)
        proportions_median = column_sums_median / len(final_df_median)

        sorted_columns = column_sums_mean.sort_values(ascending=False)
        # Print the sorted columns
        print(f'mean sorted values: {sorted_columns}')

        #print each mean and median percentage correct
        print(f'mean percent correct: {(sorted_columns.iloc[0] / len(collapsed_list))*100}')

        sorted_columns = column_sums_median.sort_values(ascending=False)
        # Print the sorted columns
        print(f'median sorted values: {sorted_columns}')
        print('=========')
        print(sorted_columns.iloc[0])

        #print each mean and median percentage correct
        print(f'median percent correct: {(sorted_columns.iloc[0] / len(collapsed_list))*100}')
    
        #plot the summed values of correct 
        plt.figure()
        plt.bar(final_df_mean.columns, proportions_mean)
        plt.xlabel('Column')
        plt.ylabel('Proportion')
        plt.title('Proportions of Summed Values - Mean')
        plt.xticks(rotation=90)
        plt.savefig('best_mean_ma.png',dpi=350)
        plt.figure()
        plt.bar(final_df_median.columns, proportions_median)
        plt.xlabel('Column')
        plt.ylabel('Proportion')
        plt.title('Proportions of Summed Values - Median')
        plt.xticks(rotation=90)
        plt.savefig('best_median_ma.png',dpi=350)

        # final_list.append(df_inst)
        #     except Exception as e:
        #         print(e)
        #         print(f'{abv} data are not available')
        #     sleep(4) #I get get banned for a small period of time if I do not do this
        # final_test_data = concat(final_list)

    def test_each_team_classify(self):
        model = keras.models.load_model('deep_learning_mlb_class_test.h5')
        # model_regress = keras.models.load_model('deep_learning_mlb_regress_test.h5')
        # final_df_mean = DataFrame()
        # final_df_median = DataFrame()
        final_dict = {}
        final_dict_mean = {}
        save_betting_teams = []
        save_betting_teams_opposite = []
        
        #delete all previous .pngs
        folder_path = os.path.join(os.getcwd(),'histogram_teams')
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # Check if the file is a PNG image
            if filename.lower().endswith(".png") and os.path.isfile(file_path):
                # Delete the file
                os.remove(file_path)
        for abv in tqdm(sorted(self.teams_abv)):
            # try:
            print() #tqdm things
            print(f'current team: {abv}, year: {2023}')
            df_inst = web_scrape_mlb.get_data_team(abv,2023)
            for col in df_inst.columns:
                df_inst[col].replace('', np.nan, inplace=True)
                df_inst[col] = df_inst[col].astype(float)
            #Actual
            df_inst.dropna(inplace=True)
            game_result_series = df_inst['game_result']
            df_inst.drop(columns=self.drop_cols_manual,inplace=True)
            range_data = np.arange(2,40)
            per_team_best_rolling_vals = []
            per_team_best_rolling_vals_mean = []
            
            #iterate over every game 
            num_iter = 0
            for game in range(range_data[-1],len(df_inst)-1):
                ground_truth = game_result_series.iloc[game+1]
                dict_range_median = {}
                dict_range_mean = {}
                for roll_val in range_data:
                    data1_median = df_inst.iloc[0:game].rolling(roll_val).median()
                    data1_mean = df_inst.iloc[0:game].ewm(span=roll_val,min_periods=roll_val-1).mean()
                    X_std_1 = self.scaler.transform(data1_median.iloc[-1:])
                    X_std_1_mean = self.scaler.transform(data1_mean.iloc[-1:])
                    X_pca_1 = self.pca.transform(X_std_1)
                    X_pca_1_mean = self.pca.transform(X_std_1_mean)
                    team_1_df2023 = DataFrame(X_pca_1, columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])
                    team_1_df2023_mean = DataFrame(X_pca_1_mean, columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])
                    prediction_median = model.predict(team_1_df2023.iloc[-1:])
                    prediction_mean = model.predict(team_1_df2023_mean.iloc[-1:])
                    #median prediction
                    if prediction_median[0][0] > 0.5:
                        result_median = 1
                    else:
                        result_median = 0
                    if int(ground_truth) == result_median:
                        range_median = 1
                    else:
                        range_median = 0
                    dict_range_median[roll_val] = [range_median]
                    #mean prediction
                    if prediction_mean[0][0] > 0.5:
                        result_mean = 1
                    else:
                        result_mean = 0
                    if int(ground_truth) == result_mean:
                        range_mean = 1
                    else:
                        range_mean = 0
                    dict_range_mean[roll_val] = [range_mean]
                num_iter += 1
                #extract keys that have a value of 1 
                keys_with_value_one = [key for key, value in dict_range_median.items() if value == [1]]
                keys_with_value_one_mean = [key for key, value in dict_range_mean.items() if value == [1]]
                per_team_best_rolling_vals.append(keys_with_value_one)
                per_team_best_rolling_vals_mean.append(keys_with_value_one_mean)
            #remove empty sublists and combine into one list
            merged_list = [item for sublist in per_team_best_rolling_vals if sublist for item in sublist]
            merged_list_mean = [item for sublist in per_team_best_rolling_vals_mean if sublist for item in sublist]
            print(f'Total number of games: {num_iter}')
            #Plot median
            plt.figure(figsize=[15,5])
            plt.hist(merged_list, bins=range(min(merged_list), max(merged_list)+2), rwidth=0.8, align='left')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(f'{abv} Histogram Median. total games: {num_iter}')
            plt.xticks(range(min(merged_list), max(merged_list)+1))
            for rect in plt.gca().patches:
                x = rect.get_x() + rect.get_width() / 2
                y = rect.get_height()
                prop = round(y/num_iter,2)
                if prop >= 0.7:
                    save_betting_teams.append(abv)
                plt.gca().annotate(f'{prop}', (x, y), ha='center', va='bottom',fontsize=8)
            plt.savefig(os.path.join(os.getcwd(),'histogram_teams',f'{abv}_median_hist.png'),dpi=300)
            plt.close()

            #Plot mean
            plt.figure(figsize=[15,5])
            plt.hist(merged_list_mean, bins=range(min(merged_list_mean), max(merged_list_mean)+2), rwidth=0.8, align='left')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(f'{abv} Histogram EWM. total games: {num_iter}')
            plt.xticks(range(min(merged_list_mean), max(merged_list_mean)+1))
            for rect in plt.gca().patches:
                x = rect.get_x() + rect.get_width() / 2
                y = rect.get_height()
                plt.gca().annotate(f'{round(y/num_iter,2)}', (x, y), ha='center', va='bottom',fontsize=8)
            plt.savefig(os.path.join(os.getcwd(),'histogram_teams',f'{abv}_EWM_hist.png'),dpi=300)
            plt.close()

            #write best value to file - median
            counter = Counter(merged_list)
            most_frequent_value_median = counter.most_common(1)[0][0]
            #write best value to file - EWM
            counter = Counter(merged_list_mean)
            most_frequent_value_mean = counter.most_common(1)[0][0]
            # best_value_dict = {f'{abv}': [most_frequent_value]}

            # # Read existing data from the YAML file
            # try:
            #     final_df_median = read_csv('best_values.csv')
            # except FileNotFoundError:
            #     pass
            # # Update existing data with new data
            # final_df_median = concat([final_df_median, DataFrame(best_value_dict)])
            # # Write the updated data to the YAML file
            # final_df_median.to_csv('best_values.csv',index=False)
            final_dict[abv] = int(most_frequent_value_median)
            final_dict_mean[abv] = int(most_frequent_value_mean)
        with open('best_values_median.yaml', 'w') as file:
            yaml.dump(final_dict, file)
        with open('best_values_mean.yaml', 'w') as file:
            yaml.dump(final_dict, file)
        #Remove any duplicates from list
        save_betting_teams = list(set(save_betting_teams))
        print(f'teams that have the highest predictability: {save_betting_teams}')
        with open("betting_teams.txt", "w") as file:
            for item in save_betting_teams:
                file.write(item + "\n")
    
    def test_each_team_classify_test(self):
        model = keras.models.load_model('deep_learning_mlb_class_test.h5')
        final_dict = {}
        final_dict_mean = {}
        save_betting_teams = []
        
        #delete all previous .pngs
        folder_path = os.path.join(os.getcwd(),'histogram_teams')
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # Check if the file is a PNG image
            if filename.lower().endswith(".png") and os.path.isfile(file_path):
                # Delete the file
                os.remove(file_path)
        count_teams = 1
        for abv in tqdm(sorted(self.teams_abv)):
            # try:
            print() #tqdm things
            print(f'current team: {abv}, year: {2023}')
            df_inst = web_scrape_mlb.get_data_team(abv,2023)
            for col in df_inst.columns:
                df_inst[col].replace('', np.nan, inplace=True)
                df_inst[col] = df_inst[col].astype(float)
            #Actual
            df_inst.dropna(inplace=True)
            game_result_series = df_inst['game_result']
            df_inst.drop(columns=self.drop_cols_manual,inplace=True)
            range_data = np.arange(2,40)
            per_team_best_rolling_vals = []
            per_team_best_rolling_vals_mean = []

            #use ARIMA to estimate the next game values
            df_forecast = self.forecast_features(df_inst.iloc[:-1])
            ground_truth = game_result_series.iloc[-1]
            #predict
            prediction_median = model.predict(df_forecast)
            if prediction_median[0][0] > 0.5:
                result_median = 1
            else:
                result_median = 0
            if int(ground_truth) == result_median:
                range_median = 1
            else:
                range_median = 0
            save_betting_teams.append(range_median)
            print('=======================================')
            print(f'Prediction: {result_median} vs. Actual: {int(ground_truth)}')
            print('=======================================')
            print(f'Accuracy out of {count_teams} teams: {sum(save_betting_teams) / count_teams}')
            print('=======================================')
            count_teams += 1
            
            





        #     #predict games
        #     #iterate over every game 
        #     num_iter = 0
        #     for game in range(range_data[-1],len(df_inst)-1):
        #         ground_truth = game_result_series.iloc[game+1]
        #         dict_range_median = {}
        #         dict_range_mean = {}
        #         for roll_val in range_data:
        #             data1_median = df_inst.iloc[0:game].rolling(roll_val).median()
        #             data1_mean = df_inst.iloc[0:game].ewm(span=roll_val,min_periods=roll_val-1).mean()
        #             X_std_1 = self.scaler.transform(data1_median.iloc[-1:])
        #             X_std_1_mean = self.scaler.transform(data1_mean.iloc[-1:])
        #             X_pca_1 = self.pca.transform(X_std_1)
        #             X_pca_1_mean = self.pca.transform(X_std_1_mean)
        #             team_1_df2023 = DataFrame(X_pca_1, columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])
        #             team_1_df2023_mean = DataFrame(X_pca_1_mean, columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])
        #             prediction_median = model.predict(team_1_df2023.iloc[-1:])
        #             prediction_mean = model.predict(team_1_df2023_mean.iloc[-1:])
        #             #median prediction
        #             if prediction_median[0][0] > 0.5:
        #                 result_median = 1
        #             else:
        #                 result_median = 0
        #             if int(ground_truth) == result_median:
        #                 range_median = 1
        #             else:
        #                 range_median = 0
        #             dict_range_median[roll_val] = [range_median]
        #             #mean prediction
        #             if prediction_mean[0][0] > 0.5:
        #                 result_mean = 1
        #             else:
        #                 result_mean = 0
        #             if int(ground_truth) == result_mean:
        #                 range_mean = 1
        #             else:
        #                 range_mean = 0
        #             dict_range_mean[roll_val] = [range_mean]
        #         num_iter += 1
        #         #extract keys that have a value of 1 
        #         keys_with_value_one = [key for key, value in dict_range_median.items() if value == [1]]
        #         keys_with_value_one_mean = [key for key, value in dict_range_mean.items() if value == [1]]
        #         per_team_best_rolling_vals.append(keys_with_value_one)
        #         per_team_best_rolling_vals_mean.append(keys_with_value_one_mean)
        #     #remove empty sublists and combine into one list
        #     merged_list = [item for sublist in per_team_best_rolling_vals if sublist for item in sublist]
        #     merged_list_mean = [item for sublist in per_team_best_rolling_vals_mean if sublist for item in sublist]
        #     print(f'Total number of games: {num_iter}')
        #     #Plot median
        #     plt.figure(figsize=[15,5])
        #     plt.hist(merged_list, bins=range(min(merged_list), max(merged_list)+2), rwidth=0.8, align='left')
        #     plt.xlabel('Value')
        #     plt.ylabel('Frequency')
        #     plt.title(f'{abv} Histogram Median. total games: {num_iter}')
        #     plt.xticks(range(min(merged_list), max(merged_list)+1))
        #     for rect in plt.gca().patches:
        #         x = rect.get_x() + rect.get_width() / 2
        #         y = rect.get_height()
        #         prop = round(y/num_iter,2)
        #         if prop >= 0.7:
        #             save_betting_teams.append(abv)
        #         plt.gca().annotate(f'{prop}', (x, y), ha='center', va='bottom',fontsize=8)
        #     plt.savefig(os.path.join(os.getcwd(),'histogram_teams',f'{abv}_median_hist.png'),dpi=300)
        #     plt.close()

        #     #Plot mean
        #     plt.figure(figsize=[15,5])
        #     plt.hist(merged_list_mean, bins=range(min(merged_list_mean), max(merged_list_mean)+2), rwidth=0.8, align='left')
        #     plt.xlabel('Value')
        #     plt.ylabel('Frequency')
        #     plt.title(f'{abv} Histogram EWM. total games: {num_iter}')
        #     plt.xticks(range(min(merged_list_mean), max(merged_list_mean)+1))
        #     for rect in plt.gca().patches:
        #         x = rect.get_x() + rect.get_width() / 2
        #         y = rect.get_height()
        #         plt.gca().annotate(f'{round(y/num_iter,2)}', (x, y), ha='center', va='bottom',fontsize=8)
        #     plt.savefig(os.path.join(os.getcwd(),'histogram_teams',f'{abv}_EWM_hist.png'),dpi=300)
        #     plt.close()

        #     #write best value to file - median
        #     counter = Counter(merged_list)
        #     most_frequent_value_median = counter.most_common(1)[0][0]
        #     #write best value to file - EWM
        #     counter = Counter(merged_list_mean)
        #     most_frequent_value_mean = counter.most_common(1)[0][0]
        #     # best_value_dict = {f'{abv}': [most_frequent_value]}

        #     # # Read existing data from the YAML file
        #     # try:
        #     #     final_df_median = read_csv('best_values.csv')
        #     # except FileNotFoundError:
        #     #     pass
        #     # # Update existing data with new data
        #     # final_df_median = concat([final_df_median, DataFrame(best_value_dict)])
        #     # # Write the updated data to the YAML file
        #     # final_df_median.to_csv('best_values.csv',index=False)
        #     final_dict[abv] = int(most_frequent_value_median)
        #     final_dict_mean[abv] = int(most_frequent_value_mean)
        # with open('best_values_median.yaml', 'w') as file:
        #     yaml.dump(final_dict, file)
        # with open('best_values_mean.yaml', 'w') as file:
        #     yaml.dump(final_dict, file)
        # #Remove any duplicates from list
        # save_betting_teams = list(set(save_betting_teams))
        # print(f'teams that have the highest predictability: {save_betting_teams}')
        # with open("betting_teams.txt", "w") as file:
        #     for item in save_betting_teams:
        #         file.write(item + "\n")
    
    def forecast_features(self,game_data):
        #standardize and PCA
        X_std_1 = self.scaler.transform(game_data)
        X_pca_1 = self.pca.transform(X_std_1)
        team_1_df2023 = DataFrame(X_pca_1, columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])
        # Train a VAR model
        model = VAR(team_1_df2023)
        model_fit = model.fit()
        # print(model_fit.summary())
        # Forecast feature values for the next game
        next_game_features = model_fit.forecast(model_fit.endog, steps=1)
        next_game_features = next_game_features[0] 
        team_df_forecast = DataFrame(np.array(next_game_features).reshape(1,self.pca.n_components_), columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])

        return team_df_forecast

    def run_analysis(self):
        if argv[1] == "test":
            self.get_teams()
            self.split()
            # self.test_ma()
            # self.test_each_team_classify_test()
            self.test_each_team_classify()
        else:
            self.get_teams()
            self.split()
            self.deep_learn()
            self.deep_learn_regress()
            self.predict_two_teams()
def main():
    mlbDeep().run_analysis()
if __name__ == '__main__':
    main()