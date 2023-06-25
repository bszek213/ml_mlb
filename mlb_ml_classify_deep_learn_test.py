#deep learning implementation - MLB
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
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

# Ignore the warning
warnings.filterwarnings("ignore")

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

        #Classification
        self.y = self.all_data['game_result'].astype(int)
        self.drop_cols_manual = ['game_result','inherited_runners','inherited_score']
        self.x = self.all_data.drop(columns=self.drop_cols_manual)

        #Regression
        self.y_regress = self.all_data['RS'].astype(int)
        self.x_regress = self.all_data.drop(columns=self.drop_cols_manual)

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
        print('Number of components Regresso:', self.pca_regress.n_components_)
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
            self.model_regress = keras.models.load_model('deep_learning_mlb_regress_test.h5')
        else:
            #best params
            # Best: 0.999925 using {'alpha': 0.1, 'batch_size': 32, 'dropout_rate': 0.2,
            #  'learning_rate': 0.001, 'neurons': 16}
            optimizer = keras.optimizers.Adam(learning_rate=0.001,
                                            #   kernel_regularizer=regularizers.l2(0.001)
                                              )
            self.model_regress = keras.Sequential([
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
            self.game_loc_team1 = int(input(f'{self.team_1} : #Away = 0, Home = 1: '))
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
            #PCA and standardize
            X_std_1 = self.scaler.transform(team_1_df2023)
            X_std_2 = self.scaler.transform(team_2_df2023) 
            X_pca_1 = self.pca.transform(X_std_1)
            X_pca_2 = self.pca.transform(X_std_2)
            team_1_df2023 = DataFrame(X_pca_1, columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])
            team_2_df2023 = DataFrame(X_pca_2, columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])
            ma_range = [15]
            # print(team_1_df2023)
            #avoid dropping column issue
            data1_mean = DataFrame()
            data2_mean = DataFrame()
            team_1_pred = []
            team_2_pred = []
            median_bool = True
            for ma in tqdm(ma_range):
                if median_bool == True:
                    data1_mean = team_1_df2023.rolling(ma).median()
                    data2_mean = team_2_df2023.rolling(ma).median()
                else:
                    data1_mean = team_1_df2023.ewm(span=ma,min_periods=ma-1).mean()
                    data2_mean = team_2_df2023.ewm(span=ma,min_periods=ma-1).mean()

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
                prediction = self.model.predict(data1_mean.iloc[-1:])
                prediction2 = self.model.predict(data2_mean.iloc[-1:])
                team_1_pred.append(prediction[0][0]*100)
                team_2_pred.append(prediction2[0][0]*100)
            self.save_outcomes_1 = team_1_pred
            self.save_outcomes_2 = team_2_pred
            # print(f'prediction {self.team_1}: {team_1_pred}%')
            # print(f'prediction {self.team_2}: {team_2_pred}%')
            print('====================================')
            print(f'{self.team_1} prediction: {self.save_outcomes_1}')
            print(f'{self.team_2} prediction: {self.save_outcomes_2}')
            print('====================================')
            if abs(sum(team_1_pred) - sum(team_2_pred)) <= 10: #arbitrary
                print('Game will be close.')
            if sum(team_1_pred) > sum(team_2_pred):
                self.team_outcome = self.team_1
                # print(f'{self.team_1} wins')
            elif sum(team_1_pred) < sum(team_2_pred):
                self.team_outcome = self.team_2
            #     print(f'{self.team_2} wins')
            print('====================================')
            # self.predict_two_teams_running()

    def test_ma(self):
        final_list = []
        model = keras.models.load_model('deep_learning_mlb_class_test.h5')
        final_df_mean = DataFrame()
        final_df_median= DataFrame()
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
            game_result_series = df_inst['game_result']
            df_inst.drop(columns=self.drop_cols_manual,inplace=True)
            df_inst.dropna(inplace=True)
            #PCA and standardize
            X_std_1 = self.scaler.transform(df_inst)
            X_pca_1 = self.pca.transform(X_std_1)
            team_1_df2023 = DataFrame(X_pca_1, columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])
            
            ma_range = np.arange(2,len(team_1_df2023)-1)
            # range_mean = []
            # range_median = []
            dict_range_mean = {}
            dict_range_median = {}
            for ma in tqdm(ma_range):
                data1_mean = team_1_df2023.iloc[0:-1].ewm(span=ma,min_periods=ma-1).mean()
                data1_median = team_1_df2023.iloc[0:-1].rolling(ma).median()
                prediction_mean = model.predict(data1_mean.iloc[-1:])
                prediction_median = model.predict(data1_median.iloc[-1:])
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
            final_df_mean = concat([final_df_mean, DataFrame(dict_range_mean)])
            final_df_median = concat([final_df_median, DataFrame(dict_range_median)])
            # print(final_df_mean)
            # print(final_df_mean.dropna(axis=1))
            sleep(3)
        
                
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


    def run_analysis(self):
        if argv[1] == "test":
            self.get_teams()
            self.split()
            self.test_ma()
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