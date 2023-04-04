#deep learning implementation - MLB
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
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
# from sys import argv
import joblib
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from difflib import get_close_matches
from keras.callbacks import TensorBoard, EarlyStopping
# from datetime import datetime, timedelta
# from sklearn.metrics import roc_curve
import seaborn as sns
from sklearn.decomposition import PCA


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
        year_list = [2018,2019,2020,2021,2022] #,2023
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
        print(f'Columns dropped  >= 0.90: {to_drop}')
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
        # self.delete_opp()
        for col in self.all_data.columns:
            if 'Unnamed' in col:
                self.all_data.drop(columns=col,inplace=True)
        self.convert_to_float()
        self.y = self.all_data['game_result'].astype(int)
        self.x = self.all_data.drop(columns=['game_result'])
        self.pre_process()
        #Dropna and remove all data from subsequent y data
        real_values = ~self.x_no_corr.isna().any(axis=1)
        self.x_no_corr.dropna(inplace=True)
        self.y = self.y.loc[real_values]
        #StandardScaler
        self.scaler = StandardScaler()
        X_std = self.scaler.fit_transform(self.x_no_corr)
        #PCA data down to 95% explained variance
        self.pca = PCA(n_components=0.95)
        X_pca = self.pca.fit_transform(X_std)
        # Check the number of components that were retained
        print('Number of components:', self.pca.n_components_)
        self.x_no_corr = DataFrame(X_pca, columns=[f'PC{i}' for i in range(1, self.pca.n_components_+1)])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_no_corr, self.y, train_size=0.8)
        # normalize data
        # self.scaler = StandardScaler()
        # self.x_train = self.scaler.transform(self.x_train)
        # self.x_test = self.scaler.transform(self.x_test)
    def deep_learn(self):
        if exists('deep_learning_mlb_class.h5'):
            self.model = keras.models.load_model('deep_learning_mlb_class.h5')
        else:
            #best params
            # Best: 0.999925 using {'alpha': 0.1, 'batch_size': 32, 'dropout_rate': 0.2,
            #  'learning_rate': 0.001, 'neurons': 16}
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
            self.model = keras.Sequential([
                    layers.Dense(16, input_shape=(self.x_no_corr.shape[1],)),
                    layers.LeakyReLU(alpha=0.1),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(16),
                    layers.LeakyReLU(alpha=0.1),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(16),
                    layers.LeakyReLU(alpha=0.1),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(16),
                    layers.LeakyReLU(alpha=0.1),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(16),
                    layers.LeakyReLU(alpha=0.1),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(16),
                    layers.LeakyReLU(alpha=0.1),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(1, activation='sigmoid')
                ])
            self.model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
            self.model.summary()
            #run this to see the tensorBoard: tensorboard --logdir=./logs
            tensorboard_callback = TensorBoard(log_dir="./logs")
            early_stop = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)
            self.model.fit(self.x_train,self.y_train,epochs=500, batch_size=64, verbose=0,
                                    validation_data=(self.x_test,self.y_test),callbacks=[tensorboard_callback]) 
            self.model.save('deep_learning_mlb_class.h5')
    def predict_two_teams(self):
        while True:
                print(f'ALL TEAMS: {sorted(self.teams_abv)}')
                team_1 = input('team_1: ').upper()
                if team_1 == 'EXIT':
                    break
                team_2 = input('team_2: ').upper()
                #Game location
                game_loc_team1 = int(input(f'{team_1} : #Away = 0, Home = 1: '))
                if game_loc_team1 == 0:
                    game_loc_team2 = 1
                elif game_loc_team1 == 1:
                    game_loc_team2 = 0
                #2023 data
                year = 2023
                team_1_df2023 = web_scrape_mlb.get_data_team(team_1,year)
                sleep(4)
                team_2_df2023 = web_scrape_mlb.get_data_team(team_2,year)
                #Remove Game Result
                team_1_df2023.drop(columns=['game_result'],inplace=True)
                team_2_df2023.drop(columns=['game_result'],inplace=True)
                #Drop the correlated features
                team_1_df2023.drop(columns=self.drop_cols, inplace=True)
                team_2_df2023.drop(columns=self.drop_cols, inplace=True)
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
                ma_range = np.arange(2,5,1)
                # print(team_1_df2023)
                #avoid dropping column issue
                data1_mean = DataFrame()
                data2_mean = DataFrame()
                team_1_pred = []
                team_2_pred = []
                for ma in tqdm(ma_range):
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
                print(f'prediction {team_1}: {team_1_pred}%')
                print(f'prediction {team_2}: {team_2_pred}%')
                print('====================================')
                if abs(sum(team_1_pred) - sum(team_2_pred)) <= 10: #arbitrary
                    print('Game will be close.')
                if sum(team_1_pred) > sum(team_2_pred):
                    print(f'{team_1} wins')
                elif sum(team_1_pred) < sum(team_2_pred):
                    print(f'{team_2} wins')
                print('====================================')
    def run_analysis(self):
        self.get_teams()
        self.split()
        self.deep_learn()
        self.predict_two_teams()
def main():
    mlbDeep().run_analysis()
if __name__ == '__main__':
    main()