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
# from datetime import datetime, timedelta
# from sklearn.metrics import roc_curve
import seaborn as sns

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
        year_list = [2018,2019,2020,2021,2022,2023]
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
        print(f'length of data after duplicates are dropped: {len(self.all_data)}')
    
    def convert_to_float(self):
        for col in self.all_data.columns:
            self.all_data[col].replace('', np.nan, inplace=True)
            self.all_data[col] = self.all_data[col].astype(float)
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
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_no_corr, self.y, train_size=0.8)
        # normalize data
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

    def run_analysis(self):
        self.get_teams()
def main():
    mlbDeep().run_analysis()
if __name__ == '__main__':
    main()