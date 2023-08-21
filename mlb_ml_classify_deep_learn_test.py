#deep learning implementation - MLB
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM
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
# import seaborn as sns
from sys import argv
# import joblib
# from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
# from difflib import get_close_matches
from keras.callbacks import TensorBoard, EarlyStopping
# from datetime import datetime, timedelta
# from sklearn.metrics import roc_curve
import seaborn as sns
from sklearn.decomposition import FactorAnalysis#PCA
import warnings
import os
import yaml
from collections import Counter
from pickle import dump, load
from sklearn.metrics import make_scorer, mean_squared_error
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.vector_ar.var_model import VAR
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from colorama import Fore, Style
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l2, l1
from sklearn.linear_model import LinearRegression
from psutil import virtual_memory
from sys import exit
from keras.optimizers import Adam, RMSprop
import keras_tuner as kt
from kerastuner.tuners import RandomSearch
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
warnings.filterwarnings("ignore")

"""
TODO: 1. When the model accruacy is below 30% use the opposite outcome of the models prediction

What I have learned:
1. Perform Standardization before PCA
2. Performing Standardization and PCA before rolling mean/median is worse than running it after rolling mean/median
3. mode as a way of creating future data does not work.
4. new methods of forecasting the features for future games: 
VAR=50% acc, XGBoost=50% acc, RandomForest=63% acc, LinearRegression=53% acc, MLP=50% acc, KNeighestNeighbors= 57% acc
RidgeCV= 50% acc, Ridge= 50% acc, DecisionTree= 60% acc
"""
NUM_FEATURES = 45
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mape_tf(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100

def step_decay(epoch):
    initial_lr = 0.01  # Initial learning rate
    drop = 0.25         # Factor by which the learning rate will be reduced
    epochs_drop = 40   # Number of epochs after which to apply the drop

    # Calculate the new learning rate for the current epoch
    new_lr = initial_lr * (drop ** (epoch // epochs_drop))
    return new_lr

def rmse(y_true,y_pred):
    squared_diff = (y_pred - y_true) ** 2
    return np.sqrt(np.mean(squared_diff))

def check_ram_usage():
    ram_percent = virtual_memory().percent
    return ram_percent > 98

def create_model_classifier(hp):
    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])
    unit_size = hp.Int('units', min_value=5, max_value=100, step=5)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    # l1_regularizer = hp.Float('l1_regularizer', min_value=1e-6, max_value=1e-2, sampling='log')
    l2_regularizer = hp.Float('l2_regularizer', min_value=1e-6, max_value=1e-2, sampling='log')

    inputs = Input(shape=(NUM_FEATURES,))

    shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(l2_regularizer))(inputs)
    shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    # shared_hidden_layer = Dense(units, activation='relu')(inputs)
    # shared_hidden_layer = Dense(units, activation='tanh')(shared_hidden_layer)
    # shared_hidden_layer = Dense(units, activation='relu')(shared_hidden_layer)
    # shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    # shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)

    output_layers = []
    for i in range(NUM_FEATURES):
        output_layer = Dense(1, activation='tanh', name=f'target_{i+1}')(shared_hidden_layer)
        output_layers.append(output_layer)

    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)
    model = Model(inputs=inputs, outputs=output_layers)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

    return model

def create_model_regressor(hp):
    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])
    unit_size = hp.Int('units', min_value=5, max_value=100, step=5)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    # l1_regularizer = hp.Float('l1_regularizer', min_value=1e-6, max_value=1e-2, sampling='log')
    l2_regularizer = hp.Float('l2_regularizer', min_value=1e-6, max_value=1e-2, sampling='log')

    inputs = Input(shape=(NUM_FEATURES,))

    shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(l2_regularizer))(inputs)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    # shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(l2_regularizer))(shared_hidden_layer)
    # shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    # shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
    # shared_hidden_layer = Dense(units, activation='relu')(inputs)
    # shared_hidden_layer = Dense(units, activation='tanh')(shared_hidden_layer)
    # shared_hidden_layer = Dense(units, activation='relu')(shared_hidden_layer)
    # shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
    # shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)

    output_layers = []
    for i in range(NUM_FEATURES):
        output_layer = Dense(1, activation='tanh', name=f'target_{i+1}')(shared_hidden_layer)
        output_layers.append(output_layer)

    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)
    model = Model(inputs=inputs, outputs=output_layers)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

    return model

def create_sequences(x_data,sequence_length):
    x_lstm_sequences = []
    y_lstm_sequences = []

    # Loop through your data to create sequences
    for i in range(len(x_data) - sequence_length + 1):
        x_sequence = x_data.iloc[i:i+sequence_length-1]  # Take two consecutive rows for x
        y_sequence = x_data.iloc[i+sequence_length-1]    # Take the next row for y
        
        x_lstm_sequences.append(x_sequence)
        y_lstm_sequences.append(y_sequence)

    x_lstm = np.array(x_lstm_sequences)
    y_lstm = np.array(y_lstm_sequences)

    return x_lstm, y_lstm

def create_lstm_model(units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(1, NUM_FEATURES),return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dense(units=NUM_FEATURES))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

class mlbDeep():
    def __init__(self):
        print('instantiate class mlbDeep')
        self.all_data = DataFrame()
        self.teams_abv = ["ARI","ATL","BAL","BOS","LAD","CHC","CHW","CIN","CLE","COL",
                     "DET","HOU","KCR","LAA","MIA","MIL","MIN","NYM","NYY","OAK",
                     "PHI","PIT","SDP","SFG","SEA","STL","TBR","TEX","TOR","WSN"]
        # if exists(join(getcwd(),'randomForestModelTuned.joblib')):
        #     self.RandForRegressor=joblib.load("./randomForestModelTuned.joblib")
        self.num_features = 45 
    def get_teams(self):
        year_list_find = []
        year_list = np.arange(2010,2024,1).tolist()
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
        print(f'number of features: {len(self.x.columns)}')
        self.manual_components = self.num_features# len(self.x.columns) - 2 #50 features currently.  35
        self.pca = FactorAnalysis(n_components=self.manual_components)
        self.pca_regress = FactorAnalysis(n_components=self.manual_components)
        # self.pca = PCA(n_components=0.95)
        # self.pca_regress = PCA(n_components=0.95)
        X_pca = self.pca.fit_transform(X_std)
        X_pca_regress = self.pca_regress.fit_transform(X_std_regress)

        # Check the number of components that were retained
        print('Number of components Classifier:', len(self.pca.components_))
        print('Number of components Regressor:', len(self.pca_regress.components_))
        self.x_data = DataFrame(X_pca, columns=[f'PC{i}' for i in range(1, len(self.pca.components_)+1)])
        self.x_data_regress = DataFrame(X_pca_regress, columns=[f'PC{i}' for i in range(1, len(self.pca_regress.components_)+1)])

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
            early_stop = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=2)
            self.model.fit(self.x_train,self.y_train,epochs=500, batch_size=128, verbose=2,
                                    validation_data=(self.x_test,self.y_test),callbacks=[tensorboard_callback,early_stop]) 
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
            early_stop = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=2)
            self.model_regress.fit(self.x_train_regress,self.y_train_regress,epochs=500, batch_size=128, verbose=2,
                                    validation_data=(self.x_test_regress,self.y_test_regress),callbacks=[tensorboard_callback,early_stop]) 
            self.model_regress.save('deep_learning_mlb_regress_test.h5')

    def deep_learn_features(self):
        # mse_scorer = make_scorer(mean_squared_error)
        mse_scorer = make_scorer(mape, greater_is_better=False)

        #split data
        # Separate odd and even rows
        x_data = self.x_data.iloc[::2]  # Odd rows
        y_data = self.x_data.iloc[1::2]  # Even rows

        # Adjust lengths if necessary
        if len(x_data) > len(y_data):
            x_data = x_data[:-1]  # Remove last row from x_train
        elif len(y_data) > len(x_data):
            y_data = y_data[:-1]  # Remove last row from y_train
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8)

        # Separate odd and even rows REGRESSION
        x_data_regress= self.x_data_regress.iloc[::2]  # Odd rows
        y_data_regress = self.x_data_regress.iloc[1::2]  # Even rows
        # Adjust lengths if necessary
        if len(x_data_regress) > len(y_data_regress):
            x_data_regress = x_data_regress[:-1]  # Remove last row from x_train
        elif len(y_data_regress) > len(x_data_regress):
            y_data_regress = y_data_regress[:-1]  # Remove last row from y_train
        x_train_regress, x_test_regress, y_train_regress, y_test_regress = train_test_split(x_data_regress, y_data_regress, train_size=0.8)
        
        #LSTM 
        if not exists('feature_LSTM.h5'):
            seq_length = 2 
            x_lstm, y_lstm  = create_sequences(self.x_data,seq_length)
            # print(self.x_data)
            # print('====')
            # print(x_lstm)
            # print('====')
            # print(y_lstm)
            # print('====')
            # input()
            x_train_lstm, x_val_lstm, y_train_lstm, y_val_lstm = train_test_split(x_lstm, y_lstm, test_size=0.2, random_state=42)
            #Tune LSTM
            # Define parameter grid for Grid Search
            param_grid = {
                'units': [50, 100, 150, 200],
                'dropout': [0.2, 0.4, 0.1],
                # Add more hyperparameters to tune
            }
            # Create KerasRegressor model
            lstm_model = KerasRegressor(build_fn=create_lstm_model, 
                                        epochs=50, 
                                        batch_size=256, 
                                        verbose=2)

            # Use Grid Search to find best hyperparameters
            grid = GridSearchCV(estimator=lstm_model, param_grid=param_grid, cv=3)
            grid_result = grid.fit(x_train_lstm, y_train_lstm, validation_data=(x_val_lstm, y_val_lstm))      
            best_hyperparams = grid_result.best_params_
            best_model = grid_result.best_estimator_.model
            best_score = grid_result.best_score_
            print(f'Best hyperparams: {best_hyperparams}') #Best hyperparams: {'dropout': 0.4, 'units': 50}
            print(f'Lowest MSE: {best_score}')
            best_model.save('feature_LSTM.h5')
            # Define your LSTM model
            # model = Sequential()
            # model.add(LSTM(units=50, input_shape=(1, 50), return_sequences=True))  # Input shape excludes the last row in each sequence
            # model.add(LSTM(units=50,return_sequences=False))
            # model.add(Dense(units=50))  # Output layer

            # # Compile the model
            # model.compile(optimizer='adam', loss='mean_squared_error')

            # # Train the model using your formatted sequences
            # model.fit(x_train_lstm, y_train_lstm, validation_data=(x_val_lstm,y_val_lstm), epochs=100, batch_size=128, verbose=2)
            # model.save('feature_LSTM.h5')
        #linear regression
        if not exists('feature_linear_regression.pkl'):
            lin_model = LinearRegression().fit(x_train,y_train)
            y_pred = lin_model.predict(x_test)
            y_test_np = y_test.to_numpy()
            counter_err = []
            for iter,inst in enumerate(y_pred):
                mape_list = []
                for i in range(len(inst)):  
                    mape_list.append(mape(y_test_np[iter][i],inst[i]))
                counter_err.append(np.mean(mape_list))
            lin_error = np.median(counter_err)
            print(f'Linear Regression error: {lin_error}')
            # plt.hist(counter_err,bins = 50, range=(0, 1000))
            # plt.show()
            with open('feature_linear_regression.pkl', 'wb') as file:
                    dump(lin_model, file)

        #FEATURE REGRESSOR FOR CLASSIFICATION
        if exists('feature_deep_learning_mlb_regress_test.h5'):
            print('load trained feature regression model')
            self.model_feature_regress_model = keras.models.load_model('feature_deep_learning_mlb_regress_test.h5',custom_objects={'mape_tf': mape_tf})
        else:
            #best params
            #FIND BEST PARAMETERS
            tuner = RandomSearch(
                create_model_classifier,
                objective='val_loss',
                max_trials=10,
                directory='tuner_results_classifier',
                project_name='model_tuning')
            tuner.search_space_summary()
            tuner.search(x_train, y_train, validation_data=(x_test, y_test), epochs=150)

            # Get the best model and summary of the best hyperparameters
            best_model = tuner.get_best_models(num_models=1)[0]
            best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_model.summary()
            hyperparams = best_hyperparameters.values
            print(hyperparams)
            # Define the input layer In Multi-Task Learning approach
            inputs = Input(shape=(x_train.shape[1],))

            # Shared hidden layers 
            unit_size = hyperparams['units']#int(x_train.shape[1] / 2)
            dropout_rate = hyperparams['dropout_rate']
            regularize = hyperparams['l2_regularizer']
            shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(regularize))(inputs)
            shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)

            # Task-specific output layers
            output_layers = []
            for i in range(x_train.shape[1]):
                output_layer = Dense(1, activation='tanh', name=f'target_{i+1}')(shared_hidden_layer)
                output_layers.append(output_layer)

            # Create the multi-task learning model
            self.model_feature_regress_model = Model(inputs=inputs, outputs=output_layers)
            if hyperparams['optimizer'] == 'adam':
                optimizer = Adam(learning_rate=hyperparams['learning_rate'])
            else:
                optimizer = RMSprop(learning_rate=hyperparams['learning_rate'])
            # Compile the model
            self.model_feature_regress_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

            #Summary
            self.model_feature_regress_model.summary()
            # lr_scheduler = LearningRateScheduler(step_decay)
            early_stop = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
            # Train the model
            # tensorboard_callback = TensorBoard(log_dir="./logs")
            # y_train_array = y_train.values
            history = self.model_feature_regress_model.fit(x_train, 
                      y_train, 
                      epochs=500, 
                      batch_size=128, 
                      validation_data=(x_test,y_test), 
                      verbose=2,
                      callbacks=[early_stop])
            self.model_feature_regress_model.save('feature_deep_learning_mlb_regress_test.h5')
            plt.figure(figsize=(15,15))
            save_each_label_error = []
            for i in range(1,x_train.shape[1]+1):
                col_name = f"val_target_{i}_mse"
                plt.plot(history.history[col_name],color='grey',alpha=0.4) #,label=col_name
                save_each_label_error.append(history.history[col_name])
            # Convert the list of lists into a NumPy array
            data_array = np.array(save_each_label_error)
            # Calculate the mean at each sample
            mean_at_each_sample = np.median(data_array, axis=0)
            final_mse_dnn = mean_at_each_sample[-1]
            print("Final MSE:", final_mse_dnn)
            plt.plot(mean_at_each_sample,linewidth=4,color='black',label='mean of all error')
            plt.legend()
            plt.savefig('Multi_Task_learning_output.png',dpi=350)
            plt.close()

        #FEATURE REGRESSOR FOR REGRESSION
        if exists('feature_deep_learning_mlb_regress_runs.h5'):
            self.model_feature_regress_model_regress = keras.models.load_model('feature_deep_learning_mlb_regress_runs.h5')
        else:
            #best params
            #FIND BEST PARAMETERS
            tuner = RandomSearch(
                create_model_regressor,
                objective='val_loss',
                max_trials=10,
                directory='tuner_results_regressor',
                project_name='model_tuning')
            tuner.search_space_summary()
            tuner.search(x_train_regress, y_train_regress, validation_data=(x_test_regress, y_test_regress), epochs=150)

            # Get the best model and summary of the best hyperparameters
            best_model = tuner.get_best_models(num_models=1)[0]
            best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_model.summary()
            hyperparams = best_hyperparameters.values
            print(hyperparams)
            print(x_train.shape[1])
            # Define the input layer In Multi-Task Learning approach
            inputs = Input(shape=(x_train.shape[1],))

            # Shared hidden layers 
            unit_size = hyperparams['units']#int(x_train.shape[1] / 2)
            dropout_rate = hyperparams['dropout_rate']
            regularize = hyperparams['l2_regularizer']
            shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(regularize))(inputs)
            shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            # shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            # shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            # shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            # shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            # shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            # shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            # shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            # shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            # shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            # shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            # shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            # shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            # shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            # shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            # shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            # shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            # shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            # shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            # shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            # shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            # shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            # shared_hidden_layer = Dense(unit_size, activation='tanh',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            # shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            # shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)
            # shared_hidden_layer = Dense(unit_size, activation='relu',kernel_regularizer=l2(regularize))(shared_hidden_layer)
            # shared_hidden_layer = BatchNormalization()(shared_hidden_layer)
            # shared_hidden_layer = Dropout(dropout_rate)(shared_hidden_layer)

            # Task-specific output layers
            output_layers = []
            for i in range(x_train.shape[1]):
                output_layer = Dense(1, activation='tanh', name=f'target_{i+1}')(shared_hidden_layer)
                output_layers.append(output_layer)

            # Create the multi-task learning model
            self.model_feature_regress_model = Model(inputs=inputs, outputs=output_layers)
            if hyperparams['optimizer'] == 'adam':
                optimizer = Adam(learning_rate=hyperparams['learning_rate'])
            else:
                optimizer = RMSprop(learning_rate=hyperparams['learning_rate'])
            # Compile the model
            self.model_feature_regress_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

            #Summary
            self.model_feature_regress_model.summary()
            # lr_scheduler = LearningRateScheduler(step_decay)
            early_stop = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
            # Train the model
            # tensorboard_callback = TensorBoard(log_dir="./logs")
            # y_train_array = y_train.values
            history = self.model_feature_regress_model.fit(x_train, 
                      y_train, 
                      epochs=500, 
                      batch_size=128, 
                      validation_data=(x_test,y_test), 
                      verbose=2,
                      callbacks=[early_stop])
            self.model_feature_regress_model.save('feature_deep_learning_mlb_regress_runs.h5')
            plt.figure(figsize=(15,15))
            save_each_label_error = []
            for i in range(1,x_train.shape[1]+1):
                col_name = f"val_target_{i}_mse"
                plt.plot(history.history[col_name],color='grey',alpha=0.4) #,label=col_name
                save_each_label_error.append(history.history[col_name])
            # Convert the list of lists into a NumPy array
            data_array = np.array(save_each_label_error)
            # Calculate the mean at each sample
            mean_at_each_sample = np.median(data_array, axis=0)
            final_mse_dnn_regress = mean_at_each_sample[-1]
            print("Final MSE:", final_mse_dnn_regress)
            plt.plot(mean_at_each_sample,linewidth=4,color='black',label='mean of all error')
            plt.legend()
            plt.savefig('Multi_Task_learning_output_regress.png',dpi=350)
            plt.close()
        
        if not exists('feature_xgb_model.pkl'):
            param_grid = {
                    'n_estimators': [300, 400, 500],
                    'max_depth': [None, 5, 10, 20],
                    'min_child_weight': [1, 2, 4],  # Change min_samples_split to min_child_weight
                    'gamma': [0, 0.1, 0.2],  # Change min_samples_leaf to gamma
                }

            # Train the XGBoost model
            # Create the GridSearchCV object
            grid_search = GridSearchCV(estimator=xgb.XGBRegressor(), 
                                       param_grid=param_grid, 
                                       cv=3, n_jobs=5, 
                                       verbose=2,
                                       scoring=mse_scorer)

            # Fit the GridSearchCV object to the training data
            grid_search.fit(x_train, y_train,
                            eval_set=[(x_test, y_test)], 
                            early_stopping_rounds=10, verbose=True)

            # Print the best parameters and best score
            print("Best Parameters: ", grid_search.best_params_)
            print("Best Score: ", grid_search.best_score_)
            best_score_xgb = grid_search.best_score_
            with open('feature_xgb_model.pkl', 'wb') as file:
                    dump(grid_search, file)
        if not exists('feature_random_forest_model.pkl'):
            param_grid = {
                'n_estimators': [300, 400, 500],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],  # Change min_child_weight to min_samples_split
                'min_samples_leaf': [1, 2, 4],  # Change gamma to min_samples_leaf
            }

            # Train the Random Forest model
            # Create the GridSearchCV object
            grid_search = GridSearchCV(estimator=RandomForestRegressor(), 
                                    param_grid=param_grid, 
                                    cv=3, n_jobs=5, 
                                    verbose=3,
                                    scoring=mse_scorer)

            # Fit the GridSearchCV object to the training data
            grid_search.fit(x_train, y_train)

            # Print the best parameters and best score
            print("Best Parameters: ", grid_search.best_params_)
            print("Best Score: ", grid_search.best_score_)
            final_score_rf = grid_search.best_score_
            with open('feature_random_forest_model.pkl', 'wb') as file:
                    dump(grid_search, file)
            print(f'all scores MAPE: DNN - {final_mse_dnn}, xgb - {best_score_xgb}, random forest - {final_score_rf}, lin regrss - {lin_error}')
            str_out = f'all scores MAPE: DNN - {final_mse_dnn}, xgb - {best_score_xgb}, random forest - {final_score_rf}, lin regrss - {lin_error}'
            file_path = 'output_feature_regression.txt'
            # Open the file in write mode and save the data
            with open(file_path, 'w') as file:
                file.write(str_out)


    def predict_two_teams(self):
        feature_regress_model = keras.models.load_model('feature_deep_learning_mlb_regress_test.h5',custom_objects={'mape_tf': mape_tf})
        with open('feature_xgb_model.pkl', 'rb') as file:
                xgb_model = load(file)
        with open('feature_random_forest_model.pkl', 'rb') as file:
                rf_model = load(file)
        with open('feature_linear_regression.pkl', 'rb') as file:
                lin_model = load(file)
        lstm_model = keras.models.load_model('feature_LSTM.h5')
        while True:
            try:
                print(f'ALL TEAMS: {sorted(self.teams_abv)}')
                self.team_1 = input('team_1: ').upper()
                if self.team_1 == 'EXIT':
                    break
                self.team_2 = input('team_2: ').upper()
                #Game location
                # self.game_loc_team1 = int(input(f'{self.team_1} : Away = 0, Home = 1: '))
                # if self.game_loc_team1 == 0:
                #     self.game_loc_team2 = 1
                # elif self.game_loc_team1 == 1:
                #     self.game_loc_team2 = 0
                #2023 data
                year = 2023
                team_1_df2023 = web_scrape_mlb.get_data_team(self.team_1,year)
                sleep(4)
                team_2_df2023 = web_scrape_mlb.get_data_team(self.team_2,year)
                #Remove Game Result add game location
                team_1_df2023.drop(columns=self.drop_cols_manual,inplace=True)
                team_2_df2023.drop(columns=self.drop_cols_manual,inplace=True)
                # team_1_df2023.loc[team_1_df2023.index[-1],'game_location'] = self.game_loc_team1
                # team_2_df2023.loc[team_2_df2023.index[-1],'game_location'] = self.game_loc_team2
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
                team_1_df2023_regress = DataFrame(X_pca_1_regress, columns=[f'PC{i}' for i in range(1, len(self.pca_regress.components_)+1)])
                team_2_df2023_regress = DataFrame(X_pca_2_regress, columns=[f'PC{i}' for i in range(1, len(self.pca_regress.components_)+1)])
        
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
                    # team_1_df2023_roll = team_1_df2023.rolling(int(best_values[self.team_1])).median()
                    # team_2_df2023_roll = team_2_df2023.rolling(int(best_values[self.team_2])).median()
                    # team_1_df2023_roll = team_1_df2023_roll.iloc[-1:]
                    # team_2_df2023_roll = team_2_df2023_roll.iloc[-1:]

                    X_std_1 = self.scaler.transform(team_1_df2023)
                    X_std_2 = self.scaler.transform(team_2_df2023) 
                    X_pca_1 = self.pca.transform(X_std_1)
                    X_pca_2 = self.pca.transform(X_std_2)
                    data1_mean = DataFrame(X_pca_1, columns=[f'PC{i}' for i in range(1, len(self.pca.components_)+1)])
                    data2_mean = DataFrame(X_pca_2, columns=[f'PC{i}' for i in range(1, len(self.pca.components_)+1)])

                    team_1_df2023_roll = data1_mean.rolling(int(best_values[self.team_1])).median()
                    team_2_df2023_roll = data1_mean.rolling(int(best_values[self.team_2])).median()
                    data1_mean = team_1_df2023_roll.iloc[-1:]
                    data2_mean = team_2_df2023_roll.iloc[-1:]
                    # print(data1_mean)
                    #regress
                    data1_mean_regress = team_1_df2023_regress.rolling(int(best_values[self.team_1])).median()
                    data2_mean_regress = team_2_df2023_regress.rolling(int(best_values[self.team_2])).median()
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

                # Best Parameters:  {'gamma': 0.2, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 300}
                # Best Score:  0.06727402162068385
                with open('feature_xgb_model.pkl', 'rb') as file:
                    xgb_model = load(file)
                forecast_team_1, _ = self.forecast_features(team_1_df2023)
                forecast_team_2, _ = self.forecast_features(team_2_df2023)
                # prediction_team_1 = self.model.predict(xgb_model.predict(forecast_team_1))
                # prediction_team_2 = self.model.predict(xgb_model.predict(forecast_team_2))

                next_game_features_xgb_1 = xgb_model.predict(forecast_team_1.to_numpy().reshape(1, -1))
                next_game_features_xgb_2 = xgb_model.predict(forecast_team_2.to_numpy().reshape(1, -1))
                next_game_features_rf_1 = rf_model.predict(forecast_team_1.to_numpy().reshape(1, -1))
                next_game_features_rf_2 = rf_model.predict(forecast_team_2.to_numpy().reshape(1, -1))
                next_game_features_dnn_1 = feature_regress_model.predict(forecast_team_1.to_numpy().reshape(1, -1))
                next_game_features_dnn_2 = feature_regress_model.predict(forecast_team_2.to_numpy().reshape(1, -1))
                lin_features_1 = lin_model.predict(forecast_team_1.to_numpy().reshape(1, -1))
                lin_features_2 = lin_model.predict(forecast_team_2.to_numpy().reshape(1, -1))
                next_game_features_lstm_1 = lstm_model.predict(forecast_team_1.to_numpy().reshape(-1, 1, self.num_features))
                next_game_features_lstm_2 = lstm_model.predict(forecast_team_2.to_numpy().reshape(-1, 1, self.num_features))

                #reshape DNN
                dnn_list_1 = []
                for val in next_game_features_dnn_1:
                    dnn_list_1.append(val[0][0])
                #predict
                dnn_list_1 = np.array(dnn_list_1)
                dnn_list_1 = np.reshape(dnn_list_1, (1,len(dnn_list_1)))

                dnn_list_2 = []
                for val in next_game_features_dnn_2:
                    dnn_list_2.append(val[0][0])
                #predict
                dnn_list_2 = np.array(dnn_list_2)
                dnn_list_2 = np.reshape(dnn_list_2, (1,len(dnn_list_2)))
                #predict
                prediction_median_xgb_1 = self.model.predict(next_game_features_xgb_1)
                prediction_median_xgb_2 = self.model.predict(next_game_features_xgb_2)
                prediction_median_rf_1 = self.model.predict(next_game_features_rf_1)
                prediction_median_rf_2 = self.model.predict(next_game_features_rf_2)
                prediction_median_dnn_1 = self.model.predict(dnn_list_1)
                prediction_median_dnn_2 = self.model.predict(dnn_list_2)
                prediction_median_lin_1 = self.model.predict(lin_features_1)
                prediction_median_lin_2 = self.model.predict(lin_features_2)
                prediction_lstm_1 = self.model.predict(next_game_features_lstm_1)
                prediction_lstm_2 = self.model.predict(next_game_features_lstm_2)
                # next_game_features = xgb_model.predict(df_forecast_second.to_numpy().reshape(1, -1))
                # print(next_game_features)
                #predict
                # prediction_median = model.predict(next_game_features)
                #evaluate prediction
                if prediction_median_xgb_1[0][0] > 0.5:
                    result_median_xgb_1 = 1
                else:
                    result_median_xgb_1 = 0
                if prediction_median_xgb_2[0][0] > 0.5:
                    result_median_xgb_2 = 1
                else:
                    result_median_xgb_2 = 0
                if prediction_median_rf_1[0][0] > 0.5:
                    result_median_rf_1 = 1
                else:
                    result_median_rf_1 = 0
                if prediction_median_rf_2[0][0] > 0.5:
                    result_median_rf_2 = 1
                else:
                    result_median_rf_2 = 0
                if prediction_median_dnn_1[0][0] > 0.5:
                    result_median_dnn_1 = 1
                else:
                    result_median_dnn_1 = 0
                if prediction_median_dnn_2[0][0] > 0.5:
                    result_median_dnn_2 = 1
                else:
                    result_median_dnn_2 = 0
                if prediction_lstm_1[0][0] > 0.5:
                    resultl_lstm_1 = 1
                else:
                    resultl_lstm_1 = 0
                if prediction_lstm_2[0][0] > 0.5:
                    resultl_lstm_2 = 1
                else:
                    resultl_lstm_2 = 0

                # ensemble_1 = result_median_xgb_1 + result_median_rf_1 + result_median_dnn_1# + result_median_mlp + result_median_dt
                # ensemble_2 = result_median_xgb_2 + result_median_rf_2 + result_median_dnn_2
                #all models
                # if ensemble_1 >=2:
                #     result_game_1= 1
                # else:
                #     result_game_1 = 0
                # if ensemble_2 >=2:
                #     result_game_2= 1
                # else:
                #     result_game_2 = 0
                print('====================================')
                print('Classifier')
                if self.save_outcomes_1 > self.save_outcomes_2:
                    print(Fore.GREEN + Style.BRIGHT + f'{self.team_1} prediction: {self.save_outcomes_1}' + Style.RESET_ALL)
                    print(f'{self.team_2} prediction: {self.save_outcomes_2}')
                else:
                    print(f'{self.team_1} prediction: {self.save_outcomes_1}')
                    print(Fore.GREEN + Style.BRIGHT + f'{self.team_2} prediction: {self.save_outcomes_2}' + Style.RESET_ALL)
                print('====================================')
                print('Regressor')
                if team_1_pred_regress > team_2_pred_regress:
                    print(Fore.GREEN + Style.BRIGHT +  f'{self.team_1} Predicted Scores: {team_1_pred_regress}' + Style.RESET_ALL)
                    print(f'{self.team_2} Predicted Scores: {team_2_pred_regress}')
                else:
                    print(f'{self.team_1} Predicted Scores: {team_1_pred_regress}')
                    print(Fore.GREEN + Style.BRIGHT + f'{self.team_2} Predicted Scores: {team_2_pred_regress}' + Style.RESET_ALL)
                print('====================================')
                print('Classifier and Random Forest on Feature Forecasting')
                num_true_conditions_team1 = 0
                num_true_conditions_team2 = 0

                # Check conditions for team 1
                if round(prediction_median_xgb_1[0][0]*100, 2) > 0.5:
                    num_true_conditions_team1 += 1

                if round(prediction_median_rf_1[0][0]*100, 2) > 0.5:
                    num_true_conditions_team1 += 1

                if round(prediction_median_dnn_1[0][0]*100, 2) > 0.5:
                    num_true_conditions_team1 += 1

                # Check conditions for team 2
                if round(prediction_median_xgb_2[0][0]*100, 2) > 0.5:
                    num_true_conditions_team2 += 1

                if round(prediction_median_rf_2[0][0]*100, 2) > 0.5:
                    num_true_conditions_team2 += 1

                if round(prediction_median_dnn_2[0][0]*100, 2) > 0.5:
                    num_true_conditions_team2 += 1

                if num_true_conditions_team2 < 2 and num_true_conditions_team1 < 2:
                    print(Fore.RED + Style.BRIGHT +f'{self.team_1} Predicted Winning Probability: {round(prediction_median_xgb_1[0][0]*100,2)}% XGB, {round(prediction_median_rf_1[0][0]*100,2)}% RF, {round(prediction_median_dnn_1[0][0]*100,2)}% DNN' + Style.RESET_ALL)
                    print(Fore.RED + Style.BRIGHT + f'{self.team_2} Predicted Winning Probability: {round(prediction_median_xgb_2[0][0]*100,2)}% XGB, {round(prediction_median_rf_2[0][0]*100,2)}% RF, {round(prediction_median_dnn_2[0][0]*100,2)}% DNN' + Style.RESET_ALL) 
                elif num_true_conditions_team2 >= 2 and num_true_conditions_team1 >= 2:
                    print(Fore.GREEN + Style.BRIGHT +f'{self.team_1} Predicted Winning Probability: {round(prediction_median_xgb_1[0][0]*100,2)}% XGB, {round(prediction_median_rf_1[0][0]*100,2)}% RF, {round(prediction_median_dnn_1[0][0]*100,2)}% DNN' + Style.RESET_ALL)
                    print(Fore.GREEN + Style.BRIGHT + f'{self.team_2} Predicted Winning Probability: {round(prediction_median_xgb_2[0][0]*100,2)}% XGB, {round(prediction_median_rf_2[0][0]*100,2)}% RF, {round(prediction_median_dnn_2[0][0]*100,2)}% DNN' + Style.RESET_ALL) 
                elif num_true_conditions_team1 >= 2:
                    print(Fore.GREEN + Style.BRIGHT +f'{self.team_1} Predicted Winning Probability: {round(prediction_median_xgb_1[0][0]*100,2)}% XGB, {round(prediction_median_rf_1[0][0]*100,2)}% RF, {round(prediction_median_dnn_1[0][0]*100,2)}% DNN' + Style.RESET_ALL)
                    print(f'{self.team_2} Predicted Winning Probability: {round(prediction_median_xgb_2[0][0]*100,2)}% XGB, {round(prediction_median_rf_2[0][0]*100,2)}% RF, {round(prediction_median_dnn_2[0][0]*100,2)}% DNN')
                elif num_true_conditions_team2 >= 2:
                    print(f'{self.team_1} Predicted Winning Probability: {round(prediction_median_xgb_1[0][0]*100,2)}% XGB, {round(prediction_median_rf_1[0][0]*100,2)}% RF, {round(prediction_median_dnn_1[0][0]*100,2)}% DNN')
                    print(Fore.GREEN + Style.BRIGHT + f'{self.team_2} Predicted Winning Probability: {round(prediction_median_xgb_2[0][0]*100,2)}% XGB, {round(prediction_median_rf_2[0][0]*100,2)}% RF, {round(prediction_median_dnn_2[0][0]*100,2)}% DNN' + Style.RESET_ALL)
                print(f'{self.team_1} Predicted Winning Probability: {round(prediction_median_lin_1[0][0]*100,2)}% LinRegress')
                print(f'{self.team_2} Predicted Winning Probability: {round(prediction_median_lin_2[0][0]*100,2)}% LinRegress')
                print(f'{self.team_1} Predicted Winning Probability: {round(prediction_lstm_1[0][0]*100,2)}% LSTM')
                print(f'{self.team_2} Predicted Winning Probability: {round(prediction_lstm_2[0][0]*100,2)}% LSTM')
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
            except:
                print('Try again')

    def test_each_team_classify(self):
        #only include the teams that have not been included yet 
        if exists('best_values_runs.yaml'):
            with open('best_values_runs.yaml', 'r') as file:
                        final_dict_runs = yaml.safe_load(file)
            with open('best_values_mean.yaml', 'r') as file:
                        final_dict_mean = yaml.safe_load(file)
            with open('best_values_median.yaml', 'r') as file:
                        final_dict = yaml.safe_load(file)
            filtered_teams = [team for team in self.teams_abv if team not in final_dict_runs.keys()]
        else:
            final_dict = {}
            final_dict_mean = {}
            final_dict_runs = {}
            filtered_teams = self.teams_abv
        save_betting_teams = []
        
        #delete all previous .pngs
        # folder_path = os.path.join(os.getcwd(),'histogram_teams')
        # for filename in os.listdir(folder_path):
        #     file_path = os.path.join(folder_path, filename)
        #     # Check if the file is a PNG image
        #     if filename.lower().endswith(".png") and os.path.isfile(file_path):
        #         # Delete the file
        #         os.remove(file_path)

        for abv in tqdm(sorted(filtered_teams)):
            if check_ram_usage():
                print("RAM usage is above 98%")
                exit()
            model = keras.models.load_model('deep_learning_mlb_class_test.h5')
            model_regress = keras.models.load_model('deep_learning_mlb_regress_test.h5')
            print() #tqdm things
            print(f'current team: {abv}, year: {2023}')
            df_inst = web_scrape_mlb.get_data_team(abv,2023)
            for col in df_inst.columns:
                df_inst[col].replace('', np.nan, inplace=True)
                df_inst[col] = df_inst[col].astype(float)

            df_inst.dropna(inplace=True)
            #Actual Regressor
            game_result_series_runs = df_inst['RS'].astype(int)
            df_inst_runs = df_inst.drop(columns=['RS'])
            df_inst_runs.drop(columns=self.drop_cols_manual,inplace=True)
            per_team_best_rolling_vals_runs = []
            per_team_best_rolling_vals_runs_key = []

            #Actual Classifier
            game_result_series = df_inst['game_result']
            df_inst.drop(columns=self.drop_cols_manual,inplace=True)
            per_team_best_rolling_vals = []
            per_team_best_rolling_vals_mean = []
            
            #iterate over every game 
            num_iter = 0
            range_data = np.arange(2,40)
            for game in range(range_data[-1],len(df_inst)-1):
                ground_truth = game_result_series.iloc[game+1]
                ground_truth_runs = game_result_series_runs.iloc[game+1]
                dict_range_median = {}
                dict_range_mean = {}
                dict_range_runs= {}
                #transform
                X_std_1 = self.scaler.transform(df_inst)
                X_std_1_mean = self.scaler.transform(df_inst)
                X_pca_1 = self.pca.transform(X_std_1)
                X_pca_1_mean = self.pca.transform(X_std_1_mean)
                team_1_df2023 = DataFrame(X_pca_1, columns=[f'PC{i}' for i in range(1, len(self.pca.components_)+1)])
                team_1_df2023_mean = DataFrame(X_pca_1_mean, columns=[f'PC{i}' for i in range(1, len(self.pca.components_)+1)])
                for roll_val in range_data:
                    team_1_df2023 = team_1_df2023.iloc[0:game].rolling(roll_val).median()
                    team_1_df2023_mean = team_1_df2023_mean.iloc[0:game].ewm(span=roll_val,min_periods=roll_val-1).mean()
                    data_runs = df_inst_runs.iloc[0:game].rolling(roll_val).median()
                    #apply standardization and PCA - Classifier 
                    # X_std_1 = self.scaler.transform(data1_median.iloc[-1:])
                    # X_std_1_mean = self.scaler.transform(data1_mean.iloc[-1:])
                    # X_pca_1 = self.pca.transform(X_std_1)
                    # X_pca_1_mean = self.pca.transform(X_std_1_mean)
                    # team_1_df2023 = DataFrame(X_pca_1, columns=[f'PC{i}' for i in range(1, len(self.pca.components_)+1)])
                    # team_1_df2023_mean = DataFrame(X_pca_1_mean, columns=[f'PC{i}' for i in range(1, len(self.pca.components_)+1)])
                    prediction_median = model.predict(team_1_df2023.iloc[-1:])
                    prediction_mean = model.predict(team_1_df2023_mean.iloc[-1:])

                    #apply standardization and PCA - Regressor 
                    X_std_runs = self.scaler_regress.transform(data_runs.iloc[-1:])
                    X_pca_runs = self.pca_regress.transform(X_std_runs)
                    team_1_runs = DataFrame(X_pca_runs, columns=[f'PC{i}' for i in range(1, len(self.pca_regress.components_)+1)])
                    prediction_runs = model_regress.predict(team_1_runs.iloc[-1:])

                    #Regression error 
                    pts_error = rmse(ground_truth_runs,prediction_runs) #RMSE
                    dict_range_runs[roll_val] = [pts_error]

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
                    #attempt to free up memory
                    del prediction_runs
                    del prediction_median
                    del prediction_mean
                num_iter += 1
                
                #Extract key that has lowest RMSE
                min_key = min(dict_range_runs, key=dict_range_runs.get)
                min_mse_value = dict_range_runs[min_key]
                per_team_best_rolling_vals_runs_key.append(min_key)
                per_team_best_rolling_vals_runs.append(min_mse_value)

                #extract keys that have a value of 1 
                keys_with_value_one = [key for key, value in dict_range_median.items() if value == [1]]
                keys_with_value_one_mean = [key for key, value in dict_range_mean.items() if value == [1]]
                per_team_best_rolling_vals.append(keys_with_value_one)
                per_team_best_rolling_vals_mean.append(keys_with_value_one_mean)
            del model_regress
            del model
            del df_inst
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

            #Plot pts regression
            plt.figure(figsize=[15,5])
            plt.plot(per_team_best_rolling_vals_runs_key,per_team_best_rolling_vals_runs,
                     linestyle="",marker='*',markersize=5)
            # plt.hist(per_team_best_rolling_vals_mean_pts, 
            #          bins=range(min(per_team_best_rolling_vals_mean_pts_key), 
            #                     max(per_team_best_rolling_vals_mean_pts_key)+2), rwidth=0.8, align='left')
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # plt.title(f'{abv} Histogram pts regression MAPE. total games: {num_iter}')
            # plt.xticks(range(min(per_team_best_rolling_vals_mean_pts_key), max(per_team_best_rolling_vals_mean_pts_key)+1))
            # for rect in plt.gca().patches:
            #     x = rect.get_x() + rect.get_width() / 2
            #     y = rect.get_height()
            #     plt.gca().annotate(f'{round(y/num_iter,2)}', (x, y), ha='center', va='bottom',fontsize=8)
            plt.xlabel('rolling value')
            plt.ylabel('RMSE of runs')
            plt.savefig(join(getcwd(),'histogram_teams',f'{abv}_runs_hist.png'),dpi=300)
            plt.close()

            #write best value to file - median
            counter = Counter(merged_list)
            most_frequent_value_median = counter.most_common(1)[0][0]
            #write best value to file - EWM
            counter = Counter(merged_list_mean)
            most_frequent_value_mean = counter.most_common(1)[0][0]
            #write best value to file - pts regressor
            counter = Counter(per_team_best_rolling_vals_runs_key)
            most_frequent_value_runs = counter.most_common(1)[0][0]

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
            final_dict_runs[abv] = int(most_frequent_value_runs)

            with open('best_values_median.yaml', 'w') as file:
                yaml.dump(final_dict, file)
            with open('best_values_mean.yaml', 'w') as file:
                yaml.dump(final_dict_mean, file)
            with open('best_values_runs.yaml', 'w') as file:
                yaml.dump(final_dict_runs, file)
        #Remove any duplicates from list
        save_betting_teams = list(set(save_betting_teams))
        print(f'teams that have the highest predictability: {save_betting_teams}')
        with open("betting_teams.txt", "w") as file:
            for item in save_betting_teams:
                file.write(item + "\n")
    
    def test_each_team_forecast(self):
        model = keras.models.load_model('deep_learning_mlb_class_test.h5')
        feature_regress_model = keras.models.load_model('feature_deep_learning_mlb_regress_test.h5',custom_objects={'mape_tf': mape_tf})
        feature_regress_runs = keras.models.load_model('feature_deep_learning_mlb_regress_runs.h5')
        lstm_model = keras.models.load_model('feature_LSTM.h5')
        with open('feature_xgb_model.pkl', 'rb') as file:
                xgb_model = load(file)
        with open('feature_random_forest_model.pkl', 'rb') as file:
                rf_model = load(file)
        with open('feature_linear_regression.pkl', 'rb') as file:
            lin_model = load(file)
        final_dict = {}
        final_dict_mean = {}
        save_betting_teams = []
        dnn_out = []
        lin_out = 0
        rf_out = 0
        xgb_out = 0 
        count_teams = 1
        mape_total = []
        lstm_out = 0
        for abv in tqdm(sorted(self.teams_abv)):
        #     # try:
            print() #tqdm things
            print(f'current team: {abv}, year: {2023}')
            df_inst = web_scrape_mlb.get_data_team(abv,2023)
            for col in df_inst.columns:
                df_inst[col].replace('', np.nan, inplace=True)
                df_inst[col] = df_inst[col].astype(float)
            df_inst.dropna(inplace=True)
            #Actual - Regressor 
            game_result_series_runs = df_inst['RS'].astype(int)
            df_inst_runs = df_inst.drop(columns=['RS'])
            df_inst_runs.drop(columns=self.drop_cols_manual,inplace=True)
            #Actual - Classifier
            game_result_series = df_inst['game_result']
            df_inst.drop(columns=self.drop_cols_manual,inplace=True)

            #get forecast
            _, df_forecast_second = self.forecast_features(df_inst)
            ground_truth = game_result_series.iloc[-1]

            #Regression runs
            ground_truth_runs = game_result_series_runs.iloc[-1]
            _, df_forecast_second_runs = self.forecast_feature_regress(df_inst_runs)

            feature_data_regress = df_forecast_second_runs.to_numpy().reshape(1, -1)
            next_game_features_dnn_runs = feature_regress_runs.predict(feature_data_regress)
            # print(next_game_features_dnn_runs)

            dnn_list_runs = []
            for val in next_game_features_dnn_runs:
                dnn_list_runs.append(val[0][0])
            dnn_list_runs = np.array(dnn_list_runs)
            dnn_list_runs = np.reshape(dnn_list_runs, (1,len(dnn_list_runs)))
            prediction_runs = self.model_regress.predict(dnn_list_runs)
            mape_total.append(abs(ground_truth_runs - prediction_runs))
            

            next_game_features_xgb = xgb_model.predict(df_forecast_second.to_numpy().reshape(1, -1))
            next_game_features_rf = rf_model.predict(df_forecast_second.to_numpy().reshape(1, -1))
            next_game_features_dnn = feature_regress_model.predict(df_forecast_second.to_numpy().reshape(1, -1))
            next_game_features_lin = lin_model.predict(df_forecast_second.to_numpy().reshape(1, -1))
            next_game_features_lstm = lstm_model.predict(df_forecast_second.to_numpy().reshape(-1, 1, self.num_features))

            dnn_list = []
            for val in next_game_features_dnn:
                dnn_list.append(val[0][0])
            #predict
            dnn_list = np.array(dnn_list)
            dnn_list = np.reshape(dnn_list, (1,len(dnn_list)))
            prediction_median_xgb = model.predict(next_game_features_xgb)
            prediction_median_rf = model.predict(next_game_features_rf)
            prediction_median_dnn = model.predict(dnn_list)
            prediction_median_lin = model.predict(next_game_features_lin)
            prediction_lstm = model.predict(next_game_features_lstm)
            # next_game_features = xgb_model.predict(df_forecast_second.to_numpy().reshape(1, -1))
            # print(next_game_features)
            #predict
            # prediction_median = model.predict(next_game_features)
            #evaluate prediction
            if prediction_median_xgb[0][0] > 0.5:
                result_median_xgb = 1
            else:
                result_median_xgb = 0
            if prediction_median_rf[0][0] > 0.5:
                result_median_rf = 1
            else:
                result_median_rf = 0
            if prediction_median_dnn[0][0] > 0.5:
                result_median_dnn = 1
            else:
                result_median_dnn = 0
            if prediction_median_lin[0][0] > 0.5:
                result_median_lin = 1
            else:
                result_median_lin = 0
            if prediction_lstm[0][0] > 0.5:
                result_lstm = 1
            else:
                result_lstm = 0
            # if int(ground_truth) == result_median:
            #     range_median = 1
            # else:
            #     range_median = 0

            ensemble = result_median_xgb + result_median_rf + result_median_dnn# + result_median_mlp + result_median_dt
            #all models
            if ensemble >=2:
                result_game = 1
            else:
                result_game = 0
            if int(ground_truth) == result_game:
                save_betting_teams.append(1)
            else:
                save_betting_teams.append(0)
            if int(ground_truth) == result_median_dnn:
                dnn_out.append(1)
            else:
                dnn_out.append(0)
            if int(ground_truth) == result_median_lin:
                lin_out += 1
            if int(ground_truth) == result_median_rf:
                rf_out += 1
            if int(ground_truth) == result_median_xgb:
                xgb_out += 1
            if int(ground_truth) == result_lstm:
                lstm_out += 1
            print('=======================================')
            print(f'Prediction: {result_game} vs. Actual: {int(ground_truth)}')
            print('=======================================')
            print(f'Accuracy out of {count_teams} teams: {sum(save_betting_teams) / count_teams}')
            print(f'DNN Accuracy out of {count_teams} teams: {sum(dnn_out) / count_teams}')
            print(f'LinRegress Accuracy out of {count_teams} teams: {lin_out / count_teams}')
            print(f'RandomForest Accuracy out of {count_teams} teams: {rf_out / count_teams}')
            print(f'XGB Accuracy out of {count_teams} teams: {xgb_out / count_teams}')
            print(f'LSTM Accuracy out of {count_teams} teams: {lstm_out / count_teams}')
            print('=======================================')
            print(f'Points actual {ground_truth_runs} vs prediction {prediction_runs} ')
            print(f'MAPE over all games: {np.mean(mape_total)}%')
            print('=======================================')
            count_teams += 1
            del df_inst
            del df_inst_runs
    
    def forecast_features(self,game_data):
        #standardize and FA
        X_std_1 = self.scaler.transform(game_data)
        X_pca_1 = self.pca.transform(X_std_1)
        team_1_df2023 = DataFrame(X_pca_1, columns=[f'PC{i}' for i in range(1, len(self.pca.components_)+1)])
        team_df_forecast = team_1_df2023.iloc[-1:] #last game
        team_df_forecast_second = team_1_df2023.iloc[-2] #2nd to last game
        return team_df_forecast, team_df_forecast_second
    
    def forecast_feature_regress(self,game_data):
        #standardize and FA - regression
        X_std_1 = self.scaler_regress.transform(game_data)
        X_pca_1 = self.pca_regress.transform(X_std_1)
        team_1_df2023 = DataFrame(X_pca_1, columns=[f'PC{i}' for i in range(1, len(self.pca_regress.components_)+1)])#pd.DataFrame(X_pca_1, columns=[f'PC{i}' for i in range(1, self.regress_pca.n_components_+1)])
        team_df_forecast = team_1_df2023.iloc[-1:] #last game
        team_df_forecast_second = team_1_df2023.iloc[-2] #2nd to last game
        return team_df_forecast, team_df_forecast_second

    def run_analysis(self):
        if argv[1] == "test":
            self.get_teams()
            self.split()
            self.deep_learn()
            self.deep_learn_regress()
            self.deep_learn_features()
            self.test_each_team_classify()
            self.test_each_team_forecast()
        else:
            self.get_teams()
            self.split()
            self.deep_learn()
            self.deep_learn_regress()
            self.deep_learn_features()
            self.predict_two_teams()
def main():
    mlbDeep().run_analysis()
if __name__ == '__main__':
    main()