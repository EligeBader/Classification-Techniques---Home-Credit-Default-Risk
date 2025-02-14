# %%
import pandas as pd
import numpy as np
import pickle
import dill
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder, OneHotEncoder
import imblearn
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
import keras 

# %%


def load_data(file):
    df = pd.read_parquet(file, engine='pyarrow')

    return df

with open('read_file.pickle', 'wb') as f:
    dill.dump(load_data, f)





def drop_features(df, features_to_drop=[]):
    df = df.drop(columns=features_to_drop)

    return df

with open('drop_features.pickle', 'wb') as f:
    dill.dump(drop_features, f)




# %%
def split_data(df, target, feature_selected= None, features_dropped =[], balanced_data=True):

    if balanced_data == True:
        if feature_selected == None:
            X = df.drop(columns= [target] + features_dropped)
            y = df[target]

        else:
            X = df[feature_selected]
            y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test

    else:
        if feature_selected == None:
            X = df.drop(columns= [target] + features_dropped)
            y = df[target]

        else:
            X = df[feature_selected]
            y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rus = imblearn.under_sampling.RandomUnderSampler()
        xtrain_rus, ytrain_rus = rus.fit_resample(X_train, y_train)

        return xtrain_rus, X_test, ytrain_rus, y_test


with open('split_data.pickle', 'wb') as f:
    dill.dump(split_data, f)




# %%
def clean_data(df):
    #Use SimpleImputer

    #Check Columns having nulls
    nulls_col = df.columns[df.isnull().sum() > 0]
    nulls_col  = list(nulls_col)


    # Separate numeric and categorical features
    numeric_features = [feat for feat in nulls_col if df[feat].dtype.kind in 'bifc']
    categorical_features = [feat for feat in nulls_col if feat not in numeric_features]

    # Impute missing values for numeric features
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])

    # Impute missing values for categorical features    
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])


    return df

with open('clean_data.pickle', 'wb') as f:
    dill.dump(clean_data, f)




# %%
def encode_data(df, target, categorical_cols, train, model):
    file_name = 'trained_data.pickle'
    if not train: 
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                mod = dill.load(f)

             # Transform the categorical columns 
            tempvar = mod.transform(df[categorical_cols])
            print(type(df))
            print(type(tempvar))

            if model == TargetEncoder or model == OrdinalEncoder:
                # Update the original DataFrame with encoded values 
                for i, col in enumerate(categorical_cols): 
                    df[col] = tempvar[:, i]
            else:
                df.drop(columns = categorical_cols, axis =1 , inplace=True)
                df = pd.concat([df, tempvar], axis = 1)


    else:

        # Initialize and fit the TargetEncoder 
        mod = model() 
        if model == TargetEncoder:
            mod.fit(df[categorical_cols], df[target])
        else:
            mod.fit(df[categorical_cols])

        
        # Transform the categorical columns 
        tempvar = mod.transform(df[categorical_cols])
        print(type(df))
        print(type(tempvar))
        
        if model == TargetEncoder or model == OrdinalEncoder:
            # Update the original DataFrame with encoded values 
            for i, col in enumerate(categorical_cols): 
                df[col] = tempvar[:, i]
        else:
            df.drop(columns = categorical_cols, axis =1 , inplace=True)
            df = pd.concat([df, tempvar], axis = 1)
            
               
                
        
        with open(file_name, 'wb') as f:
            dill.dump(mod, f)

    return df


with open('encode_data.pickle', 'wb') as f:
    dill.dump(encode_data, f)



# %%
def train_model(xtrain, ytrain, model_class, **kwargs):
    model = model_class(**kwargs)
    model.fit(xtrain, ytrain)

    return model


with open('train_model.pickle', 'wb') as f:
    dill.dump(train_model, f)   




# %%
def predict_model(df_test, model, features = []):

    if type(model) == keras.src.models.sequential.Sequential:
        X_new = df_test.drop(columns=features)
        y_new_pred = model.predict(X_new)[:, 1]

    else:
        X_new = df_test.drop(columns=features)
        y_new_pred = model.predict_proba(X_new)[:, 1] 

    return y_new_pred


with open('predict_model.pickle', 'wb') as f:
    dill.dump(predict_model, f) 



"""# Build Neural Network Model"""

def neural_network_model(X, y, loss='binary_crossentropy', metrics='auc', activations='relu', output_activation='softmax', widths=[64], num_layers=0, epochs=50, batch_size=32, learning_rate=0.001, validation_split=0.3333):
    model_nn = Sequential()
    model_nn.add(Input((X.shape[1],)))
    
    if isinstance(activations, list):
        for i in range(num_layers):
            activation = activations[i % len(activations)]  # Rotate through the activations list
            width = widths[i % len(widths)]  # Rotate through the widths list
            model_nn.add(Dense(width, activation=activation))
            model_nn.add(Dropout(0.4))
    else:
        for i in range(num_layers):
            width = widths[i % len(widths)]  # Rotate through the widths list
            model_nn.add(Dense(width, activation=activations))
            model_nn.add(Dropout(0.4))
    
    model_nn.add(Dense(widths[-1], activation=output_activation))  # Output layer activation

     # Early Stopping callback
    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=10)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model_nn.compile(loss=loss, optimizer=opt, metrics=[metrics])

   

    history = model_nn.fit(X, tf.keras.utils.to_categorical(y), epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[es])

    return model_nn, history