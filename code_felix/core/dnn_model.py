from keras import Sequential, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
import numpy as np

from code_felix.utils_.util_log import logger


class ELO_model:

    def __init__(self,  input_dim,model_type, dropout=0.5):
        self.best_model = './output/checkpoint/dnn_st_best_tmp.hdf5'
        self.model_type = model_type
        self.input_dim = input_dim
        self.dropout = dropout


    def get_dnn_model(self, model_type,  input_dim, dropout):
        if model_type =='dnn1':
            return self.get_dnn_model_dnn1(input_dim, dropout)
        elif model_type =='dnn2':
            return self.get_dnn_model_dnn2(input_dim, dropout)
        elif model_type =='dnn3':
            return self.get_dnn_model_dnn3(input_dim, dropout)
        elif model_type =='dnn4':
            return self.get_dnn_model_dnn4(input_dim, dropout)
        elif model_type =='dnn5':
            return self.get_dnn_model_dnn5(input_dim, dropout)
        elif model_type =='dnn6':
            return self.get_dnn_model_dnn6(input_dim, dropout)
        elif model_type =='dnn7':
            return self.get_dnn_model_dnn7(input_dim, dropout)
        elif model_type == 'dnn8':
            return self.get_dnn_model_dnn8(input_dim, dropout)
        elif model_type == 'dnn9':
            return self.get_dnn_model_dnn9(input_dim, dropout)
        else:
            return self.get_dnn_model_dnn0(input_dim, dropout)

    def get_dnn_model_dnn0(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(input_dim * 2, input_shape=(input_dim,)))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))


        model.add(Dense(32, ))
        model.add(Dropout(dropout))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn1(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(input_dim*2, input_shape=(input_dim,)))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(16, ))
        model.add(Dropout(dropout))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn2(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(input_dim * 2, input_shape=(input_dim,)))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(128, ))
        # model.add(Dropout(dropout))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn3(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))
        # model.add(Dropout(dropout))

        # model.add(Dense(12, ))
        # # model.add(Activation('sigmoid'))
        # # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn4(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(32, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn5(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(24, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn6(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        model.add(Activation('tanh'))
        #model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn7(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        model.add(Activation('tanh'))
        #model.add(LeakyReLU(alpha=0.01))
        #model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn8(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(24, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(6, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn9(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model




    def fit(self,X_train, y_train, X_valid, y_valid):
        check_best = ModelCheckpoint(filepath=self.best_model,
                                    monitor='val_loss',verbose=1,
                                    save_best_only=True, mode='min')

        early_stop = EarlyStopping(monitor='val_loss',verbose=1,
                                   patience=50,
                                   )

        reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                   patience=20, verbose=1, mode='min')

        model = self.get_dnn_model(self.model_type, self.input_dim, self.dropout)
        history = model.fit(X_train, y_train,
                            validation_data=(X_valid, y_valid),
                            callbacks=[check_best, early_stop, reduce],
                            batch_size=128,
                            #steps_per_epoch= len(X_test)//128,
                            epochs=50000,
                            verbose=1,
                            )

        best_epoch = np.array(history.history['val_loss']).argmin()+1
        best_score = np.array(history.history['val_loss']).min()
        logger.debug(f'Best model save to:{self.best_model}, input:{X_train.shape}, bets_epoch:{best_epoch}, best_score:{best_score}')

    def predict(self,X_test):

        classifier = models.load_model(self.best_model)

        y_test =  classifier.predict(X_test)

        #logger.debug(f"y_test:{y_test.shape}")
        return y_test[:,0]




class Stacking_model:

    def __init__(self,  input_dim,model_type, dropout=0.5):
        self.best_model = './output/checkpoint/dnn_best_tmp.hdf5'
        self.model_type = model_type
        self.input_dim = input_dim
        self.dropout = dropout


    def get_dnn_model(self, model_type,  input_dim, dropout):
        if model_type =='dnn1':
            return self.get_dnn_model_dnn1(input_dim, dropout)
        elif model_type =='dnn2':
            return self.get_dnn_model_dnn2(input_dim, dropout)
        elif model_type =='dnn3':
            return self.get_dnn_model_dnn3(input_dim, dropout)
        elif model_type =='dnn4':
            return self.get_dnn_model_dnn4(input_dim, dropout)
        elif model_type =='dnn5':
            return self.get_dnn_model_dnn5(input_dim, dropout)
        elif model_type =='dnn6':
            return self.get_dnn_model_dnn6(input_dim, dropout)
        elif model_type =='dnn7':
            return self.get_dnn_model_dnn7(input_dim, dropout)
        elif model_type == 'dnn8':
            return self.get_dnn_model_dnn8(input_dim, dropout)
        elif model_type == 'dnn9':
            return self.get_dnn_model_dnn9(input_dim, dropout)
        else:
            return self.get_dnn_model_dnn0(input_dim, dropout)

    def get_dnn_model_dnn0(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn1(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0.01))
        #model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn2(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn3(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))
        # model.add(Dropout(dropout))

        # model.add(Dense(12, ))
        # # model.add(Activation('sigmoid'))
        # # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn4(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(32, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn5(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(24, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn6(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        model.add(Activation('tanh'))
        #model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn7(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        model.add(Activation('tanh'))
        #model.add(LeakyReLU(alpha=0.01))
        #model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn8(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(24, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(6, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn9(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model




    def fit(self,X_train, y_train, X_valid, y_valid):
        check_best = ModelCheckpoint(filepath=self.best_model,
                                    monitor='val_loss',verbose=1,
                                    save_best_only=True, mode='min')

        early_stop = EarlyStopping(monitor='val_loss',verbose=1,
                                   patience=20,
                                   )

        reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                   patience=10, verbose=1, mode='min')

        model = self.get_dnn_model(self.model_type, self.input_dim, self.dropout)
        history = model.fit(X_train, y_train,
                            validation_data=(X_valid, y_valid),
                            callbacks=[check_best, early_stop, reduce],
                            batch_size=128,
                            #steps_per_epoch= len(X_test)//128,
                            epochs=100,
                            verbose=1,
                            )

        best_epoch = np.array(history.history['val_loss']).argmin()+1
        best_score = np.array(history.history['val_loss']).min()
        logger.debug(f'Best model save to:{self.best_model}, input:{X_train.shape}, bets_epoch:{best_epoch}, best_score:{best_score}')

    def predict(self,X_test):

        classifier = models.load_model(self.best_model)

        y_test =  classifier.predict(X_test)

        logger.debug(f"y_test:{y_test.shape}")
        return y_test[:,0]



class Stacking_model:

    def __init__(self,  input_dim,model_type, dropout=0.5):
        self.best_model = './output/checkpoint/dnn_best_tmp.hdf5'
        self.model_type = model_type
        self.input_dim = input_dim
        self.dropout = dropout


    def get_dnn_model(self, model_type,  input_dim, dropout):
        if model_type =='dnn1':
            return self.get_dnn_model_dnn1(input_dim, dropout)
        elif model_type =='dnn2':
            return self.get_dnn_model_dnn2(input_dim, dropout)
        elif model_type =='dnn3':
            return self.get_dnn_model_dnn3(input_dim, dropout)
        elif model_type =='dnn4':
            return self.get_dnn_model_dnn4(input_dim, dropout)
        elif model_type =='dnn5':
            return self.get_dnn_model_dnn5(input_dim, dropout)
        elif model_type =='dnn6':
            return self.get_dnn_model_dnn6(input_dim, dropout)
        elif model_type =='dnn7':
            return self.get_dnn_model_dnn7(input_dim, dropout)
        elif model_type == 'dnn8':
            return self.get_dnn_model_dnn8(input_dim, dropout)
        elif model_type == 'dnn9':
            return self.get_dnn_model_dnn9(input_dim, dropout)
        else:
            return self.get_dnn_model_dnn0(input_dim, dropout)

    def get_dnn_model_dnn0(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn1(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0.01))
        #model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn2(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn3(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))
        # model.add(Dropout(dropout))

        # model.add(Dense(12, ))
        # # model.add(Activation('sigmoid'))
        # # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn4(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(32, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn5(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(24, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn6(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        model.add(Activation('tanh'))
        #model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn7(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        model.add(Activation('tanh'))
        #model.add(LeakyReLU(alpha=0.01))
        #model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn8(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(24, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout))

        model.add(Dense(6, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model

    def get_dnn_model_dnn9(self, input_dim, dropout):
        model = Sequential()
        model.add(Dense(12, input_shape=(input_dim,)))
        # model.add(Activation('sigmoid'))
        model.add(LeakyReLU(alpha=0))
        model.add(Dropout(dropout))

        model.add(Dense(12, ))
        # model.add(Activation('sigmoid'))
        # model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1, ))

        adam = Adam()
        model.compile(loss='mse', optimizer=adam, )
        model.summary()
        self.model = model
        return model




    def fit(self,X_train, y_train, X_valid, y_valid):
        check_best = ModelCheckpoint(filepath=self.best_model,
                                    monitor='val_loss',verbose=1,
                                    save_best_only=True, mode='min')

        early_stop = EarlyStopping(monitor='val_loss',verbose=1,
                                   patience=20,
                                   )

        reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                   patience=10, verbose=1, mode='min')

        model = self.get_dnn_model(self.model_type, self.input_dim, self.dropout)
        history = model.fit(X_train, y_train,
                            validation_data=(X_valid, y_valid),
                            callbacks=[check_best, early_stop, reduce],
                            batch_size=128,
                            #steps_per_epoch= len(X_test)//128,
                            epochs=100,
                            verbose=1,
                            )

        best_epoch = np.array(history.history['val_loss']).argmin()+1
        best_score = np.array(history.history['val_loss']).min()
        logger.debug(f'Best model save to:{self.best_model}, input:{X_train.shape}, bets_epoch:{best_epoch}, best_score:{best_score}')

    def predict(self,X_test):

        classifier = models.load_model(self.best_model)

        y_test =  classifier.predict(X_test)

        logger.debug(f"y_test:{y_test.shape}")
        return y_test[:,0]

