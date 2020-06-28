import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import Model
from keras.layers import Input, Dense, LSTM, SpatialDropout1D, Bidirectional
from keras.callbacks import ModelCheckpoint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = stderr


def LSTM_model(max_emb_len, num_features, num_classes):
    input = Input(shape=(max_emb_len, num_features))
    l1 = SpatialDropout1D(0.2)(input)
    l2 = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(l1)
    l3 = Dense(num_classes, activation='softmax')(l2)
    model = Model(inputs=input, outputs=l3)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def BiLSTM_model(max_emb_len, num_features, num_classes):
    input = Input(shape=(max_emb_len, num_features))
    l1 = Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(input)
    l2 = Bidirectional(LSTM(32, dropout=0.1, recurrent_dropout=0.1))(l1)
    l3 = Dense(num_classes, activation='softmax')(l2)
    model = Model(inputs=input, outputs=l3)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()


def save_history_plot(history, save_path):
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='test_loss')
    plt.legend()
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='test_acc')
    plt.legend()
    plt.savefig(save_path)


def train_lstm(x_source, y_source, label_type, convert_type, save_folder, batch_size=64, epochs=10):
    print(f"\nTrain {label_type} LSTM")
    max_emb_len = 64 if convert_type == 'length_64' else 1
    num_features = x_source.shape[2]
    num_classes = y_source.shape[1]
    model = LSTM_model(max_emb_len, num_features, num_classes)
    model_checkpoint = ModelCheckpoint(os.path.join(save_folder, f'lstm_{label_type}_{convert_type}.hdf5'),
                                       save_best_only=True)
    model.summary()
    history = model.fit(x=x_source, y=y_source, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                        callbacks=[model_checkpoint])
    save_history_plot(history, os.path.join(save_folder, f'lstm_{label_type}_{convert_type}'))
    model.load_weights(os.path.join(save_folder, f'lstm_{label_type}_{convert_type}.hdf5'))
    return model


def load_lstm(label_type, convert_type, folder):
    print(f"\nLoad {label_type} LSTM checkpoint")
    max_emb_len = 64 if convert_type == 'length_64' else 1
    num_features = 768
    num_classes = 3 if label_type == 'tonality' else 2
    model = LSTM_model(max_emb_len, num_features, num_classes)
    model.load_weights(os.path.join(folder, f'lstm_{label_type}_{convert_type}.hdf5'))
    return model


def predict(model, x, y, title=''):
    y_pred = model.predict(x)
    print('\n'+title)
    print(classification_report(y.argmax(1), y_pred.argmax(1), digits=4))
    y_pred = y_pred.argmax(1)
    return y_pred
