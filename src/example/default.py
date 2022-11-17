import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def get_data(file_url, file_name):
    urllib.request.urlretrieve(file_url, filename=file_name)

    return pd.read_table(file_name)

def text_processing(train_data):
    processed_data = train_data
    processed_data.drop_duplicates(subset=['document'], inplace=True)   # document 열의 중복 제거 (원본 값 변경 허용)
    processed_data = processed_data.dropna(how = 'any') # document 열의 NaN 제거
    processed_data['document'] = processed_data['document'].str.replace("[^ㄱ-힣 ]", "")    # document 열의 값들 한국어와 공백만 남기고 모두 제거
    processed_data['document'].replace("", np.nan, inplace=True)    # 빈칸 document값은 NaN으로 변환 (원본 값 변경 허용)
    processed_data = processed_data.dropna(how = 'any') # document 열의 NaN 제거

    return processed_data

def tokenization(data):
    okt = Okt()
    tokenized = list()

    for sentence in tqdm(data['document']):
        tokenized.append([pos_set[0] for pos_set in okt.pos(okt.normalize(sentence)) if pos_set[1] != 'Josa'])

    return tokenized

def integer_encoding(tokenized_data, processed_data, min_freq=3):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokenized_data)

    word_count = len(tokenizer.word_index) + 1
    for (_, freq) in tokenizer.word_counts.items():
        if freq < min_freq: word_count -= 1

    tokenizer = Tokenizer(word_count)
    tokenizer.fit_on_texts(tokenized_data)

    ret = tokenizer.texts_to_sequences(tokenized_data)
    sol = np.array(processed_data['label'])
    drop = [idx for (idx, sentence) in enumerate(ret) if len(sentence) == 0]

    return {
        'encoded': np.delete(ret, drop, axis=0),
        'solution': np.delete(sol, drop, axis=0),
        'size': word_count
    }

def padding(integer_encoded, max_len=None):
    return pad_sequences(integer_encoded) if max_len is None else pad_sequences(integer_encoding, max_len=max_len)

def train(train_data, solution, size, show_accuracy=True, best_model_name='best_model', embedding_dim=100, hidden_units=128, verbose=1, patience=4, epochs=15, batch_size=64, validation_split=0.2, save_best_only=True):
    model = Sequential()
    model.add(Embedding(size, embedding_dim))
    model.add(LSTM(hidden_units))
    model.add(Dense(1, activation='sigmoid'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=patience)
    mc = ModelCheckpoint(best_model_name + '.h5', monitor='val_acc', mode='max', verbose=verbose, save_best_only=save_best_only)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.fit(train_data, solution, epochs=epochs, callbacks=[es, mc], batch_size=batch_size, validation_split=validation_split)

    if show_accuracy: print("accuracy: %.4f" % (load_model(best_model_name + '.h5').evaluate(train_data, solution)[1]))

    return model

def predict(review, word_count, pad_max_len=None, best_model_name='best_model'):
    okt = Okt()
    tokenizer = Tokenizer(word_count)

    review = review.replace("[^ㄱ-힣 ]", "")
    review = [[pos_set[0] for pos_set in okt.pos(okt.normalize(review)) if pos_set[1] != 'Josa']]
    encoded = tokenizer.texts_to_sequences(review)
    pad_new = padding(encoded, max_len=pad_max_len)
    score = float(load_model(best_model_name + '.h5').predict(pad_new)) # 예측
    
    return score

class AI:
    def __init__(self, train_file_url, train_file_name):
        self.data = text_processing(get_data(train_file_url, train_file_name))
        self.tokenized = tokenization(self.data)
        self.encoded = integer_encoding(self.tokenized, self.data)
        self.padded = padding(self.encoded['encoded'])
        self.model = train(self.padded, self.encoded['solution'], self.encoded['size'])

    def is_positive(self, review, is_print=True):
        score = predict(review, self.encoded['size'])
        if is_print: print("{:.2f}% positive" % (score * 100))
        else: return score

##############################################################

ai = AI("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", "ratings_train.txt")

ai.is_positive("올해 최고의 영화! 세 번 넘게 봐도 질리지가 않네요.")