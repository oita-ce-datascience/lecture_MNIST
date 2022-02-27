import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer

#MNISTのデータを取得
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#データの確認
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#トレーニング用データを28x28　→  784へ変換。
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)

#データの確認
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#x_trainの最初を確認(0-255である)
x_train[0]

#トレーニング用データをsloatへ変換
x_train=x_train.astype("float")
x_test=x_test.astype("float")

#ピクセル情報は0 が白、255に近くにつれて黒になる。これを0から1に変換したいため255で割る。
x_train /= 255
x_test  /= 255

#0-1になっているか確認。
x_train[0]

#y_trainとy_testをワンホット表現へ変換。
y_train=np.identity(10)[y_train]
y_test=np.identity(10)[y_test]

#model構築
model = Sequential()
model.add(InputLayer(input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#学習回数とバッチサイズを設定。その後、学習。
epochs = 20
batch_size = 128
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# 検証
score = model.evaluate(x_test, y_test, verbose=1)
print()
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#modelを使用して予測。
pred=model.predict(x_test)

#0番目のテストデータ、実際の値　→  7
y_test[0]

#0番目の予測　
pred[0]

f"確率={pred[0][7]*100}%"

#念の為、他のデータにて確認
#5000番目のテストデータ、実際の値　→  3
y_test[5000]

#5000番目の予測　
pred[5000]

f"確率={pred[5000][3]*100}%"

f"確率={pred[5000][5]*100}%"
