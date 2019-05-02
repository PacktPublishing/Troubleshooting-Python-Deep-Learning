# Inspired by:
# https://stackoverflow.com/questions/45723596/keras-how-to-concatenate-two-cnn
from keras.models import Model
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D
from keras.layers import Input,Dense
from keras.layers import concatenate

# Creating the first CNN
cnn1_input = Input(shape=(100,))
cnn1 = Embedding(32, 100, input_length=100)(cnn1_input)
cnn1 = Conv1D(32, 2, padding='valid', activation='relu', strides=1)(cnn1)
cnn1 = GlobalMaxPooling1D(cnn1)

# Create the second CNN
cnn2_input = Input(shape=(100,))
cnn2 = Embedding(32, 100, input_length=100)(cnn2_input)
cnn2 = Conv1D(32, 2, padding='valid', activation='relu', strides=1)(cnn2)
cnn2 = GlobalMaxPooling1D(cnn2)

# Merge two networks
merged_cnns = concatenate([cnn1, cnn2])

# Define the common output
cnn_outputs = Dense(1, activation='sigmoid')(merged_cnns)

# Put everything together
model = Model(inputs=[cnn1_input, cnn2_input], outputs=cnn_outputs)

model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'])
