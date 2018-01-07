import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense

num_class = 36


def create_model(X_train, num_class, showSummary = False ):
    
    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=X_train[0].shape))
    model.add(Activation('relu'))       
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(96, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # model.add(Conv2D(120, (2, 2)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))  

    model.add(Flatten())        

    model.add(Dense(1024))
    model.add(Activation('relu'))  


    # model.add(MaxPooling2D(pool_size=(2,2)))  
    # model.add(Flatten())
    # model.add(Dropout(0.2))

    model.add(Dense(num_class))
    model.add(Activation('softmax'))

    if(showSummary == True):
        model.summary()
        
    return model



def train_model(X_train, X_test, y_in, y_out, val_rate = 1000, lr_ = 0.001):
    
    model = create_model(X_train, num_class)
    
    adam = keras.optimizers.Adam(lr=lr_)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    model.fit(X_train, y_in,
              batch_size=128, 
              epochs = 20,
              validation_data=(X_test[:val_rate], y_out[:val_rate]))
    
    return model
    

def get_score(X_test, y_out,model, verb = 1):
    
    score = model.evaluate(X_test, y_out, verbose=verb)
    print("acc: %.2f%%" % ( score[1]*100))
    print("loss: %.2f%%" % (score[0]*100))
  
    
