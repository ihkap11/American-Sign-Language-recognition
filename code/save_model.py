import keras
from keras.models import model_from_yaml


def save_model(model):
    
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    # serialize weights to HDF5
    model.save_weights("model_95.66.h5")
    print("Saved model to disk")
    

def load_model():
    
    # load YAML and create model
    yaml_file = open('model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    loaded_model.load_weights("model_95.66.h5")
    print("Loaded model from disk")
    
    return loaded_model
    


def score_adam_optimizer(X_test, y_out, loaded_model):
    
    adam = keras.optimizers.Adam(lr= 0.001)
    loaded_model.compile(optimizer=adam, 
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    
    score = loaded_model.evaluate(X_test, y_out, verbose=1)
    print("acc: %.2f%%" % ( score[1]*100))
    print("loss: %.2f%%" % (score[0]*100))   



def score_rmsprop_optimizer(X_test, y_out, loaded_model):
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    score = loaded_model.evaluate(X_test, y_out, verbose=1)
    print("acc: %.2f%%" % ( score[1]*100))
    print("loss: %.2f%%" % (score[0]*100))
    

