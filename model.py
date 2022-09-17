import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
def build_model_v1(img_size=224):
    x = tf.keras.layers.Input(shape=(img_size,img_size,3))
    y = tf.keras.applications.ResNet50(include_top=False)(x)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    y = tf.keras.layers.Dense(64, activation="leaky_relu")(y)
    y = tf.keras.layers.Dense(1, activation="sigmoid")(y)
    
    return tf.keras.models.Model(x,y)

def build_modelv_2(img_size=224):
    x = tf.keras.layers.Input(shape=(img_size,img_size,3))
    y = tf.keras.applications.EfficientNetB0(include_top=False)(x)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    y = tf.keras.layers.Dense(64, activation="leaky_relu")(y)
    y = tf.keras.layers.Dense(1, activation="sigmoid")(y)
    
    return tf.keras.models.Model(x,y) 

def model_predict(base_model):
    x = tf.keras.layers.Input(shape=(96,96,3))
    y = base_model(include_top=False)(x)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    y = tf.keras.layers.Dense(64, activation="leaky_relu")(y)
    y = tf.keras.layers.Dense(1, activation="sigmoid")(y)
    model = tf.keras.models.Model(x,y)
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=["acc"])
    model.fit(train_loader, epochs=1, validation_data=valid_loader, callbacks=[es])
    y_pred = model.predict(test_loader)
    return y_pred

class model_ensemble():
    def __init__(self, train_ds, valid_ds, test_ds, dim=224, epochs=10, base_models=[EfficientNetB0, EfficientNetB1, EfficientNetB2]):
        self.dim=dim
        self.train_ds=train_ds
        self.valid_ds=valid_ds
        self.test_ds=test_ds
        self.epochs=10
        self.base_models=base_models

    def model_predict(self):
        s