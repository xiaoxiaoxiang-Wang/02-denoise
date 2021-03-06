from tensorflow import keras

def dncnn(channel):
    input = keras.layers.Input(shape=(100,100,channel),name = 'input')
    conv = conv_layer(input,64,(3,3),bn=False)
    for i in range(15):
        conv = conv_layer(conv,64,(3,3))
    output = conv_layer(conv,1,(3,3),relu = False,bn=False)
    output = keras.layers.Subtract()([input, output])  # input - noise
    model = keras.models.Model(inputs=input, outputs=output)
    return model


def conv_layer(input, filters,kernel_size,relu=True, bn = True):
    layer = keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=(1, 1),kernel_initializer='Orthogonal', padding='same',
               use_bias=False)(input)
    if bn:
        layer = keras.layers.BatchNormalization(axis=-1, momentum=0.0,epsilon=0.0001)(layer)
    if relu:
        layer = keras.layers.Activation(activation='relu')(layer)

    return layer

if __name__ == '__main__':
    model = dncnn(3)

    model.layers[0].output_shape
    model.summary()
    keras.utils.plot_model(model, to_file='model.png', show_shapes=True)