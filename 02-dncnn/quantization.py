import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('./models')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()