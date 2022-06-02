from threading import Thread

from tensorflow.python import ipu
import tensorflow as tf


#
# Configure the IPU system
#
cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 1
cfg.configure_ipu_system()


#
# The input data and labels
#
def create_dataset():
  mnist = tf.keras.datasets.mnist

  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000)
  train_ds = train_ds.map(lambda d, l:
                          (tf.cast(d, tf.float32), tf.cast(l, tf.int32)))
  train_ds = train_ds.batch(32, drop_remainder=True)
  return train_ds.repeat()


#
# The host side queue
#
outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()


#
# A custom training loop
#
@tf.function(experimental_compile=True)
def training_step(steps_per_execution, iterator, in_model, optimizer):
  for _ in tf.range(steps_per_execution):
    features, labels = next(iterator)
    with tf.GradientTape() as tape:
      predictions = in_model(features, training=True)
      prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, predictions)
      loss = tf.reduce_mean(prediction_loss)
      grads = tape.gradient(loss, in_model.trainable_variables)
      optimizer.apply_gradients(zip(grads, in_model.trainable_variables))

    outfeed_queue.enqueue(loss)


#
# Execute the graph
#
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  # Create the Keras model and optimizer.
  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  optimizer = tf.keras.optimizers.Adam()

  # Create an iterator for the dataset.
  train_iterator = iter(create_dataset())

  # Run the custom training loop over the dataset.
  num_iterations = 100
  steps_per_execution = 10

  for begin_step in range(0, num_iterations, steps_per_execution):
    strategy.run(training_step, args=[steps_per_execution, train_iterator, model, optimizer])
    mean_loss = sum(outfeed_queue) / steps_per_execution
    print(mean_loss)