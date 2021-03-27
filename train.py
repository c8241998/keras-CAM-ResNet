import tensorflow as tf
from cam import get_model

if __name__=='__main__':

    # define your dataset here
    train_dataset = None
    val_dataset = None

    BATCH_SIZE = 16

    train_dataset = train_dataset.shuffle(len(train_dataset)).batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(1)

    model = get_model(classes = 2)

    # freeze resnet
    for layer in model.layers[0].layers:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    checkpoint_filepath = '/tmp/checkpoint'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='auto',
        save_best_only=True)

    model.fit(train_dataset,
              epochs=20,
              validation_data=val_dataset,
              shuffle=True,
              callbacks=[model_checkpoint_callback]
              )

    model.load_weights(checkpoint_filepath)

    model.evaluate(val_dataset)