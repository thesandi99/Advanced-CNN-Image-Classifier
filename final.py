import tensorflow as tf
from tensorflow.keras import layers, models, Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from tensorflow.keras.callbacks import ReduceLROnPlateau


class Config:
    image_size = 224
    batch_size = 32  
    epochs = 10

class CNNModel:
    def __init__(self, input_shape, num_classes, model_name="cnn_model"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = self.create_model()

    def create_model(self):
        raise NotImplementedError("Subclasses must implement create_model()")

    def forward_pass(self, input_data, output_layer_names=None):
        if output_layer_names is None:
            return self.model(input_data)

        intermediate_outputs = {}
        for layer_name in output_layer_names:
            try:
                intermediate_outputs[layer_name] = self.model.get_layer(layer_name).output
            except ValueError:
                print(f"Warning: Layer '{layer_name}' not found in the model.")

        intermediate_model = Model(inputs=self.model.input, outputs=list(intermediate_outputs.values()))
        intermediate_results = intermediate_model.predict(input_data)

        return dict(zip(intermediate_outputs.keys(), intermediate_results))
        
# --- Custom Layers ---
class ParallelConvBlock(layers.Layer):
    def __init__(self, filters_list, kernel_sizes_list, strides_list, activation='relu', name=None, **kwargs):
        super(ParallelConvBlock, self).__init__(name=name, **kwargs)
        self.filters_list = filters_list
        self.kernel_sizes_list = kernel_sizes_list
        self.strides_list = strides_list
        self.activation = activation
        self.conv_layers = []

        if not (len(filters_list) == len(kernel_sizes_list) == len(strides_list)):
            raise ValueError("filters_list, kernel_sizes_list, and strides_list must have the same length.")

        for i in range(len(filters_list)):
            self.conv_layers.append(
                layers.Conv2D(
                    filters=filters_list[i],
                    kernel_size=kernel_sizes_list[i],
                    strides=strides_list[i],
                    padding='same',
                    activation=self.activation,
                    name=f'conv2d_{i}'
                )
            )

    def call(self, inputs):
        outputs = []
        for conv_layer in self.conv_layers:
            outputs.append(conv_layer(inputs))
        return layers.concatenate(outputs)

    def get_config(self):
        config = super(ParallelConvBlock, self).get_config()
        config.update({
            'filters_list': self.filters_list,
            'kernel_sizes_list': self.kernel_sizes_list,
            'strides_list': self.strides_list,
            'activation': self.activation,
        })
        return config

class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, use_parallel=False, filters_list=None, kernel_sizes_list=None, strides_list=None, name=None, **kwargs):
        super(ResidualBlock, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_parallel = use_parallel
        self.filters_list = filters_list
        self.kernel_sizes_list = kernel_sizes_list
        self.strides_list = strides_list

        if use_parallel:
            if not all([filters_list, kernel_sizes_list, strides_list]):
                raise ValueError("When use_parallel=True, provide filters_list, kernel_sizes_list, and strides_list.")
            self.conv1 = ParallelConvBlock(filters_list, kernel_sizes_list, strides_list, name=f'{self.name}_parallel_conv1')
            self.adjusted_filters_list = [f * len(filters_list) for f in filters_list]
        else:
            self.conv1 = layers.Conv2D(filters, kernel_size, padding='same', activation=None, name=f'{self.name}_conv1')

        self.bn1 = layers.BatchNormalization(name=f'{self.name}_bn1')
        self.activation1 = layers.Activation('relu', name=f'{self.name}_relu1')

        if use_parallel:
            self.conv2 = ParallelConvBlock(self.adjusted_filters_list, kernel_sizes_list, strides_list, name=f'{self.name}_parallel_conv2')
        else:
            self.conv2 = layers.Conv2D(filters, kernel_size, padding='same', activation=None, name=f'{self.name}_conv2')

        self.bn2 = layers.BatchNormalization(name=f'{self.name}_bn2')
        self.add = layers.Add(name=f'{self.name}_add')
        self.activation2 = layers.Activation('relu', name=f'{self.name}_relu2')
        self.shortcut = None

    def build(self, input_shape):
        if self.use_parallel:
            if input_shape[-1] != sum(self.adjusted_filters_list):
                self.shortcut = layers.Conv2D(sum(self.adjusted_filters_list), (1, 1), padding='same', name=f'{self.name}_shortcut')
            else:
                self.shortcut = lambda x: x
        else:
            if input_shape[-1] != self.filters:
                self.shortcut = layers.Conv2D(self.filters, (1, 1), padding='same', name=f'{self.name}_shortcut')
            else:
                self.shortcut = lambda x: x
        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        shortcut = self.shortcut(inputs)
        x = self.add([x, shortcut])
        x = self.activation2(x)
        return x

    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'use_parallel': self.use_parallel,
            'filters_list': self.filters_list,
            'kernel_sizes_list': self.kernel_sizes_list,
            'strides_list': self.strides_list,
        })
        return config

class SelfAttention(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(SelfAttention, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.W_q = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                   initializer='glorot_uniform',
                                   trainable=True,
                                   name='query_weight')
        self.W_k = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                   initializer='glorot_uniform',
                                   trainable=True,
                                   name='key_weight')
        self.W_v = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                   initializer='glorot_uniform',
                                   trainable=True,
                                   name='value_weight')
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        q = tf.matmul(inputs, self.W_q)
        k = tf.matmul(inputs, self.W_k)
        v = tf.matmul(inputs, self.W_v)

        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores / tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32)))
        attention_output = tf.matmul(attention_scores, v)
        return attention_output

    def get_config(self):
      return super(SelfAttention, self).get_config()

# --- Model Definition ---
class AdvancedCNN(CNNModel):
    def __init__(self, input_shape, num_classes, use_attention=True, use_parallel=True):
        self.use_attention = use_attention
        self.use_parallel = use_parallel
        super().__init__(input_shape, num_classes, model_name="advanced_cnn")

    def create_model(self):
        inputs = layers.Input(shape=self.input_shape, name='input_layer')
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2d')(inputs)

        x = ResidualBlock(64, 3, use_parallel=self.use_parallel, filters_list=[16, 32],
                          kernel_sizes_list=[(3, 3), (5, 5)], strides_list=[(1, 1), (1, 1)], name='residual_block')(x)
        x = layers.MaxPooling2D((2, 2), name='max_pooling2d')(x)
        x = ResidualBlock(128, 3, use_parallel=self.use_parallel, filters_list=[32, 64],
                          kernel_sizes_list=[(3, 3), (5, 5)], strides_list=[(1, 1), (1, 1)], name='residual_block_1')(x)
        x = layers.MaxPooling2D((2, 2), name='max_pooling2d_1')(x)
        x = ResidualBlock(256, 3, use_parallel=self.use_parallel, filters_list=[64, 128],
                          kernel_sizes_list=[(3, 3), (5, 5)], strides_list=[(1, 1), (1, 1)], name='residual_block_2')(x)
        x = layers.MaxPooling2D((2, 2), name='max_pooling2d_2')(x)

        if self.use_attention:
            x = SelfAttention(name='self_attention')(x)

        x = layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
        x = layers.Dense(128, activation='relu', name='dense')(x)
        x = layers.Dropout(0.5, name='dropout')(x)
        outputs = layers.Dense(self.num_classes, activation='sigmoid', name='dense_1')(x)

        model = Model(inputs, outputs)
        return model



    def compile_model(self, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], jit_compile=True):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=jit_compile)

    def summary(self):
        self.model.summary()

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self,filepath):
        self.model = models.load_model(filepath, custom_objects={
            'ParallelConvBlock': ParallelConvBlock,
            'ResidualBlock': ResidualBlock,
            'SelfAttention': SelfAttention
            })
        return self.model

# --- Data Loading and Preprocessing (tf.data) ---
def load_and_preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [Config.image_size, Config.image_size])
    img = img / 255.0
    return img, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def create_dataset(file_paths, labels, batch_size, is_training=True):
    file_paths = tf.convert_to_tensor(file_paths.to_numpy(), dtype=tf.string)
    labels = tf.convert_to_tensor(labels.to_numpy(), dtype=tf.float32)  # Use float32 for sigmoid

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(file_paths))

    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def load_and_preprocess_test_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [Config.image_size, Config.image_size])
    img = img / 255.0
    return img

# --- Visualization ---
def visualize_activations(model, img_array, layer_names):
    img_tensor = np.expand_dims(img_array, axis=0)
    outputs = [model.model.get_layer(layer_name).output for layer_name in layer_names]
    activation_model = Model(inputs=model.model.input, outputs=outputs)
    activations = activation_model.predict(img_tensor)

    for layer_name, layer_activation in zip(layer_names, activations):
        if len(layer_activation.shape) != 4:
            print(f"Skipping layer '{layer_name}' (not a convolutional layer).")
            continue

        num_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        images_per_row = min(num_features, 16)
        n_cols = num_features // images_per_row
        n_cols = max(1, n_cols)
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= (channel_image.std() + 1e-5)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.show()


# --- Main Execution ---
if __name__ == '__main__':
    # --- Setup ---
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # Mixed precision (if supported)


    # --- Data Loading ---
    base_dir = '/kaggle/input/ai-vs-human-generated-dataset'
    train_csv_path = os.path.join(base_dir, 'train.csv')
    test_csv_path = os.path.join(base_dir, 'test.csv')

    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Reduce data (optional, but useful for faster experimentation)
    df_train = df_train.sample(frac=0.5, random_state=42)
    # df_test = df_test.sample(frac=0.9, random_state=42)

    df_test['id'] = df_test['id'].apply(lambda x: os.path.join(base_dir, x))
    df_train['file_name'] = df_train['file_name'].apply(lambda x: os.path.join(base_dir, x))

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        df_train['file_name'],
        df_train['label'],
        test_size=0.05,
        random_state=42,
        stratify=df_train['label']
    )
    print(f"Train Data: {len(train_paths)}")
    print(f"Validation Data: {len(val_paths)}")

    input_shape = (Config.image_size, Config.image_size, 3)
    num_classes = 1

    # --- Create tf.data Datasets ---
    train_dataset = create_dataset(train_paths, train_labels, Config.batch_size, is_training=True)
    validation_dataset = create_dataset(val_paths, val_labels, Config.batch_size, is_training=False)

    # --- Model Creation and Compilation ---
    advanced_cnn = AdvancedCNN(input_shape, num_classes, use_attention=True, use_parallel=True)
    advanced_cnn.compile_model(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], jit_compile=True)
    advanced_cnn.summary()

    # --- Training ---
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-6)

    history = advanced_cnn.model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=Config.epochs,
        callbacks=[reduce_lr]  # Add the callback
    )

    # --- Visualize Training History ---
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    # --- Inference ---
    test_dataset = tf.data.Dataset.from_tensor_slices(df_test['id'].to_numpy())
    test_dataset = test_dataset.map(load_and_preprocess_test_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(Config.batch_size)

    predictions = advanced_cnn.model.predict(test_dataset)
    predictions = (predictions > 0.5).astype(int).flatten()

    submission_df = pd.DataFrame({
        'id': df_test['id'].apply(lambda x: os.path.basename(x)),
        'label': predictions
    })

    submission_df.to_csv('submission.csv', index=False)
    print("Submission file created successfully!")

    # --- Visualization (Optional) ---
    image_path = '/kaggle/input/ai-vs-human-generated-dataset/test_data_v2/01f833bdb5a6489da275bee55249a053.jpg' # Example

    try:
        img = Image.open(image_path)
        img = img.resize(input_shape[:2])
        img_array = np.array(img) / 255.0

        plt.imshow(img_array)
        plt.title("Original Image (Resized)")
        plt.show()

        layer_names_to_visualize = [
            'conv2d',
            'residual_block',
            'residual_block_1',
            'residual_block_2',
            'self_attention'
        ]
        visualize_activations(advanced_cnn, img_array, layer_names_to_visualize)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
    except Exception as e:
      print(f"An error occurred: {e}")