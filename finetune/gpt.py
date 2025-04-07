import os
import pandas as pd
import numpy as np
from PIL import Image
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# Configure TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        print("Memory growth setting failed")
else:
    print("No GPU devices found, using CPU")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageClassifier:
    def __init__(self, image_size=(128, 128), batch_size=8):
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = None
        
    def load_data(self, base_path, train_df, test_df, limit=None):
        """Load and preprocess image data"""
        logging.info("Starting data loading...")
        
        def load_and_preprocess_image(full_path):
            try:
                with Image.open(full_path) as img:
                    img = img.convert("RGB").resize(self.image_size)
                    return np.array(img, dtype=np.float32) / 255.0
            except Exception as e:
                logging.error(f"Error processing {full_path}: {str(e)}")
                return None

        # Process training data
        X, y = [], []
        for idx, row in train_df.iterrows():
            if limit and len(X) >= limit:
                break
            
            img_path = os.path.join(base_path, row['file_name'])
            img_array = load_and_preprocess_image(img_path)
            
            if img_array is not None:
                X.append(img_array)
                y.append(row['label'])
            
            if idx % 100 == 0:
                logging.info(f"Processed {idx} training images")
        
        # Process test data with limit
        X_test, test_ids = [], []
        for idx, row in test_df.iterrows():
          #  if limit and len(X_test) >= limit:
           #     break
                
            img_path = os.path.join(base_path, f"{row['id']}")
            img_array = load_and_preprocess_image(img_path)
            
            if img_array is not None:
                X_test.append(img_array)
                test_ids.append(row['id'])
            
            if idx % 100 == 0:
                logging.info(f"Processed {idx} test images")

        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        
        logging.info(f"Final shapes - Train: {X.shape}, Test: {X_test.shape}")
        return X, y, X_test, test_ids

    def build_model(self):
        """Create a simpler CNN model architecture"""
        self.model = Sequential([
            # First block
            Conv2D(32, (3, 3), activation='relu', padding='same',
                  input_shape=(self.image_size[0], self.image_size[1], 3)),
            MaxPooling2D((2, 2)),
            
            # Second block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            
            # Dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=10):
        """Train the model with basic settings"""
        if self.model is None:
            self.build_model()
        
        # Clear session and garbage collect
        tf.keras.backend.clear_session()
        
        # Compile model with basic settings
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Simple early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train with smaller batches
        try:
            with tf.device('/CPU:0'):  # Force CPU usage
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=self.batch_size,
                    callbacks=[early_stopping],
                    verbose=1
                )
            return history
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            raise

    def predict_and_save(self, X_test, test_ids , output_file="submission.csv"):
        """Generate and save predictions"""
        try:
            with tf.device('/CPU:0'):  # Force CPU usage
                predictions = self.model.predict(
                    X_test,
                    batch_size=self.batch_size
                )
            
            pred_classes = (predictions > 0.5).astype(int).flatten()
            
            submission_df = pd.DataFrame({
                'id': test_ids,
                'label': pred_classes
            })
            submission_df.to_csv(output_file, index=False)
            logging.info(f"Saved predictions to {output_file}")
            
            return submission_df
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise

def main():
    # Configuration
    BASE_PATH = "/kaggle/input/ai-vs-human-generated-dataset/"
    IMAGE_SIZE = (128, 128)  # Reduced image size
    BATCH_SIZE = 32  # Very small batch size
    LIMIT = 20000  # Limit for both train and test
    
    try:
        # Load CSV files
        train_df = pd.read_csv("/kaggle/input/detect-ai-vs-human-generated-images/train.csv")
        test_df = pd.read_csv("/kaggle/input/detect-ai-vs-human-generated-images/test.csv")
        
        # Initialize classifier
        classifier = ImageClassifier(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

        print("Load Pre-Proccess DATA")
        # Load and preprocess data
        X, y, X_test, test_ids = classifier.load_data(BASE_PATH, train_df, test_df, limit=LIMIT)

        print("Print Split DATA")
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )

        print("Start Train")
        # Train model
        history = classifier.train(X_train, y_train, X_val, y_val)

        print("Move To predictions")
        #X_test = "/kaggle/input/detect-ai-vs-human-generated-images/test.csv"
        # Generate predictions and save submission
        classifier.predict_and_save(X_test, test_ids)
        
    except Exception as e:
        logging.error(f"Main execution error: {str(e)}")
        raise

if __name__ == "__main__":
    main()