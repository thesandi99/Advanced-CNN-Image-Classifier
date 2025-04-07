import os
import pandas as pd
import numpy as np
from PIL import Image
import logging
import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout, 
                                   BatchNormalization, GlobalAveragePooling2D,
                                   Input, concatenate)
from tensorflow.keras.applications import EfficientNetB4, Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                      ModelCheckpoint, CSVLogger)
from sklearn.metrics import classification_report, roc_curve, auc, f1_score
from tensorflow.keras import mixed_precision
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import time
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class HighPerformanceClassifier:
    def __init__(self, image_size=(299, 299), batch_size=16):
        self.image_size = image_size
        self.batch_size = batch_size
        self.models = []  # For ensemble
        self.augmentation = self.create_augmentation_layer()
        print(f"\nInitialized Classifier with image size {image_size}")
    
    def load_data(self, base_path, train_df, test_df, limit=None):
        """Loads and preprocesses image data from CSV files."""
        print("\nLoading and preprocessing image data...")

        # --- Process training data ---
        train_images = []
        train_labels = []
        for index, row in tqdm(train_df.iterrows(), total=len(train_df)):
            if limit and len(train_images) >= limit:
                break
            image_path = os.path.join(base_path, row['file_name'])  
            try:
                img = Image.open(image_path)
                img = img.resize(self.image_size) # Resize here to ensure consistency
                img_array = np.array(img)
                if img_array.shape != (self.image_size[0], self.image_size[1], 3):
                    print(f"Warning: Image at {image_path} has unexpected shape {img_array.shape}. Skipping.")
                    continue #Skip images with wrong shape.
                train_images.append(img_array)
                train_labels.append(row['label']) 
            except FileNotFoundError:
                print(f"Warning: Image not found at {image_path}. Skipping.")
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue # Skip images that cause errors during loading


        X = np.array(train_images)
        y = np.array(train_labels)

        # --- Process testing data ---
        # (Similar error checking for test images)
        test_images = []
        test_ids = []
        for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            image_path = os.path.join(base_path, row['id'])  
            try:
                img = Image.open(image_path)
                img = img.resize(self.image_size) #Resize here
                img_array = np.array(img)
                if img_array.shape != (self.image_size[0], self.image_size[1], 3):
                    print(f"Warning: Image at {image_path} has unexpected shape {img_array.shape}. Skipping.")
                    continue #Skip images with wrong shape.
                test_images.append(img_array)
                test_ids.append(row['id']) 
            except FileNotFoundError:
                print(f"Warning: Image not found at {image_path}. Skipping.")
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue # Skip images that cause errors during loading

        X_test = np.array(test_images)
        test_ids = np.array(test_ids)

        print(f"Loaded {len(X)} training images and {len(X_test)} testing images.")
        return X, y, X_test, test_ids
    

    def focal_loss(self, gamma=2., alpha=.25):
        """Custom focal loss implementation"""
        def focal_loss_fixed(y_true, y_pred):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) \
                   -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
        return focal_loss_fixed
        
    def create_augmentation_layer(self):
        """Enhanced data augmentation pipeline"""
        print("\nCreating advanced augmentation pipeline...")
        return tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.GaussianNoise(0.01),
            # Uncomment if using TF 2.9+
          #  tf.keras.layers.RandomCutmix(factor=0.4),
          #  tf.keras.layers.RandomMixup(factor=0.4),
           # tf.keras.metrics.F1Score(threshold=0.5, name='f1')
        ])

    def preprocess_image(self, image):
        """Advanced image preprocessing"""
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.image.random_jpeg_quality(image, 80, 100)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.1)
        return image

    def create_dataset(self, X, y=None, is_training=False):
        """Optimized tf.data pipeline"""
        print(f"\nCreating {'training' if is_training else 'validation'} dataset...")
        
        if y is not None:
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(X)
        
        if is_training:
            dataset = dataset.map(
                lambda x, y: (self.preprocess_image(self.augmentation(x, training=True)), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.shuffle(1000)
        else:
            if y is not None:
                dataset = dataset.map(
                    lambda x, y: (self.preprocess_image(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            else:
                dataset = dataset.map(
                    lambda x: self.preprocess_image(x),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            
        dataset = dataset.cache()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    def build_model(self):
        """Create an advanced ensemble model architecture"""
        print("\nBuilding enhanced model architecture...")
        
        # EfficientNetB4 branch
        efficient_net = EfficientNetB4(
            include_top=False,
            weights='imagenet',
            input_shape=(self.image_size[0], self.image_size[1], 3)
        )
        
        # Xception branch
        xception = Xception(
            include_top=False,
            weights='imagenet',
            input_shape=(self.image_size[0], self.image_size[1], 3)
        )
        
        # Fine-tune the last few layers
        for layer in efficient_net.layers[-30:]:
            layer.trainable = True
        for layer in xception.layers[-30:]:
            layer.trainable = True
            
        # Create input
        input_tensor = Input(shape=(self.image_size[0], self.image_size[1], 3))
        
        # Process through both models
        x1 = efficient_net(input_tensor)
        x2 = xception(input_tensor)
        
        # Global pooling
        x1 = GlobalAveragePooling2D()(x1)
        x2 = GlobalAveragePooling2D()(x2)
        
        # Concatenate features
        x = concatenate([x1, x2])
        
        # Dense layers
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        # Output
        outputs = Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = Model(inputs=input_tensor, outputs=outputs)
        
        print("\nModel Architecture Summary:")
        model.summary()
        
        return model

    def train_fold(self, X_train, y_train, X_val, y_val, fold_num, epochs=100):
        """Train a single fold"""
        print(f"\nTraining fold {fold_num}...")
        
        model = self.build_model()
        
        # Learning rate schedule
        initial_learning_rate = 1e-4
        steps_per_epoch = len(X_train) // self.batch_size
        
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            steps_per_epoch * 10,  # Restart every 10 epochs
            t_mul=2.0,  # Double the period after each restart
            m_mul=0.9   # Slightly reduce max learning rate after each restart
        )
        
        # Optimizer with weight decay
        optimizer = AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-5
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=self.focal_loss(gamma=2.0, alpha=0.25),
            metrics=[
                'accuracy',
                tf.keras.metrics.F1Score(threshold=0.5, name='f1'),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_f1',
                patience=15,
                restore_best_weights=True,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_f1',
                factor=0.5,
                patience=7,
                mode='max',
                min_lr=1e-7
            ),
            ModelCheckpoint(
                f'best_model_fold_{fold_num}.keras',
                monitor='val_f1',
                save_best_only=True,
                mode='max'
            ),
            CSVLogger(f'training_log_fold_{fold_num}.csv')
        ]
        
        # Create datasets
        train_dataset = self.create_dataset(X_train, y_train, is_training=True)
        val_dataset = self.create_dataset(X_val, y_val, is_training=False)
        
        # Train
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history

    def train_with_folds(self, X, y, n_folds=5, epochs=100, min_samples=1000):
        """Train with stratified k-fold cross validation"""
        print(f"\nStarting {n_folds}-fold cross validation...")

        if len(X) < min_samples:
             raise ValueError(f"Training dataset too small. Minimum {min_samples} samples required, but only {len(X)} provided.")
            
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n{'='*50}")
            print(f"Training Fold {fold}/{n_folds}")
            print(f"{'='*50}")
            
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            # Balance training data
            print("\nBalancing training data...")
            X_train_reshaped = X_train_fold.reshape(len(X_train_fold), -1)
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train_fold)
            X_train_fold = X_train_resampled.reshape(-1, self.image_size[0], self.image_size[1], 3)
            y_train_fold = y_train_resampled
            
            # Train fold
            model, history = self.train_fold(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                fold, epochs
            )
            
            # Evaluate fold
            val_dataset = self.create_dataset(X_val_fold, y_val_fold, is_training=False)
            scores = model.evaluate(val_dataset, verbose=1)
            fold_scores.append(scores[2])  # F1 score
            
            # Add model to ensemble
            self.models.append(model)
            
            # Clear memory
            del model
            gc.collect()
            tf.keras.backend.clear_session()
        
        print("\nCross-validation Results:")
        print(f"F1 Scores: {[f'{score:.4f}' for score in fold_scores]}")
        print(f"Mean F1: {np.mean(fold_scores):.4f} (±{np.std(fold_scores):.4f})")
        
        return fold_scores

    def ensemble_predict(self, X_test, threshold=0.5):
        """Generate ensemble predictions with test-time augmentation"""
        print("\nGenerating ensemble predictions with test-time augmentation...")
        
        # Create augmented versions of test data
        augmented_datasets = [
            self.create_dataset(X_test, is_training=False)
        ]
        
        # Add augmented versions
        for _ in range(4):  # 4 augmented versions
            aug_data = np.array([
                self.augmentation(image.numpy(), training=True)
                for image in tf.data.Dataset.from_tensor_slices(X_test)
            ])
            augmented_datasets.append(self.create_dataset(aug_data, is_training=False))
        
        # Generate predictions
        all_predictions = []
        
        for model in self.models:  # For each model in ensemble
            model_predictions = []
            
            for dataset in augmented_datasets:  # For each augmented version
                preds = model.predict(dataset, verbose=1)
                model_predictions.append(preds)
            
            # Average predictions for this model
            avg_model_preds = np.mean(model_predictions, axis=0)
            all_predictions.append(avg_model_preds)
        
        # Final ensemble predictions
        final_predictions = np.mean(all_predictions, axis=0)
        final_classes = (final_predictions > threshold).astype(int)
        
        return final_predictions, final_classes

def main():
    """Main execution pipeline"""
    print("\n" + "="*50)
    print("Starting High-Performance Image Classification Pipeline")
    print("="*50)
    
    # Configuration
    BASE_PATH = "/kaggle/input/ai-vs-human-generated-dataset/"
    IMAGE_SIZE = (299, 299)  # Larger image size
    BATCH_SIZE = 16
    N_FOLDS = 5
    EPOCHS = 100
    
    try:
        # Load data
        print("\nLoading data...")
        train_df = pd.read_csv("/kaggle/input/detect-ai-vs-human-generated-images/train.csv")
        test_df = pd.read_csv("/kaggle/input/detect-ai-vs-human-generated-images/test.csv")
        
        # Initialize classifier
        classifier = HighPerformanceClassifier(
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE
        )
        
        # Load and preprocess data
        X, y, X_test, test_ids = classifier.load_data(BASE_PATH, train_df, test_df)
        
        # Train with k-fold cross validation
        fold_scores = classifier.train_with_folds(X, y, N_FOLDS, EPOCHS)
        
        # Generate ensemble predictions
        predictions, pred_classes = classifier.ensemble_predict(X_test)
        
        # Save predictions
        submission_df = pd.DataFrame({
            'id': test_ids,
            'label': pred_classes.flatten()
        })
        submission_df.to_csv('final_submission.csv', index=False)
        
        print("\nPipeline completed successfully!")
        print(f"Mean F1 Score across folds: {np.mean(fold_scores):.4f}")
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}")
        raise
    finally:
        print("\nExecution completed.")

if __name__ == "__main__":
    # Configure GPU memory growth
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("\nGPU memory growth enabled")
            print(f"Available GPUs: {len(physical_devices)}")
        else:
            print("\nNo GPU devices found, using CPU")
    except Exception as e:
        print(f"\nError configuring GPU: {str(e)}")
    
    # Set random seeds for reproducibility
    np.random.seed(122)
    tf.random.set_seed(122)
    print("\nRandom seeds set for reproducibility")
    
    # Start main execution with timing
    start_time = time.time()
    try:
        main()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time/60:.2f} minutes")
    except Exception as e:
        print(f"\nExecution failed: {str(e)}")
        raise

