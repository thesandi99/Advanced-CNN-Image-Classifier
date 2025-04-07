import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def load_and_preprocess_image(image_path):
    """Image ko load karo aur grayscale mein convert karo."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def divide_into_patches(img, patch_size=32):
    """Image ko chhote patches mein baanto."""
    patches = []
    for i in range(0, img.shape[0], patch_size):
        for j in range(0, img.shape[1], patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return patches

def compute_texture_feature(patch):
    """Pixel intensities ka standard deviation calculate karo as a texture feature."""
    return np.std(patch)

def classify_patches(patches, threshold=20):
    """Threshold ke basis par patches ko rich aur poor texture mein classify karo."""
    rich_patches = []
    poor_patches = []
    for patch in patches:
        feature = compute_texture_feature(patch)
        if feature > threshold:
            rich_patches.append(patch)
        else:
            poor_patches.append(patch)
    return rich_patches, poor_patches

def compute_correlation_contrast(rich_patches, poor_patches):
    """Rich aur poor texture patches ke beech inter-pixel correlation contrast compute karo."""
    def compute_correlation(patch):
        # Gray-Level Co-occurrence Matrix (GLCM) compute karo
        glcm = graycomatrix(patch, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        # GLCM se correlation property compute karo
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        return correlation
    
    rich_correlations = [compute_correlation(patch) for patch in rich_patches]
    poor_correlations = [compute_correlation(patch) for patch in poor_patches]
    
    # Rich aur poor patches ke liye average correlation compute karo
    avg_rich_corr = np.mean(rich_correlations) if rich_correlations else 0
    avg_poor_corr = np.mean(poor_correlations) if poor_correlations else 0
    
    # Contrast dono averages ke beech ka difference hoga
    contrast = abs(avg_rich_corr - avg_poor_corr)
    return contrast

def detect_ai_generated(image_path, threshold=0.01):
    """Correlation contrast ke basis par detect karo ki image AI-generated hai ya nahi."""
    img = load_and_preprocess_image(image_path)
    patches = divide_into_patches(img)
    rich_patches, poor_patches = classify_patches(patches)
    contrast = compute_correlation_contrast(rich_patches, poor_patches)
    
    # Simple threshold-based detection (yeh threshold hypothetical hai)
    if contrast > threshold:
        return "AI-Generated"
    else:
        return "Real"

# Example usage
image_path = '/kaggle/input/ai-vs-human-generated-dataset/test_data_v2/0a9c0ceb42124fdebbd1c1f8c2a6fb45.jpg'
result = detect_ai_generated(image_path)
print(f"Image detect hua hai: {result}")


######################################################

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from skimage.feature import graycomatrix, graycoprops

def load_and_preprocess_image(image_path):
    """Image ko load karo aur grayscale mein convert karo."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def divide_into_patches(img, patch_size=32):
    """Image ko chhote patches mein baanto."""
    patches = []
    for i in range(0, img.shape[0], patch_size):
        for j in range(0, img.shape[1], patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return patches

def compute_texture_feature(patch):
    """Pixel intensities ka standard deviation calculate karo as a texture feature."""
    return np.std(patch)

def classify_patches(patches, threshold=20):
    """Threshold ke basis par patches ko rich aur poor texture mein classify karo."""
    rich_patches = []
    poor_patches = []
    for patch in patches:
        feature = compute_texture_feature(patch)
        if feature > threshold:
            rich_patches.append(patch)
        else:
            poor_patches.append(patch)
    return rich_patches, poor_patches

def compute_correlation(patch):
    """Compute correlation using Gray-Level Co-occurrence Matrix (GLCM)."""
    glcm = graycomatrix(patch, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return correlation

def compute_correlation_contrast(rich_patches, poor_patches):
    """Rich aur poor texture patches ke beech inter-pixel correlation contrast compute karo."""
    rich_correlations = [compute_correlation(patch) for patch in rich_patches]
    poor_correlations = [compute_correlation(patch) for patch in poor_patches]
    
    avg_rich_corr = np.mean(rich_correlations) if rich_correlations else 0
    avg_poor_corr = np.mean(poor_correlations) if poor_correlations else 0
    
    contrast = abs(avg_rich_corr - avg_poor_corr)
    return contrast

def detect_ai_generated(image_path, threshold=0.01):
    """Correlation contrast ke basis par detect karo ki image AI-generated hai ya nahi."""
    img = load_and_preprocess_image(image_path)
    patches = divide_into_patches(img)
    rich_patches, poor_patches = classify_patches(patches)
    contrast = compute_correlation_contrast(rich_patches, poor_patches)
    
    return 1 if contrast > threshold else 0

# Define a helper function (must be top-level for multiprocessing)
def process_image(image_path):
    return detect_ai_generated(image_path, threshold=0.01)

# --- Inference Section ---
base_dir = '/kaggle/input/ai-vs-human-generated-dataset'
df_test = pd.read_csv(os.path.join(base_dir, 'test.csv'))
df_test['id'] = df_test['id'].apply(lambda x: os.path.join(base_dir, x))

# Use ProcessPoolExecutor for parallel processing. Adjust max_workers as needed.
with ProcessPoolExecutor(max_workers=4) as executor:
    predictions = list(tqdm(executor.map(process_image, df_test['id'].values),
                            total=len(df_test['id'].values),
                            desc="Inference"))

# Extract relative paths for submission
image_names = [os.path.join("test_data_v2", os.path.basename(path)) for path in df_test['id'].values]

submission_df = pd.DataFrame({
    'id': image_names,
    'label': predictions
})

submission_df.to_csv("submission.csv", index=False)
print("Submission file generated: submission.csv")
print(submission_df['label'].value_counts())




##################################################


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

def load_and_preprocess_image(image_path):
    """Image ko load karo aur grayscale mein convert karo."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray_img

def divide_into_patches(img, patch_size=32):
    """Image ko chhote patches mein baanto."""
    patches = []
    coordinates = []
    for i in range(0, img.shape[0], patch_size):
        for j in range(0, img.shape[1], patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
                coordinates.append((i, j))
    return patches, coordinates

def compute_texture_feature(patch):
    """Pixel intensities ka standard deviation calculate karo as a texture feature."""
    return np.std(patch)

def classify_patches(patches, coordinates, threshold=20):
    """Threshold ke basis par patches ko rich aur poor texture mein classify karo."""
    rich_patches = []
    poor_patches = []
    rich_coords = []
    poor_coords = []
    for patch, coord in zip(patches, coordinates):
        feature = compute_texture_feature(patch)
        if feature > threshold:
            rich_patches.append(patch)
            rich_coords.append(coord)
        else:
            poor_patches.append(patch)
            poor_coords.append(coord)
    return rich_patches, poor_patches, rich_coords, poor_coords

def visualize_steps(original, gray, rich_coords, poor_coords, patch_size=32):
    """Har step ka visualization plot karo."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Step 1: Original Image
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Step 2: Grayscale Image
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title("Grayscale Image")
    axes[1].axis("off")
    
    # Step 3: Rich vs Poor Texture Visualization
    axes[2].imshow(gray, cmap='gray')
    for coord in rich_coords:
        rect = plt.Rectangle((coord[1], coord[0]), patch_size, patch_size, edgecolor='red', linewidth=2, fill=False)
        axes[2].add_patch(rect)
    for coord in poor_coords:
        rect = plt.Rectangle((coord[1], coord[0]), patch_size, patch_size, edgecolor='blue', linewidth=2, fill=False)
        axes[2].add_patch(rect)
    axes[2].set_title("Rich (Red) vs Poor (Blue) Texture Patches")
    axes[2].axis("off")
    
    plt.show()

def detect_ai_generated(image_path, threshold=0.01):
    """Correlation contrast ke basis par detect karo ki image AI-generated hai ya nahi."""
    original, gray_img = load_and_preprocess_image(image_path)
    patches, coordinates = divide_into_patches(gray_img)
    rich_patches, poor_patches, rich_coords, poor_coords = classify_patches(patches, coordinates)
    
    visualize_steps(original, gray_img, rich_coords, poor_coords)
    
    return "AI-Generated" if len(rich_patches) > len(poor_patches) else "Real"

# Example usage
image_path = 'your_image.jpg'  # Replace with actual path
result = detect_ai_generated(image_path)
print(f"Image detect hui hai: {result}")

####################################

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from skimage.feature import graycomatrix, graycoprops

def load_and_preprocess_image(image_path):
    """Image ko load karo aur grayscale mein convert karo."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found at {image_path}")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_img
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None  # Error case mein None return karo

def divide_into_patches(img, patch_size=48, stride=22):
    """Image ko overlapping patches mein baanto."""
    patches = []
    for i in range(0, img.shape[0] - patch_size + 1, stride):
        for j in range(0, img.shape[1] - patch_size + 1, stride):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches

def compute_correlation(patch):
    """GLCM correlation compute karo."""
    glcm = graycomatrix(patch, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return correlation

def classify_patches(patches, threshold=0.1):
    """GLCM correlation ke basis par patches ko classify karo."""
    rich_patches = []  # Low correlation = rich texture
    poor_patches = []  # High correlation = poor texture
    for patch in patches:
        corr = compute_correlation(patch)
        if corr < threshold:  # Low correlation matlab zyada texture
            rich_patches.append(patch)
        else:
            poor_patches.append(patch)
    return rich_patches, poor_patches

def compute_correlation_contrast(rich_patches, poor_patches):
    """Rich aur poor patches ke beech correlation contrast compute karo."""
    rich_corrs = [compute_correlation(patch) for patch in rich_patches]
    poor_corrs = [compute_correlation(patch) for patch in poor_patches]
    
    avg_rich_corr = np.mean(rich_corrs) if rich_corrs else 0.5  # Default value agar empty hai
    avg_poor_corr = np.mean(poor_corrs) if poor_corrs else 0.5
    
    contrast = abs(avg_rich_corr - avg_poor_corr)
    return contrast

def detect_ai_generated(image_path, contrast_threshold=0.1):
    """AI-generated image detect karo."""
    img = load_and_preprocess_image(image_path)
    if img is None:
        return 0  # Agar image load nahi hui to default real label
    patches = divide_into_patches(img)
    rich_patches, poor_patches = classify_patches(patches)
    contrast = compute_correlation_contrast(rich_patches, poor_patches)
    return 1 if contrast > contrast_threshold else 0

def process_image(image_path):
    return detect_ai_generated(image_path, contrast_threshold=0.4)

# --- Inference Section ---
base_dir = '/kaggle/input/ai-vs-human-generated-dataset'
df_test = pd.read_csv(os.path.join(base_dir, 'test.csv'))
df_test['id'] = df_test['id'].apply(lambda x: os.path.join(base_dir, x))

with ProcessPoolExecutor(max_workers=4) as executor:
    predictions = list(tqdm(executor.map(process_image, df_test['id'].values),
                            total=len(df_test['id'].values),
                            desc="Inference"))

# Submission ke liye relative paths (competition format ke hisaab se adjust karo)
image_names = [os.path.join("test_data_v2", os.path.basename(path)) for path in df_test['id'].values]

submission_df = pd.DataFrame({
    'id': image_names,
    'label': predictions
})

submission_df.to_csv("submission.csv", index=False)
print("Submission file generated: submission.csv")
print(submission_df['label'].value_counts())