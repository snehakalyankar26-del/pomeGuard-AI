import os
import glob
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Default dataset path can be overridden with environment variable DATASET_DIR
DATASET_DIR = os.environ.get("DATASET_DIR", r"dataset")

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found at '{DATASET_DIR}'.")
        print("Please set the DATASET_DIR environment variable or place dataset in './dataset'.")
        return

    print("Loading class names...")
    class_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    print(f"Classes ({len(class_names)}): {class_names}")
    
    # Save class names mapping for the app
    joblib.dump(class_names, "class_names.pkl")
    print("Saved class_names.pkl")
    
    print("Loading PyTorch MobileNetV2 feature extractor...")
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    base_model = models.mobilenet_v2(weights=weights)
    feature_extractor = base_model.features
    feature_extractor.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features_list = []
    labels_list = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    feature_extractor = feature_extractor.to(device)
    
    print("Extracting features from dataset images...")
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(DATASET_DIR, class_name)
        img_paths = glob.glob(os.path.join(class_dir, "*.*"))
        print(f"Processing '{class_name}': {len(img_paths)} images")
        
        for img_path in img_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                input_tensor = preprocess(img)
                input_batch = input_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    feat = feature_extractor(input_batch)
                    feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                    feat = torch.flatten(feat, 1)
                
                features_list.append(feat.cpu().numpy().flatten())
                labels_list.append(label_idx)
            except Exception as e:
                print(f"Skipping damaged image {img_path}: {e}")
                
    X = np.array(features_list)
    y = np.array(labels_list)
    print(f"Feature extraction complete! Matrix shape: {X.shape}, Labels: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models_to_train = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Classifier": SVC(probability=True, kernel="linear", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42)
    }
    
    best_acc = 0.0
    best_model_name = ""
    best_model = None
    
    print("\nTraining candidate models...")
    for name, clf in models_to_train.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"  -> {name} Accuracy: {acc * 100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = clf
            
    print(f"\nBest Model: {best_model_name} with {best_acc * 100:.2f}% Accuracy!")
    joblib.dump(best_model, "best_model.pkl")
    print("Saved best_model.pkl successfully!")

if __name__ == "__main__":
    main()
