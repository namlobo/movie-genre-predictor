
# import streamlit as st
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import torch.nn as nn
# import numpy as np

# # -----------------------------
# # PAGE CONFIG
# # -----------------------------
# st.set_page_config(
#     page_title="Movie Genre Classifier",
#     page_icon="🎬",
#     layout="centered"
# )

# # -----------------------------
# # SETUP
# # -----------------------------
# NUM_CLASSES = 20  # Update to match your number of genres
# MODEL_PATH = r"C:\Users\nikita\Downloads\best_resnet34_movie_genre.pth"
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# GENRE_NAMES = [
#     "Action", "Adventure", "Animation", "Biography", "Comedy",
#     "Crime", "Documentary", "Drama", "Family", "Fantasy","Film-Noir",
#     "History", "Horror", "Musical", "Mystery", "Romance",
#     "Sci-Fi", "Thriller", "War", "Western"
# ]

# # -----------------------------
# # LOAD MODEL (cached for efficiency)
# # -----------------------------
# @st.cache_resource
# def load_model():
#     model = models.resnet34(pretrained=False)
#     num_features = model.fc.in_features
#     model.fc = nn.Linear(num_features, NUM_CLASSES)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model = model.to(DEVICE)
#     model.eval()
#     return model

# model = load_model()

# # -----------------------------
# # IMAGE PREPROCESSING
# # -----------------------------
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                          std=[0.229, 0.224, 0.225])
# ])

# # -----------------------------
# # PREDICTION FUNCTION
# # -----------------------------
# def predict_genre(image):
#     """Predict genre from uploaded image"""
#     img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
#     with torch.no_grad():
#         outputs = model(img_tensor)
#         probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
#     # Get top 5 predictions
#     top5_prob, top5_idx = torch.topk(probabilities, 5)
    
#     results = []
#     for i in range(5):
#         genre = GENRE_NAMES[top5_idx[i].item()]
#         prob = top5_prob[i].item() * 100
#         results.append((genre, prob))
    
#     return results

# # -----------------------------
# # STREAMLIT UI
# # -----------------------------
# st.title(" Movie Genre Classifier")
# st.write("Upload a movie poster to predict its genres")

# # File uploader
# uploaded_file = st.file_uploader(
#     "Choose a movie poster image...", 
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file is not None:
#     # Display uploaded image
#     image = Image.open(uploaded_file).convert('RGB')
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.image(image, caption='Uploaded Poster', use_container_width=True)

    
#     with col2:
#         # Make prediction
#         with st.spinner('Analyzing poster...'):
#             predictions = predict_genre(image)
        
#         st.success('Prediction Complete!')
        
#         # Display results
#         st.subheader("Top 5 Predicted Genres:")
        
#         for i, (genre, prob) in enumerate(predictions, 1):
#             st.write(f"**{i}. {genre}**: {prob:.2f}%")
#             st.progress(prob / 100)
        
#         # Display top prediction prominently
#         st.markdown("---")
#         top_genre, top_prob = predictions[0]
#         st.metric(
#             label="Most Likely Genre",
#             value=top_genre,
#             delta=f"{top_prob:.1f}% confidence"
#         )

# else:
#     st.info(" Please upload a movie poster image to get started!")
    
# # Footer
# st.markdown("---")
# st.markdown("ResNet34 Model")
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as load_keras_model # Renamed to avoid conflict
import os

# -----------------------------
# PAGE CONFIG & THEME SETUP
# -----------------------------
st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="centered"
)

# --- Inject custom CSS for dark theme look and feel ---
st.markdown("""
<style>
.main {
    background-color: #1e1e1e;
    color: #f0f0f0;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #bb86fc; /* Purple accent for headers */
}
.stButton>button {
    color: #000000;
    background-color: #bb86fc; 
    border-radius: 5px;
}
.stProgress > div > div > div > div {
    background-color: #03dac6; /* Teal accent for progress bar */
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# SETUP & CONSTANTS
# -----------------------------
# Based on the multi-label context
NUM_CLASSES = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- MODEL PATHS (Relative to the app.py root directory) ---
RESNET_MODEL_PATH = "models/best_resnet34_movie_genre.pth"
CUSTOM_CNN_MODEL_PATH = "models/best_single_genre_model.h5"

# --- Genre List (Must match the index order of your models) ---
GENRE_NAMES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy",
    "Crime", "Documentary", "Drama", "Family", "Fantasy", 
    "Film-Noir", "History", "Horror", "Music", "Musical", 
    "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", 
    "War", "Western"
]
#assert len(GENRE_NAMES) == NUM_CLASSES, "Genre list must match NUM_CLASSES"
# -----------------------------
# LOAD MODELS (Cached for efficiency)
# -----------------------------
@st.cache_resource
def load_pytorch_model():
    """Loads the PyTorch ResNet model (Model B)."""
    if not os.path.exists(RESNET_MODEL_PATH):
        st.error(f"PyTorch model not found at: {RESNET_MODEL_PATH}")
        return None
        
    model = models.resnet34(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_keras_model_cached():
    """Loads the Keras/TensorFlow Custom CNN model (Model A)."""
    if not os.path.exists(CUSTOM_CNN_MODEL_PATH):
        st.error(f"Keras model not found at: {CUSTOM_CNN_MODEL_PATH}")
        return None
    try:
        # Load model, compile=False is necessary due to custom training metrics/losses
        model = load_keras_model(CUSTOM_CNN_MODEL_PATH, compile=False) 
        return model
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        return None

# Load all models once
resnet_model = load_pytorch_model()
custom_cnn_model = load_keras_model_cached()

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
# PyTorch Transform (ResNet input size)
pt_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Keras Transform (Custom CNN input size: 100x150)
def keras_preprocess(image):
    image = image.resize((100, 150))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array


# -----------------------------
# PREDICTION FUNCTION (Multi-Label Sigmoid Output)
# -----------------------------
def predict_genre(image, model_choice):
    """
    Predicts movie genres using the selected model.

    Args:
        image (PIL.Image): Uploaded poster image.
        model_choice (str): Either "ResNet34 (PyTorch)" or "Custom CNN (Keras)".

    Returns:
        results (list of tuples): Top 5 predicted genres and their probabilities as Python floats.
        probabilities (numpy array): Full probability array from the model.
    """
    probabilities = None

    # --- PyTorch ResNet ---
    if model_choice == "ResNet34 (PyTorch)":
        if not resnet_model:
            return None, None
        
        # Preprocess image and move to device
        img_tensor = pt_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = resnet_model(img_tensor)
            # Multi-label probability using sigmoid
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

    # --- Keras Custom CNN ---
    elif model_choice == "Custom CNN (Keras)":
        if not custom_cnn_model:
            return None, None
        
        img_array = keras_preprocess(image)
        probabilities = custom_cnn_model.predict(img_array, verbose=0)[0]

    if probabilities is None:
        return None, None

    # --- Get Top 5 Predictions ---
    top5_indices = np.argsort(probabilities)[::-1][:5]
    top5_prob = probabilities[top5_indices]

    results = []
    for i in range(5):
        genre = GENRE_NAMES[top5_indices[i]]
        prob = float(top5_prob[i] * 100)  # <- cast to Python float for Streamlit
        results.append((genre, prob))

    return results, probabilities


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎬 Movie Genre Classifier")
st.write("Analyze a movie poster using either the Custom CNN or the ResNet34 model.")

# --- Model Selector Sidebar ---
st.sidebar.header("Model Selection")
model_options = ["Custom CNN (Keras)", "ResNet34 (PyTorch)"]
selected_model = st.sidebar.radio("Choose Architecture:", model_options)

# Display model status
if (selected_model == "Custom CNN (Keras)" and custom_cnn_model is None) or \
   (selected_model == "ResNet34 (PyTorch)" and resnet_model is None):
    st.sidebar.error("Selected model failed to load. Check 'models/' directory.")
else:
    st.sidebar.success(f"{selected_model} loaded successfully.")


# File uploader
uploaded_file = st.file_uploader(
    "Choose a movie poster image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.image(image, caption='Uploaded Poster', use_container_width=True)
    
    with col2:
        # Make prediction
        with st.spinner(f'Analyzing poster with {selected_model}...'):
            predictions, _ = predict_genre(image, selected_model)
        
        if predictions:
            st.success('Prediction Complete!')
            
            st.subheader("Top Predicted Genres:")
            
            top_genre, top_prob = predictions[0]
            
            # Display top prediction prominently
            st.metric(
                label=f"Most Likely Genre ({selected_model})",
                value=top_genre,
                delta=f"{top_prob:.1f}% confidence"
            )

            st.markdown("---")
            st.write("---Detailed Breakdown---")

            # Display remaining top predictions
            for i, (genre, prob) in enumerate(predictions, 1):
                st.write(f"**{i}. {genre}**")
                st.progress(prob / 100, text=f"{prob:.2f}%")
        else:
            st.warning("Could not make a prediction. Check model files.")

else:
    st.info(" Please upload a movie poster image to get started!")
    
# Footer
st.markdown("---")
st.markdown("Single-Label Classifier (Sigmoid Output)")