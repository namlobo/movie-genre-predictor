
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# SETUP
# -----------------------------
NUM_CLASSES = 20  # Update to match your number of genres
MODEL_PATH = r"C:\Users\nikita\Downloads\best_resnet34_movie_genre.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


GENRE_NAMES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy",
    "Crime", "Documentary", "Drama", "Family", "Fantasy","Film-Noir",
    "History", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]

# -----------------------------
# LOAD MODEL (cached for efficiency)
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet34(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_genre(image):
    """Predict genre from uploaded image"""
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Get top 5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    results = []
    for i in range(5):
        genre = GENRE_NAMES[top5_idx[i].item()]
        prob = top5_prob[i].item() * 100
        results.append((genre, prob))
    
    return results

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title(" Movie Genre Classifier")
st.write("Upload a movie poster to predict its genres")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a movie poster image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Poster', use_container_width=True)

    
    with col2:
        # Make prediction
        with st.spinner('Analyzing poster...'):
            predictions = predict_genre(image)
        
        st.success('Prediction Complete!')
        
        # Display results
        st.subheader("Top 5 Predicted Genres:")
        
        for i, (genre, prob) in enumerate(predictions, 1):
            st.write(f"**{i}. {genre}**: {prob:.2f}%")
            st.progress(prob / 100)
        
        # Display top prediction prominently
        st.markdown("---")
        top_genre, top_prob = predictions[0]
        st.metric(
            label="Most Likely Genre",
            value=top_genre,
            delta=f"{top_prob:.1f}% confidence"
        )

else:
    st.info(" Please upload a movie poster image to get started!")
    
# Footer
st.markdown("---")
st.markdown("ResNet34 Model")
