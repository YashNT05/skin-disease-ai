import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# CONFIGURATION & MEDICAL DB
# ==========================================
IMG_SIZE = (224, 224)
MODEL_PATH = 'medical_skin_model.h5'
LABELS_PATH = 'class_indices.npy'

# Same medical database as before
TREATMENT_DB = {
    'acne': {
        'condition': 'Acne Vulgaris',
        'desc': 'Clogged hair follicles under the skin.',
        'otc': 'Benzoyl Peroxide gel, Salicylic Acid cleanser, Adapalene.',
        'urgency': 'Low - Treat at home',
        'color': 'green'
    },
    'eczema': {
        'condition': 'Eczema / Atopic Dermatitis',
        'desc': 'Inflammation causing itchy, red, swollen skin.',
        'otc': 'Hydrocortisone cream (1%), Moisturizers (Cerave/Cetaphil).',
        'urgency': 'Low/Medium',
        'color': 'blue'
    },
    'tinea': {
        'condition': 'Tinea (Ringworm)',
        'desc': 'Fungal infection causing a ring-shaped rash.',
        'otc': 'Antifungal creams: Clotrimazole, Terbinafine.',
        'urgency': 'Low - Highly contagious',
        'color': 'orange'
    },
    'ringworm': {
        'condition': 'Tinea (Ringworm)',
        'desc': 'Fungal infection causing a ring-shaped rash.',
        'otc': 'Antifungal creams: Clotrimazole, Terbinafine.',
        'urgency': 'Low - Highly contagious',
        'color': 'orange'
    },
    'wart': {
        'condition': 'Viral Warts',
        'desc': 'Small, grainy skin growth caused by HPV.',
        'otc': 'Salicylic Acid pads, Cryotherapy freeze kits.',
        'urgency': 'Low',
        'color': 'green'
    },
    'melanoma': {
        'condition': 'Melanoma (Skin Cancer)',
        'desc': 'The most serious type of skin cancer.',
        'otc': 'NONE. DO NOT TREAT AT HOME.',
        'urgency': 'CRITICAL - SEE DOCTOR IMMEDIATELY',
        'color': 'red'
    },
    'basal': {
        'condition': 'Basal Cell Carcinoma',
        'desc': 'Common form of skin cancer.',
        'otc': 'None. Requires surgical removal.',
        'urgency': 'High - See Dermatologist',
        'color': 'red'
    },
    'keratosis': {
        'condition': 'Actinic Keratosis',
        'desc': 'Rough, scaly patch from sun exposure.',
        'otc': 'Sunscreen (Prevention). Treatment requires doctor.',
        'urgency': 'Medium',
        'color': 'orange'
    }
}

# ==========================================
# APP LOGIC
# ==========================================

st.set_page_config(page_title="AI Dermatologist", page_icon="🩺")

@st.cache_resource
def load_model_resources():
    """Loads the model once and keeps it in memory"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        classes = np.load(LABELS_PATH)
        return model, classes
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def get_advice(predicted_class):
    pred_lower = predicted_class.lower()
    for key, info in TREATMENT_DB.items():
        if key in pred_lower:
            return info
    return {
        'condition': predicted_class,
        'desc': 'Skin condition identified by AI.',
        'otc': 'Consult a pharmacist.',
        'urgency': 'Unknown',
        'color': 'gray'
    }

# ==========================================
# UI LAYOUT
# ==========================================
st.title("🩺 AI Skin Health Assistant")
st.markdown("Upload a photo of a skin condition to get an instant analysis and OTC recommendations.")
st.warning("⚠️ DISCLAIMER: This is an AI tool, not a doctor. Always consult a professional.")

# Load resources
model, class_names = load_model_resources()

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Layout: Image on Left, Results on Right
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        with st.spinner('Analyzing pixels...'):
            # Process Image
            image = Image.open(uploaded_file)
            image = image.resize(IMG_SIZE)
            img_array = np.array(image)
            
            # Fix: Convert RGBA (png) to RGB (jpg) if needed
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
                
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            # Predict
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            top_idx = np.argmax(score)
            confidence = 100 * np.max(score)
            predicted_label = class_names[top_idx]
            
            # Get Advice
            advice = get_advice(predicted_label)
            
            # Display Results
            st.subheader(f"Result: {advice['condition']}")
            st.progress(int(confidence))
            st.caption(f"AI Confidence: {confidence:.2f}%")
            
            # Colored Card for Advice
            color = advice['color']
            if color == 'red':
                st.error(f"**URGENCY:** {advice['urgency']}")
            elif color == 'orange':
                st.warning(f"**URGENCY:** {advice['urgency']}")
            else:
                st.success(f"**URGENCY:** {advice['urgency']}")
                
            st.info(f"**Suggested OTC Treatment:**\n\n{advice['otc']}")
            st.write(f"**Description:** {advice['desc']}")