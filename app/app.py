import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import requests
from io import BytesIO

# Load the pre-trained chest X-ray classification model
try:
    model = load_model('D:/1-brain_insipred/cv/chest_xray/models/chest_xray_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define the class names for prediction output
class_names = ['Normal', 'Pneumonia']

# Prediction class for encapsulating the prediction logic
class Prediction:
    def __init__(self, model):
        self.model = model

    def classify_image(self, image):
        try:
            # Convert and standardize the input image
            image = Image.fromarray(image).convert('RGB')
            image = image.resize((512, 512))  # Resize to model's input dimensions
            image_array = np.array(image).astype(np.float32) / 255.0  # Normalize pixel values
            image_array = np.expand_dims(image_array, axis=0)

            # Generate predictions
            predictions = self.model.predict(image_array)[0]
            predicted_class_idx = np.argmax(predictions)
            predicted_class = class_names[predicted_class_idx]
            predicted_confidence = predictions[predicted_class_idx] * 100

            return predicted_class, predicted_confidence, predictions
        except Exception as e:
            st.error(f"Error during classification: {e}")
            return None, None, None

# Initialize the Prediction class
predictor = Prediction(model)

# Streamlit app layout
st.title("📊 Chest X-Ray Classification")
st.markdown(
    """
    Upload one or more chest X-ray images or provide an image URL, and the model will classify each as either **Normal** or **Pneumonia**.
    """
)

# Input option selection
input_option = st.radio("Choose how to upload the image(s):", ("Upload Image(s)", "Image URL"))

# Initialize images list
images = []

# Patient name input
patient_name = st.text_input("### Patient Name")

if input_option == "Upload Image(s)":
    uploaded_images = st.file_uploader("### Step 1: Upload Your Chest X-Ray Image(s)", type=["jpg", "jpeg", "png"],
                                       accept_multiple_files=True)
    if uploaded_images:
        for uploaded_image in uploaded_images:
            try:
                image = np.array(Image.open(uploaded_image))
                images.append((image, uploaded_image.name))  # Store image with its file name
            except Exception as e:
                st.error(f"Error loading image: {e}")

elif input_option == "Image URL":
    image_url = st.text_input("### Step 1: Enter the Image URL")
    if image_url:
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                images.append((np.array(Image.open(BytesIO(response.content))), image_url))  # Store image with URL
                st.markdown(f"[Image URL]({image_url})")  # Make the URL a hyperlink
            else:
                st.error("Error fetching image from URL: Unable to retrieve the image.")
        except Exception as e:
            st.error(f"Error fetching image from URL: {e}")

# Store classification results
results = []

# Display submit button after images are uploaded
if images:
    submit_button = st.button("Submit", key="submit")
    if submit_button:
        st.write("### Step 2: Review the Uploaded Image(s) and Results")

        for idx, (image, image_name) in enumerate(images):
            # Use Markdown to create a hyperlinked text for the patient name
            patient_display_name = f"[{patient_name}](#{image_name})"  # Create a hyperlink
            st.write(f"#### Patient: {patient_display_name}")

            # Display the image
            col1, col2 = st.columns([2, 1])
            with col1:
                # Resize the image to make it smaller
                st.image(image, caption=image_name, use_column_width=True, clamp=True)

            with col2:
                # Run classification and display results
                st.subheader("Prediction Results")
                with st.spinner("Processing..."):
                    predicted_class, predicted_confidence, predictions = predictor.classify_image(image)

                if predicted_class is not None:
                    # Organized output with styled boxes
                    st.markdown(f"""
                    <div style="border: 2px solid #2196F3; padding: 10px; border-radius: 5px;">
                        <p style="font-size: 16px; font-weight: bold;">Predicted Class: {predicted_class}</p>
                        <p style="font-size: 16px;">Confidence: {predicted_confidence:.2f}%</p>
                        <p style="font-size: 16px; font-weight: bold;">Class Confidence Levels:</p>
                        <ul style="list-style-type: none; padding: 0;">
                            <li style="color: #4CAF50;">Normal: {predictions[0] * 100:.1f}%</li>
                            <li style="color: #F44336;">Pneumonia: {predictions[1] * 100:.1f}%</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    # Store the results for the report
                    results.append((idx + 1, patient_name, predicted_class, predicted_confidence, predictions))

                    # Add a line for separation
                    st.markdown("---")

        # Show the button to start again after predictions
        if st.button("Start Again"):
            images = []  # Clear the images list
            st.experimental_rerun()  # Rerun the app to refresh the state

        # Additional Information section after predictions
        st.write("### Additional Information")
        st.markdown(
            """
            - This model is trained to differentiate between normal and pneumonia-affected chest X-rays.
            - Confidence levels are displayed as a percentage for each class.
            """
        )
