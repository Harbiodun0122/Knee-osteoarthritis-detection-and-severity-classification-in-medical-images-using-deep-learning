import cv2
import glob
import torch
import matplotlib
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from tensorflow.keras.models import Model
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from tensorflow.keras.preprocessing import image
from detectron2.utils.visualizer import ColorMode, Visualizer
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load your trained models
kidney_model = tf.keras.models.load_model("kidney_best.h5")
# knee_model = tf.keras.models.load_model(r"C:/Users/Harbiodun/Downloads/Assignment/Project/improved-48-0.84.keras")


# # Load the Configuration and Model
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 
# cfg.MODEL.WEIGHTS = r"C:/Users/Harbiodun/Downloads/Assignment/Project/model_final.pth"
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # Set a threshold for the detection confidence as 90%
# cfg.MODEL.DEVICE = "cpu"  # Use CPU if no GPU is available
# predictor = DefaultPredictor(cfg)


# Define the Streamlit app
def main():
    st.set_page_config(page_title="Medical Image Classification", page_icon="🩺")
    st.title("Medical Image Classification")
    st.write("Upload images to classify diseases.")

    # Create tabs
    tabs = st.tabs(["Kidney Disease Classification", "Pneumonia Classification", "Knee Osteoarthritis Classification"])

    with tabs[0]:
        kidney_disease_classification()

    # with tabs[1]:
    #     knee_osteoarthritis_classification()

def kidney_disease_classification():
    st.header("Kidney Disease Classification")
    kidney_image = st.file_uploader("Upload a kidney image", type=["jpg", "png", "jpeg"], key="kidney")
    if kidney_image is not None:
        kidney_img = image.load_img(kidney_image, target_size=(200, 200),color_mode='grayscale')
        kidney_img = image.img_to_array(kidney_img)
        kidney_img = np.expand_dims(kidney_img, axis=0)
        kidney_img = kidney_img / 255.0
        
        kidney_col1, kidney_col2 = st.columns(2)
        with kidney_col1:
            st.image(kidney_img, caption="Uploaded Kidney Image", width=500)

        with kidney_col2:
            try:
                # with st.spinner('Classifying...'):
                kidney_prediction = kidney_model.predict(kidney_img)
                kidney_class_labels = ['Cyst', 'Normal','Stone','Tumor']
                predicted_kidney_class = kidney_class_labels[np.argmax(kidney_prediction)]
                kidney_confidence = kidney_prediction[0][np.argmax(kidney_prediction)]

                # Use HTML for right-aligned text
                st.markdown(
                    f"""
                    <div style="text-align: right;">
                        <p><strong>Prediction:</strong> {predicted_kidney_class}</p>
                        <p><strong>Confidence:</strong> {round(kidney_confidence * 100, 2)}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error in classification: {e}")


# def knee_osteoarthritis_classification():
#     st.header("Knee Osteoarthritis Classification")
#     knee_image = st.file_uploader("Upload a knee image", type=["jpg", "png", "jpeg"], key="knee")
#     if knee_image is not None:        
#         im = image.load_img(knee_image)
#         fixed_size_image = im.resize((500, 500))
#         im = np.array(im)
#         st.image(fixed_size_image, caption="Uploaded knee image", width=500)
        
#         knee_col1, knee_col2 = st.columns(2)
#         with knee_col1:
#             try:
#                 with st.spinner('Extracting knee joint...'):
#                     outputs = predictor(im)
                
#                     # Extract bounding box coordinates
#                     boxes = outputs["instances"].pred_boxes
                    
#                     # Assuming only one knee joint is detected, extract the first bounding box
#                     box = boxes[0].tensor.cpu().numpy()[0]  # [x1, y1, x2, y2]
                    
#                     # Crop the image using the bounding box coordinates
#                     x1, y1, x2, y2 = map(int, box)
#                     cropped_image = im[y1:y2, x1:x2]

#                     st.image(cropped_image, caption="Knee joint detected and extracted", width=500)

#             except Exception as e:
#                 st.error(f"Error detecting knee joint: {e}")
        
#         with knee_col2:
#             try:
#                 with st.spinner('Classifying...'):
#                     cropped_image = tf.image.resize(cropped_image, (160, 335))
#                     cropped_image = np.expand_dims(cropped_image / 255, 0)
#                     knee_prediction = knee_model.predict(cropped_image)
#                     knee_class_labels = { 0 : "Normal", 1 : "Doubtful", 2 : "Mild", 3 : "Moderate", 4 : "Severe"}
#                     predicted_knee_class = np.argmax(knee_prediction, axis=1)
#                     if predicted_knee_class[0] in knee_class_labels:
#                         st.markdown(
#                             f"""
#                             <div style="text-align: right;">
#                                 <p>Prediction: {knee_class_labels[predicted_knee_class[0]]}</p>
#                             </div>
#                             """,
#                             unsafe_allow_html=True
#                         )
#                     osteoarthritis_confidence = knee_prediction[0][np.argmax(knee_prediction)]
#                     st.markdown(
#                             f"""
#                             <div style="text-align: right;">
#                                 <p>Confidence: {round(osteoarthritis_confidence * 100, 2)}%</p>
#                             </div>
#                             """,
#                             unsafe_allow_html=True
#                         )

#             except Exception as e:
#                 st.error(f"Error in classification: {e}")

        

if __name__ == "__main__":
    main()
