import net
import torch
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import os
from face_alignment import align
import numpy as np
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

adaface_models = {
    'ir_50':"adaface_ir50_ms1mv2.ckpt",
}

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture], map_location ='cpu')['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

if __name__ == "__main__":
    image = None
    image2 = None
    try:
#         st.set_page_config(layout="wide")
        with st.sidebar:
            selected = option_menu('My Projects',

                                   ['Compare Pictures'],
                                   icons=['person'],
                                   default_index=0)
        # Diabetes Prediction Page
        if (selected == 'Compare Pictures'):
            # page title
            st.title('Comparing Pictures using ML')
            col1, col2 = st.columns(2)

            with col1:
                uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "avif", "png"], key="img1")
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    new_image = image.resize((400, 400))
                    st.image(new_image, caption='Uploaded Image.', use_column_width=True)
            with col2:
                uploaded_file2 = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "avif", "png"], key="img2")
                if uploaded_file2 is not None:
                    image2 = Image.open(uploaded_file2)
                    new_image2 = image2.resize((400, 400))
                    st.image(new_image2, caption='Uploaded Image.', use_column_width=True)
    except:
        st.error('Invalid Image', icon="🚨")
    if st.button('Submit'):
        if image is None or image2 is None:
            st.error("Please upload both images.")
        else:
            try:
                model = load_pretrained_model('ir_50')
                features = []
                # temp1 = image.convert('rgb')
                aligned_rgb_img1 = align.get_aligned_face(None, rgb_pil_image=image)
                bgr_tensor_input1 = to_input(aligned_rgb_img1)
                feature1, _ = model(bgr_tensor_input1)
                features.append(feature1)
                # temp2 = image2.convert('rgb')
                aligned_rgb_img2 = align.get_aligned_face(None, rgb_pil_image=image2)
                bgr_tensor_input2 = to_input(aligned_rgb_img2)
                feature2, _ = model(bgr_tensor_input2)
                features.append(feature2)
                similarity_scores = torch.cat(features) @ torch.cat(features).T
                similarity_scores = similarity_scores[0][1]
                result = round(similarity_scores.item(), 3)
                score = "No match"
                if result > 0.9:
                    score = "Perfect match"
                elif result >= 0.75 and result <= 0.9:
                    score = "Excellent match"
                elif result >= 0.5 and result < 0.75:
                    score = "Good match"
                match_para = '<p style="font-family:Courier; color:Black; font-size: 40px;"><b>Type of match: {match}</b></p>'.format(match=score)
                sim_para = '<p style="font-family:Courier; color:Black; font-size: 40px;"><b>Similarity Score: {sim}</b></p>'.format(sim=result)
                st.markdown(match_para, unsafe_allow_html=True)
                st.markdown(sim_para, unsafe_allow_html=True)
            except:
                st.warning("Model can't process these images")

