import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
def main():
    st.set_page_config(layout="wide")
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
        image = None
        image2 = None
        with col1:
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "avif", "png"], key="img1")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                new_image = image.resize((400, 400))
                st.image(new_image, caption='Uploaded Image.', use_column_width=True)
        st.write("\t")
        with col2:
            uploaded_file2 = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "avif", "png"], key="img2")
            if uploaded_file2 is not None:
                image2 = Image.open(uploaded_file2)
                new_image2 = image2.resize((400, 400))
                st.image(new_image2, caption='Uploaded Image.', use_column_width=True)
        if st.button('Submit'):
            if image is None or image2 is None:
                st.error("Please upload both images.")
            else:
                

if __name__ == '__main__':
    main()

