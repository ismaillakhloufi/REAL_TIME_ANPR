
from PIL import Image
import streamlit as st


st.title('ANPR WITH YOLO V8')
image = Image.open('plates.jpg')

st.image(image, caption='Image Caption')

st.subheader('Greetin gs')


st.write("")

st.markdown("<span style='font-size:20px;'>This project consists of using YOLO V8 for  Automatic Moroccan Number plate Recognition."
            "</span>", unsafe_allow_html=True)

st.markdown("<span style='font-size:20px;'>You can upload an image of a car to detect it's Moroccan license plate number or a video where all vehicules will be tracked and their license plates detected and saved in a csv file"
            "</span>", unsafe_allow_html=True)


st.markdown("""
<ul style='font-size: 18px; font-weight:bold;'>
  <li>🙋‍♂ <a href="https://www.linkedin.com/in/ismaillakhloufi/" target="_blank" style="text-decoration: none; color: black;">ISMAIL LAKHLOUFI</a></li>
  <li>💾 <a href="https://github.com/ismaillakhloufi/" target="_blank" style="text-decoration: none; color: black;">Github</a> </li>
  <li> </li>
</ul>
""", unsafe_allow_html=True)





