import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image

def get_image(image_path):
  image = Image.open(image_path)

  return np.array(image)

def get_hex(color):
  return f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"

st.set_page_config(
  page_title="Colour Detector",
  layout="wide"
)

hide_default_format = """
      <style>
      #MainMenu {visibility: hidden; }
      footer {visibility: hidden;}
      </style>
      """
st.markdown(hide_default_format, unsafe_allow_html=True)

@st.cache
def get_colors(path="https://raw.githubusercontent.com/codebrainz/color-names/master/output/colors.csv"):
  df = pd.read_csv(path, names=["color_name", "hex", "r", "g", "b"])
  return df

data = get_colors()

st.title("Colour Detector")

image = st.file_uploader("Upload image", type=['png', 'jpg'])

if image:
  left, right = st.columns(2)

  with left:
    st.image(image)
    image_arr = get_image(image)

    number_of_colours = st.slider(
      "How many colors should be extracted?", 0, 5, 3)

    mod_image = image_arr.reshape(
      image_arr.shape[0] * image_arr.shape[1], image_arr.shape[-1]
      )


  with right:
    clf = KMeans(n_clusters = number_of_colours)

    with st.spinner("Detecting Colors..."):
      labels = clf.fit_predict(mod_image)
    labels = Counter(labels)

    colors = clf.cluster_centers_
    hex = [get_hex(colors[i]) for i in labels]
    for color in hex:
      string = """
          <div class="color" style="background-color:{};height:60px;width:60px;margin:25px;border-radius:6px;border:black 3px">
          </div>""".format(color)

      st.markdown(string, unsafe_allow_html=True)

      if color in data['hex']:
        st.code(f"{color}: {data[data['hex'] == color]['color_name']}")
      else:
        st.code(color)