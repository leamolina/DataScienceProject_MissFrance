import streamlit as st

from Interface.data import page_data
from Interface.prediction import page_prediction
from Interface.presentation import page_presentation

st.sidebar.title('Projet ML/Data Science :')

pages = {
    '👑 Présentation  ': page_presentation,
    '📊 Analyse des données  ': page_data,
    '🏆 Prédictions ': page_prediction,
}

choice_page = st.sidebar.radio('', list(pages.keys()))
pages[choice_page]()
