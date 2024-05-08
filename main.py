import pandas as pd
import streamlit as st

from Interface.data import page_data
from Interface.prediction import page_prediction
from Interface.presentation import page_presentation
from Interface.evaluation import page_evaluation



st.sidebar.title('Projet')

pages = {
    'Présentation': page_presentation,
    'Nos données': page_data,
    'Evaluation du modèle' : page_evaluation,
    'Prédiction': page_prediction,
}

choice_page = st.sidebar.radio('Aller à', list(pages.keys()))
pages[choice_page]()