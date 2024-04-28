import streamlit as st #Pour ce faire : pip install streamlit & pip install streamlit-vis-timeline (pour la frise chronologique) & !pip install streamlit_timeline
import pandas as pd
#from streamlit_timeline import timeline
import plotly.figure_factory as ff
import numpy as np
import pydeck as pdk




# Fonction pour afficher la page d'accueil
def accueil():
    # Récupération des données
    data = pd.read_csv(chemin_database)
    # data = pd.read_csv(chemin_ana)
    st.title("MISS FRANCE 2025")
    st.image(chemin_logo, use_column_width=False, output_format='auto', width=100, clamp=False, channels='RGB')
    st.write("Bienvenu dans notre projet  \nNous allons essayer de prédire la Miss France 2025")

    st.audio(chemin_audio)
    st.video(chemin_video, format="video/mp4", start_time=0)


def page_un():
    st.title("Nos données")
    st.write("Ici on va présenter notre Dataset (histogrammes, graphes,  ...) ")

    #Affichage frise chronologique (tentative)
    with open(f'/Sources/timeline.json', "r") as f:
        data = f.read()

    #timeline(data, height=600)


    #Tentarive de map
    france_coords = [
        (48.8566, 2.3522),  # Paris
        (43.6045, 1.4442),  # Toulouse
        (45.7640, 4.8357),  # Lyon
        (43.2965, 5.3698),  # Marseille
        (48.1173, -1.6778),  # Rennes
        (47.2184, -1.5536),  # Nantes
        (44.8374, -0.5761),  # Bordeaux
        (49.2578, 4.0319),  # Reims
        (48.5734, 7.7521),  # Strasbourg
        (48.8566, 2.3522)  # Revenir à Paris pour fermer la boucle
    ]

    # Créer le DataFrame avec les données
    chart_data = pd.DataFrame(france_coords, columns=['lat', 'lon'])
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=46.603354,
            longitude=1.888334,
            zoom=5,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'HexagonLayer',
                data=chart_data,
                get_position='[lon, lat]',
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),

        ],
    ))


def prediction():
    st.title("Prédictions")
    st.write("Ici on va prédire notre future Miss France.")


st.sidebar.image(chemin_logo, width=100)
st.sidebar.markdown(
    "<div style='text-align: center;'><h1>Navigation</h1></div>",
    unsafe_allow_html=True
)
page = st.sidebar.radio(
    "",
    ("Accueil", "Prédictions")
)

# Affichage de la page sélectionnée
if page == "Accueil":
    accueil()
elif page == "Prédictions":
    prediction()
