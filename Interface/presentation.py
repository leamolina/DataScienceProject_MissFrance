import streamlit as st


def page_presentation():

    st.title('Qui sera Miss France 2025 ? :crown:')

    chemin_video = "./Sources/Video_avec_audio.mp4"
    st.video(chemin_video, format='video/mp4', start_time=0)

    st.subheader('Notre projet : ', divider='orange')
    st.write('Bienvenue dans notre projet de prédiction de Miss France 2025 ! :crown:\n\n Nous nous sommes lancées '
             'dans ce projet passionnant car nous nous réunissons depuis des années le soir de l’émission pour '
             'discuter, débattre et faire nos pronostics. \n\nC\'est devenu une tradition chère à nos cœurs, '
             'un moment de partage et de convivialité que nous attendons avec impatience.'
             '\n\n Notre but : '
             'Déterminer QUI sera Miss France 2025 et quelles candidates feront parties du top 12 ?')

    st.subheader('En quoi consiste notre modèle ?', divider='orange')
    st.write('Notre modèle tente de prédire avec précision le top 12 des candidates. Grâce à notre approche '
             'méthodique et à l\'analyse minutieuse d\'un vaste ensemble de données, nous essayons de fournir des '
             'prédictions aussi fiables que possible.\n\nNos données proviennent d\'une variété de sources sur '
             'Internet, y compris des sites officiels, et des articles de presse. Nous avons collecté des '
             'informations sur les caractéristiques des candidates telles que leur région d\'origine, leur âge, '
             'leur taille, et bien plus encore. \n\nL\'objectif est de créer un modèle robuste et complet '
             'nous permettant d’obtenir un score de '
             'prédiction pour chacune des candidates et déterminer leur classement final. \n\nN\'hésitez pas à '
             'cliquer sur la rubrique PRÉDICTION pour découvrir qui sont les candidates les plus prometteuses selon '
             'notre analyse approfondie des données !')
    st.write('\n\n\n\nPour cette 95ème élection, ce sont 30 candidates âgées de 18 à 28 ans qui sont en lice pour la '
             'couronne de Miss France 2025.')
