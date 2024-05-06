import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def page_data():
    # Logo en haut à droite
    chemin_logo = './Sources/Logo_MissFrance.png'
    col1, col2, col3 = st.columns((1, 4, 1))
    with col3:
        st.image(chemin_logo, use_column_width=True, width=10)
    st.title('Nos données')
    for _ in range(3):
        st.write('')

    # Texte de présentation de nos données
    st.write('Notre modèle s\'appuie sur différentes sources de données: ')
    st.write('- La région des candidates (source: miss.fandom.com)')
    st.write(
        '- Les caractéristiques physiques des candidates : âge, taille, couleur des yeux et des cheveux... (source: '
        'miss.fandom.com & remplissage manuel)')
    st.write(
        '- Si elle a gagné le prix de culture générale. Les candidates passent toutes un examen écrit pour évaluer '
        'leur niveau de culture générale. Celle qui obtient la meilleure note gange le prix de culture générale, '
        'un atout majeur dans la compétition. (source: Wikipédia)')
    st.write(
        '- Si elle est tombée lors du prime : Bien que cela puisse prêter à rire, les passionnés du concours savent '
        'qu\'une candidate qui tombe le soir de l\'élection a très peu de chances d’atteindre le podium. A ce jour, '
        'la seule qui y est parvenue est Indira Ampiot, Miss France 2024. (source: TikTok, Youtube, Instagram)')

    # Récupération et affichage de la base de données
    for _ in range(5): st.write('')
    chemin_database = './Databases/data_missFrance.csv'
    data_missFrance = pd.read_csv(chemin_database, delimiter=';')
    st.subheader('Notre base de données:')
    with st.expander('Cliquer ici pour voir notre base de données complète avant le pré-traitement des données'):
        st.dataframe(data_missFrance)

    # Importance du prix de culture générale
    # On veut savoir combien, par rang, combien de candidates ont gagné le prix de culture générale
    for _ in range(5): st.write('')
    st.subheader('Importance de la culture générale dans le classement :')
    st.write(
        'Comme expliqué précédement, le test de culture générale est un examen déterminant dans le classement d\'une '
        'candidate. Le jury évalue effectivement les candidates sur des questions d\'actualité, de sciences, '
        'de logique et d\'histoire. L\'histogramme ci-dessus représente le pourcentage (pour chaque rang) de '
        'candidates ayant remporté le prix de culture générale ')
    data_percent = {}
    sum = 0
    for i in range(1, 13):
        filtered_df = data_missFrance[data_missFrance['rang'] == i]
        ranked_candidates = filtered_df[filtered_df['laureat_culture_generale'] == 1]
        data_percent[i] = (len(ranked_candidates) / len(filtered_df)) * 100
        sum += data_percent[i]
    st.bar_chart(data=data_percent, color='#f63366', use_container_width=True)
    st.write('Au total, ' + str(sum) + '% des lauréates au prix de culture générale ont atteint le top 12.')
    st.write(
        'Ce prix étant attribué chaque année qu\'à une seule candidate, cela signifie que seulement 2 lauréates au '
        'prix de culture générale n\'ont pas atteint le podium. Nous pouvons en conclure que pour une candidate, '
        'un bon score est primordial pour maximiser ses chances au concours.')

    # Nombre de gagnantes par région (diagramme)
    for _ in range(5): st.write('')
    st.subheader('Nombre de gagnantes par région :')
    for _ in range(2): st.write('')
    col = st.columns([4, 7])

    # Sur la colonne de gauche on met l'explication
    col[0].write(
        'Sur une trentaine de régions chaque année, trois se distinguent en particulier : le Nord-Pas-de-Calais, '
        'la Guadeloupe et la Normandie. Plusieurs facteurs peuvent expliquer leur surreprésentation : ')
    col[0].write(
        '- Un engagement plus fort des habitants de la région: Line Renaud l\'a exprimé en ces termes lorsqu\'elle a '
        'été interrogée sur le sujet : "Les gens du Nord sont investis, ils votent à 100 % !"')

    st.write(
        '- Une meilleure préparation au concours : en 2019, le délégué régional des Miss dans le Nord-Pas-de-Calais '
        'avait annoncé mettre en place une préparation méticuleuse des candidates, comprenant des quiz de culture '
        'générale et un fort accent sur le développement de leur éloquence.')

    # Sur la colonne de droite, on met notre diagramme
    filtered_df = data_missFrance[data_missFrance['rang'] == 1]
    list_region = list(set(filtered_df[
                               'region'].tolist()))  # Récupération de la liste (sans doublons et triée) de
    # différentes régions présentes dans notre dataset
    labels = []
    sizes = []
    for region in list_region:
        labels.append(region)
        percent = filtered_df[filtered_df['region'] == region].shape[0] / 16
        sizes.append(percent)
    fig1, ax1 = plt.subplots()
    fig1.set_facecolor('#290425')
    # Il faut 11 couleurs
    colors = ['darkorchid', 'mediumpurple', 'purple', 'violet', 'magenta', 'orchid', 'mediumvioletred', 'hotpink',
              'palevioletred', '#f63366', 'm']
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'color': '#DCB253'}, colors=colors)
    col[1].pyplot(fig1)

    # Importance de ne pas tomber le soir de l'éléction & impact que cela peut avoir sur le classement final
    # Pour chaque classement, on récupère le nombre de candidates qui sont tombées le soir de l'élection
    for _ in range(5): st.write('')
    st.subheader('Importance de ne pas tomber le soir de l\'élection :')
    st.write('Le graphe ci-dessous représente le nombre de candidates qui sont tombées par classement.')
    data_has_fallen = {}
    sum = 0
    for i in range(1, 13):
        filtered_df = data_missFrance[data_missFrance['rang'] == i]
        ranked_candidates = filtered_df[filtered_df['est_tombee'] == 1]
        data_has_fallen[i] = len(ranked_candidates)
        sum += data_has_fallen[i]
    st.bar_chart(data=data_has_fallen, color='#f63366', use_container_width=True)
    st.write('Au total, seulement ' + str(
        sum) + ' candidates qui sont tombées le soir de l\'éléction ont atteint le podium.')
    st.write(
        'Un contre-exemple intéressant est celui d\'Indira Ampiot. La candidate guadeloupéenne a trébuché le soir de '
        'son éléction, mais cela ne l\'a pas empêché de remporter la couronne et de devenir Miss France 2023.')

    # Caractéristiques physiques
    for _ in range(5): st.write('')
    st.subheader('Les caractéristiques physiques: essentielles dans le classement ?')
    col = st.columns([8, 15])
    for _ in range(2): col[0].write('')
    col[0].write(
        'Est-ce qu\'une candidate a plus de chances de gagner si elle est brune ou blonde ? si elle a les yeux bleus '
        'ou verts ?')
    # Colonne de gauche : nombre de gagnantes par couleur de cheveux
    col[1].write('Répartition des couleurs de cheveux dans le classement')
    data_hair_color = {}
    hair_color = ['Noirs', 'Bruns', 'Chatains', 'Roux', 'Blonds']
    for color in hair_color:
        list_color = {}
        filtered_df = data_missFrance[data_missFrance['couleur_cheveux'] == color]
        for i in range(1, 13):
            filtered_df_rank = filtered_df[filtered_df['rang'] == i]
            list_color[i] = len(filtered_df_rank)
            data_hair_color[color] = list_color
    colors = ['#C8547C', '#f63366', '#6B006D', '#B70072', '#FC4B9C']
    col[1].bar_chart(data=data_hair_color, color=colors, use_container_width=True)

    # Colonne de droite : nombre de gagnantes par couleur de yeux
    col[1].write('Répartition des couleurs de yeux dans le classement')
    data_eyes_color = {}
    eyes_color = ['Noirs', 'Marrons', 'Gris', 'Bleus', 'Verts']
    for color in eyes_color:
        list_color = {}
        filtered_df = data_missFrance[data_missFrance['couleur_yeux'] == color]
        for i in range(1, 13):
            filtered_df_rank = filtered_df[filtered_df['rang'] == i]
            list_color[i] = len(filtered_df_rank)
            data_eyes_color[color] = list_color
    colors = ['#C8547C', '#f63366', '#6B006D', '#B70072', '#FC4B9C']
    col[1].bar_chart(data=data_eyes_color, color=colors, use_container_width=True)

    # Analyse des résultats
    col[0].write(
        'Nous pouvons facilement constater une grande mixité des couleurs de cheveux et. Nous pouvons analyser ce '
        'résultat de deux manières différentes : ')
    col[0].write(
        '- Le concours Miss France n\'est pas un simple concours de beauté, mais beaucoup d\'autres critères sont '
        'pris en compte par le jury et par le public : l\'élégance, la culture générale, et l\'éloquance entre autres.')
    col[0].write(
        '- Même si la beauté joue un rôle important dans la séléction d\'une candidate, il est dificile de quantifier '
        'cette beauté et de trouver des caractéristiques suffisantes pour représenter la beauté d\'une candidate.')
