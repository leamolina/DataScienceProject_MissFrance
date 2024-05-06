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
    for _ in range(3) : st.write('')

    # Texte de présentation de nos données
    st.write('Notre modèle s\'appuie sur différentes sources de données: ')
    st.write('- La région des candidates (source: miss.fandom.com)')
    st.write('- Les caractéristiques physiques des candidates : âge, taille, couleur des yeux et des cheveux... (source: miss.fandom.com & remplissage manuel)')
    st.write('- Si elle a gagné le prix de culture générale. Les candidates passent toutes un examen écrit pour évaluer leur niveau de culture générale. Celle qui obtient la meilleure note gange le prix de culture générale, un atout majeur dans la compétition. (source: Wikipédia)')
    st.write('- Si elle est tombée lors du prime : Bien que cela puisse prêter à rire, les passionnés du concours savent qu\'une candidate qui tombe le soir de l\'élection a très peu de chances d’atteindre le podium. A ce jour, la seule qui y est parvenue est Indira Ampiot, Miss France 2024. (source: TikTok, Youtube, Instagram)')

    # Récupération et affichage de la base de données
    for _ in range(3) : st.write('')
    chemin_database = './Databases/data_missFrance.csv'
    data_missFrance = pd.read_csv(chemin_database, delimiter=';')
    st.subheader('Notre base de données:')
    with st.expander('Cliquer ici pour voir notre base de données complète avant le pré-traitement des données'):
        st.dataframe(data_missFrance)

    # Importance du prix de culture générale
    # On veut savoir combien, par rang, combien de candidates ont gagné le prix de culture générale
    for _ in range(3) : st.write('')
    st.subheader('Importance de la culture générale dans le classement :')
    st.write('Comme expliqué précédement, la culture générale est .... ')
    data_percent = {}
    sum = 0
    for i in range(1, 13):
        filtered_df = data_missFrance[data_missFrance['rang'] == i]
        ranked_candidates = filtered_df[filtered_df['laureat_culture_generale'] == 1]
        data_percent[i] = (len(ranked_candidates)/len(filtered_df)) * 100
        sum+=data_percent[i]
    st.bar_chart(data=data_percent, color= '#f63366', use_container_width = True)
    st.write('Au total, ' + str(sum) + '% des lauréates au prix de culture générale ont atteint le top 12.')
    st.write('ANALYSE DES RESULTATS')

    # Nombre de gagnantes par région (diagramme)
    for _ in range(3): st.write('')
    st.subheader('Nombre de gagnantes par région :')
    for _ in range(2): st.write('')
    col = st.columns(2)

    #Sur la colonne de gauche on met notre diagramme
    filtered_df = data_missFrance[data_missFrance['rang'] == 1]
    list_region = list(set(filtered_df[ 'region'].tolist())) # Récupération de la liste (sans doublons et triée) de différentes régions présentes dans notre dataset
    labels = []
    sizes = []
    for region in list_region:
        labels.append(region)
        percent = filtered_df[filtered_df['region'] == region].shape[0]/16
        sizes.append(percent)
    fig1, ax1 = plt.subplots()
    fig1.set_facecolor('#290425')
    #Il faut 11 couleurs
    colors = ['#DCB253', '#B93A1E', '#74A414', '#A219C0', '#5C4793', '#478D93', '#936747', '#934768']
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'color': "#f63366"}, colors=colors)
    col[0].pyplot(fig1)

    #Sur la colonne de droite on analyse les résultats
    for _ in range(2): col[1].write('')
    col[1].write("Blabla explication du diagramme")


    # Importance de ne pas tomber le soir de l'éléction & impact que cela peut avoir sur le classement final

    # Nombre de gagnantes par couleur de cheveux ?

    # Nombre de gagnantes par couleur de yeux ?

    # Regrouper les données par année et par couleur de cheveux, puis compter le nombre d'occurrences
    for _ in range(3) : st.write('')
    st.subheader('Répartition de la couleur de cheveux des Miss par année')
    data_subset = data_missFrance[['annee', 'couleur_cheveux']]
    cheveux_par_annee = data_subset.groupby(['annee', 'couleur_cheveux']).size().reset_index(name='nombre')

    # Afficher l'histogramme
    st.bar_chart(cheveux_par_annee, x='annee', y='nombre', color='couleur_cheveux', use_container_width=True)
    chart_data = pd.DataFrame(cheveux_par_annee, data_subset)
    #st.bar_chart(chart_data)