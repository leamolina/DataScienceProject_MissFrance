import pickle

import pandas as pd
import streamlit as st

import Modele.data_split as ds
import Modele.multiModelTop12Predictor as my_model


# Fonctions diverses :
def give_real_rank(df, annee):
    filtered_df = df[df['annee'] == annee]
    rank = {}
    for i in range(1, 13):
        miss = filtered_df.loc[df['rang'] == i, 'name'].tolist()[0]
        rank[miss] = i
    return rank


# Option 1 : afficher la prédiction 2019
def define_tab1(tab1, myModel, ct):

    tab1.subheader('Première option : voir les prédictions de miss france 2019')

    # Récupération des données
    data_missFrance = pd.read_csv('./Databases/data_missFrance.csv', delimiter=';')
    data_missFrance_copy = data_missFrance.copy()
    data_missFrance = data_missFrance.drop(['audience', 'name', 'image'], axis=1)

    annee_test = 2019
    X_train, X_test, y_train, y_test = ds.data_split(data_missFrance, annee_test) #Récupération du test_set
    filtered_df = data_missFrance_copy[data_missFrance_copy['annee'] == annee_test]
    list_candidate = filtered_df['name'].tolist() # Récupération de la liste des candidates (pour l'affichage des prédictions)

    # Transformation des données
    X_test = ct.transform(X_test)

    # Affichage des prédictions
    prediction = myModel.predict(X_test, list_candidate)
    real_rank = give_real_rank(data_missFrance_copy, annee_test)
    i = 0
    while i < 12:
        columns = tab1.columns(4)
        for j in range(4):
            name = prediction[i + 1]
            path_image = 'Sources/Images_Candidates_2019/' + name + '.webp'
            columns[j].image(path_image)
            columns[j].write(str(i + 1) + ': ' + name)
            if name in real_rank:
                columns[j].write('Réel rang : ' + str(real_rank[name]))
            else:
                columns[j].write('Réel rang : non classée')
            i += 1
            tab1.write('')
    tab1.write('score de prédiction:')
    tab1.write(myModel.evaluate_prediction(X_test, list_candidate, real_rank))


# Option 2
def define_tab2(tab2, myModel, ct):

    tab2.subheader('Seconde option : voir les prédictions de miss France 2025 (données à entrer manuellement)')


    data_missFrance = pd.read_csv('./Databases/data_missFrance.csv', delimiter=';')
    list_region = sorted(list(set(data_missFrance[ 'region'].tolist())))  # Récupération de la liste (sans doublons et triée) de différentes régions présentes dans notre dataset
    nb_candidates = tab2.number_input('Choisir le nombre de candidates', 1, 30)
    tab2.write('Vous avez choisi ' + str(nb_candidates) + ' candidate(s)')

    for _ in range(3): tab2.write('')  # Saut de trois lignes

    # Listes vides pour stocker les réponses des formulaires
    list_regions = ['' for _ in range(nb_candidates)]
    list_names = ['' for _ in range(nb_candidates)]
    list_ages = [0 for _ in range(nb_candidates)]
    list_heights = [0 for _ in range(nb_candidates)]
    list_hairs_color = ['' for _ in range(nb_candidates)]
    list_hairs_length = ['' for _ in range(nb_candidates)]
    list_eyes_color = ['' for _ in range(nb_candidates)]
    list_skintone = ['' for _ in range(nb_candidates)]
    list_general_knowledge = [0 for _ in range(nb_candidates)]
    list_has_fallen = [0 for _ in range(nb_candidates)]
    list_years = [2025 for _ in range(nb_candidates)]

    # Affichage des différents formulaires (les uns après les autres)
    for i in range(nb_candidates):

        tab2.header('Informations concernant la candidate ' + str(i + 1))

        with tab2.form(key='form_' + str(i), border=True):

            # Informations générales
            columns_infos_generales = tab2.columns([2, 2, 1, 1])
            list_names[i] = columns_infos_generales[0].text_input('Nom de la candidate ' + str(i + 1))
            list_regions[i] = columns_infos_generales[1].selectbox('Région de la candidate ' + str(i + 1), list_region)
            list_ages[i] = columns_infos_generales[2].number_input('Âge de la candidate ' + str(i + 1), 18, 40)
            list_heights[i] = columns_infos_generales[3].slider('Taille de la candidate ' + str(i + 1) + ' (cm)', 130,
                                                                200)

            # Caractéristiques physiques
            columns_caracteristiques_physiques = tab2.columns(4)
            list_hairs_color[i] = columns_caracteristiques_physiques[0].selectbox(
                'Couleur des cheveux de la candidate ' + str(i + 1), ['Noirs', 'Bruns', 'Chatains', 'Roux', 'Blonds'])
            list_hairs_length[i] = columns_caracteristiques_physiques[1].selectbox(
                'Longueur des cheveux de la candidate ' + str(i + 1), ['Longs', 'Mi-longs', 'Courts'])
            list_eyes_color[i] = columns_caracteristiques_physiques[2].selectbox(
                'Couleur des yeux de la candidate ' + str(i + 1), ['Noirs', 'Marrons', 'Gris', 'Bleus', 'Verts'])
            list_skintone[i] = columns_caracteristiques_physiques[3].selectbox(
                'Couleur de la peau de la candidate ' + str(i + 1), ['Noire', 'Métisse', 'Blanche'])

            # Autres informations
            column_autre = tab2.columns(2)
            list_general_knowledge[i] = int(
                column_autre[0].radio('La candidate ' + str(i + 1) + ' a-t-elle eu le prix de culture générale ? ',
                                      ['Oui', 'Non']) == 'Oui')
            list_has_fallen[i] = int(
                column_autre[1].radio('La candidate ' + str(i + 1) + ' est-elle tombée le soir de l\'éléction ? ',
                                      ['Oui', 'Non']) == 'Oui')

            # Bouton de validation du formulaire
            if i == nb_candidates - 1:
                for _ in range(3): tab2.write('')
                submited = tab2.button('Voir les prédictions 👑')

        for _ in range(3): tab2.write('')  # Saut de 3 lignes

    # Vérification que les informations ont bien été remplies (la seule case n'ayant pas de valeur par défaut est la case 'nom')
    if submited:
        all_filled = True
        for i in range(nb_candidates):
            if list_names[i] == '':
                tab2.error('⚠️ Merci d\'indiquer le nom de la candidate ' + str(i + 1))
                all_filled = False
                break

        # Si les données sont complètes, nous pouvons commencer la prédiction
        if all_filled:

            # Création de la dataframe & transformation des données
            data_candidates = pd.DataFrame.from_dict(
                {'annee': list_years, 'region': list_regions, 'name': list_names, 'age': list_ages,
                 'taille': list_heights, 'couleur_cheveux': list_hairs_color, 'longueur_cheveux': list_hairs_length,
                 'couleur_yeux': list_eyes_color, 'couleur_peau': list_skintone,
                 'laureat_culture_generale': list_general_knowledge, 'est_tombee': list_has_fallen})
            data_candidates = ct.transform(data_candidates)

            # Prédiction et affichage du classement
            prediction = myModel.predict(data_candidates, list_names)
            liste_results = ['👑 Miss France 2025', '🥇 1ère Dauphine', '🥈 2ème Dauphine', '🥉 3ème Dauphine',
                             '4ème Dauphine', '5ème', '6ème', '7ème', '8ème', '9ème', '10ème', '11ème', '12ème']
            for i in range(len(prediction)):
                tab2.subheader(liste_results[i] + ' : ' + prediction[i + 1])


def page_prediction():

    # Logo en haut à droite
    chemin_logo = './Sources/Logo_MissFrance.png'
    col1, col2, col3 = st.columns((1, 4, 1))
    with col3:
        st.image(chemin_logo, use_column_width=True, width=10)

    st.title('Prédiction')
    st.write('Vous pouvez choisir parmi deux options.')
    st.write('Premièrement, vous pouvez voir les prédictions de notre année (l\'année 2019).')
    st.write(
        'Deuxièmement, vous pouvez entrer manuellement les données des candidates pour Miss France 2025 et obtenir les prédictions.')

    # Récupération du modèle
    list_of_models = []
    for i in range(12):
        path = './Modele/train/model_' + str(i) + '.pkl'
        model = pickle.load(open(path, 'rb'))
        list_of_models.append(model)
    myModel = my_model.MultiModelTop12Predictor(list_of_models)

    # Récupération du column transformer
    path_ct = './Modele/train/column_transformer.pkl'
    ct = pickle.load(open(path_ct, 'rb'))

    # Tabs
    tab1, tab2 = st.tabs(['Option 1', 'Option 2'])
    define_tab1(tab1, myModel, ct)
    define_tab2(tab2, myModel, ct)