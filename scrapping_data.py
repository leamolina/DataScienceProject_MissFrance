import math
import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd

list_years = []
list_regions = []
list_names = []
list_ages = []
list_heights = []
urls_images = []
list_ranks = []
list_hairs = []
list_eyes = []

#Permet de passer d'une chaine de caractères au nombre de centimètres de la candidate
def convert_height(str_height):
    meters, centimeters = str_height.split(',')
    height = int(meters) * 100 + int(centimeters.replace('m', ''))
    return height


#Permet de récupérerer les caractéristiques physiques de la candidate : couleur des cheveux et couleur des yeux
def get_info(url):
    hair = " "
    eyes = " "
    response2 = requests.get(url)
    if response2.ok:
        soup2 = BeautifulSoup(response2.text, 'lxml')
        tables = soup2.find_all('table')
        if(len(tables)>0):
            table = tables[0]
            rows = table.find_all('tr')
            if(len(rows)>0):
                found_hair = False
                found_eye = False
                i = 0
                while((not found_hair or not found_eye) and  i<len(rows)):
                    th = rows[i].find_all('th')
                    td = rows[i].find_all('td')
                    for j in range(len(th)):
                        if(th[j].text.strip()=="Cheveux"):
                            found_hair = True
                            hair = td[j].text.strip()
                        if(th[j].text.strip()=="Yeux"):
                            eyes = td[j].text.strip()
                            found_eye = True
                    i+=1
    return hair, eyes


#Permet de réupérer l'url de l'image de la candidate
def get_image(url):
    response2 = requests.get(url)
    if response2.ok:
        soup2 = BeautifulSoup(response2.text, 'lxml')
        tables = soup2.find_all('table')
        if(len(tables) >0):
            table = tables[0]
            rows = table.find_all('tr')
            if(len(rows)>0):
                tr = rows[1]
                cols = tr.find_all('td')
                if(len(cols)>0):
                    cols = cols[0]
                    a = cols.find_all('a')
                    if(len(a)>0):
                        url_image = a[0]['href']
                        return url_image



#Permet d'analyser le str du rang (sur wikipédia) et récupérer le classement (s'il y en a un)
def find_rank(rank):
    worlds = rank.split(" ")
    list_worlds = [world.split("\n") for world in worlds]
    worlds = [item for sublist in list_worlds for item in sublist]
    if(len(worlds) == 0 or len(worlds)==1): return np.NaN
    if(worlds[0] == "Prix"): return np.NaN
    if(worlds[0] == "Top"):
        nb= int(worlds[2].replace("(", "").replace(")", "").replace("e", ""))
        if("dauphine)" in worlds or "dauphine" in worlds): return nb+1
        else: return nb
    if(worlds[0] == "Miss") : return 1
    if(len(worlds)>0 and worlds[1] == "dauphine"): return int(worlds[0].replace("e", "").replace("r", ""))+1
    else : return np.NaN



#Scrapping années 2009 à 2023
for year in range(2009,2024):
    url = "https://miss.fandom.com/fr/wiki/Miss_France_" + str(year)
    response = requests.get(url)
    print('On commence à scrapper l\'année ' + str(year))
    if response.ok:
        soup = BeautifulSoup(response.text, 'lxml')
        tables = soup.find_all('table')

        # Ranking:
        rang_annee = {}
        rank = 1
        table_classement = tables[1] #Tableau des classements
        rows = table_classement.find_all('tr')
        #La première ligne c'est les titres
        for row in rows[1:]:
            cols = row.find_all('td')
            if len(cols) > 0:
                c = cols[1].find_all('a')
                if len(c) > 0:
                    i = 1
                    while (i < len(c)):
                        winner = c[i].text
                        if (winner != ""):
                            winner = winner.replace("Miss", "").strip()
                            rang_annee[winner] = rank
                            rank += 1
                            i += 1

                        i += 1

        #Tout sauf le ranking
        table_candidates = tables[2] #Tableau des candidates
        rows = table_candidates.find_all('tr')
        for row in rows[1:]:
            cols = row.find_all('td')
            if len(cols) > 0:
                region = cols[0].text.replace("Miss", "").strip()
                name = cols[1].text.strip()

                #Détection de l'url concernant la miss
                a = cols[1].find('a')
                if(a is not None):
                    #urls_images.append('https://miss.fandom.com' + a['href'])
                    #On récupère l'image à partir de l'url:
                    urls_images.append(get_image('https://miss.fandom.com' + a['href']))
                    hair, eye = get_info('https://miss.fandom.com' + a['href'])
                    list_hairs.append(hair)
                    list_eyes.append(eye)


                else:
                    urls_images.append(np.NaN)
                    list_hairs.append(" ")
                    list_eyes.append(" ")

                age = int(cols[2].text.split(" ")[0])
                taille = convert_height(cols[3].text.strip())
                list_years.append(year)
                list_regions.append(region)
                list_names.append(name)
                list_ages.append(age)
                list_heights.append(taille)
                list_ranks.append(rang_annee.get(region,np.NaN))


#Scrapping année 2024

year = 2024
rang_annee = {}
url = "https://fr.wikipedia.org/wiki/Miss_France_2024"
page = requests.get(url)


if page.ok:
    soup = BeautifulSoup(page.text, 'lxml')
    tables = soup.find_all('table')
    if(len(tables)>=6):
        table_candidates = tables[6] #5 eme table = table des classements sur wikipédia
        rows = table_candidates.find_all('tr')
        for row in rows[1:]:
            cols = row.find_all('td')
            if len(cols) > 0:
                region = cols[0].text.replace("Miss", "").strip()
                name = cols[1].text.strip()
                rank = cols[6].text.strip()
                age = int(cols[2].text.split(" ")[0])
                height = convert_height(cols[3].text.strip())
                list_years.append(year)
                list_regions.append(region)
                list_names.append(name)
                list_ages.append(age)
                list_heights.append(height)
                urls_images.append(np.NaN)
                list_hairs.append(" ")
                list_eyes.append(" ")
                list_ranks.append(find_rank(rank))

ensemble_sans_doublons = set(list_regions)
# Nombre d'éléments dans l'ensemble
nombre_elements_sans_doublons = len(ensemble_sans_doublons)
print(nombre_elements_sans_doublons)
"""
data_missFrance = pd.DataFrame.from_dict({'annee': list_years, 'region': list_regions, 'name': list_names, 'age': list_ages, 'taille': list_heights, 'cheveux':list_hairs, 'yeux': list_eyes, 'rang':list_ranks, 'image': urls_images})
chemin_fichierLea = "/Users/LEAMOLINA1/Desktop/M1/S2/Projet ML : DataScience/Projet/data_missFrance_incomplete.csv"
chemin_fichierAna= "/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/data_missFrance_incomplete.csv"
data_missFrance.to_csv(chemin_fichierLea, index=False)
#data_missFrance.to_csv(chemin_fichierAna, index=False)"""
