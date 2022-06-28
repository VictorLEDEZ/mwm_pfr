# Projet Fil Rouge MWM

## Project structure

```
project_root/
├─ tests/
├─ edivio/                      (core package)
│  ├─ feature_extraction/       (audio and video feature extraction api)
│  │  ├─ audio/                 
│  │  ├─ video/
│  ├─ editing/                  (video editing logic) 
│  ├─ data/                     (package data)

```

## Installation

- Cloner le projet à partir de la commande suivante : 
	git clone https://github.com/VictorLEDEZ/mwm_pfr.git
- Récupérer le sous-module darkflow-mast pour la détection d'objets : 
	- git submodule init
	- git submodule update
- Créer un environnement avec la version 3.7 de python
- Installer les librairies du fichier requirement.txt dans le nouvel environnement
- Utiliser la commande suivante depuis le dossier darkflow-mast : 
	pip install .
- Créer un dossier bin au dossier darkflow-mast
- Ajouter les poids du réseau de neurones (ex : yolo.weights) dans le dossier bin 
