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

- Cloner le projet à partir de la commande suivante : <br/>
git clone https://github.com/VictorLEDEZ/mwm_pfr.git
- Récupérer le sous-module darkflow-mast pour la détection d'objets : 
	- git submodule init
	- git submodule update
- Créer un environnement avec la version 3.7 de python à partir de la commande suivante : <br/>
conda create --name myenv --file requirements.txt python=3.7.13
- Activer myenv avec la commande ci-dessous : <br/>
conda activate myenv
- Installer les libraires madmom et sf_segmenter avec les commandes suivantes : 
	- pip install madmom
	- pip install sf-segmenter
- Utiliser la commande suivante depuis le dossier darkflow-mast : <br/>
pip install .
- Créer un dossier bin au dossier darkflow-mast
- Ajouter les poids du réseau de neurones (ex : yolo.weights) dans le dossier bin. 
Les poids peuvent être téléchargés à partir de ce [lien](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU?resourcekey=0-s5xkbwFqbiBW2QEqj98vkQ).