# Emotion Detection in Text using Natural Language Processing



<br>

# Sentiment.
1. Description

L'analyse de sentiments est un domaine de l'intelligence artificielle et du traitement automatique du langage naturel (TALN) qui vise à identifier et à catégoriser les opinions exprimées dans un texte, en particulier pour déterminer si l'attitude de l'auteur est positive, négative, ou neutre. 
Model utilisée :
On a utilisé  dans notre projet deux approche différente :
VADER (Valence Aware Dictionary and sentiment Reasoner) - Bag of words approach 
On a utilisé la dataset Reviews.csv Cette base de données contient généralement des avis de consommateurs sur des produits alimentaires vendus sur Amazon. 
Étape 1:VADER Sentiment Scoring
Nous utiliserons le SentimentIntensityAnalyzer de NLTK pour obtenir les scores neg/neu/pos du texte.Cela utilise une approche « bag of words » :
Les mots vides sont supprimés, chaque mot est noté et combiné pour obtenir un score total.
L'image illustre une analyse comparative de la distribution des sentiments classés en trois catégories : Positif, Neutre et Négatif.

Étape 2. Modèle pré-entraîné Roberta
Utilisez un modèle entraîné à partir d’une grande quantité de données.
Le modèle Transformers  prend en compte les mots mais aussi le contexte lié aux autres mots.

Ici on a Combiner et comparer les deux approches Roberta et VAder 

#  Emotion
1 Description
Cette section du projet est consacrée à l'exploration des capacités de détection des émotions en utilisant un modèle pré-entraîné de traitement du langage naturel (NLP). L'objectif est de discerner avec précision les émotions variées et souvent subtiles exprimées à travers le texte. Les émotions, telles que la joie, la tristesse, la colère, la surprise, la peur, et d'autres nuances émotionnelles, jouent un rôle crucial dans la compréhension du contexte et du ton d'un message.
2 Model utilisée :
Dans cette phase de notre projet, nous essayons d'utiliser EmoRoBERTa, un modèle de traitement du langage naturel de pointe, pour relever un défi unique : la détection et l'interprétation des émojis dans les textes. Les émojis, omniprésents dans la communication numérique moderne, offrent une fenêtre fascinante sur les émotions et les intentions des utilisateurs. Cependant, leur interprétation précise nécessite une compréhension nuancée du contexte et du ton, souvent difficile à saisir pour les algorithmes standard. EmoRoBERTa, avec sa capacité avancée à analyser les nuances émotionnelles, est idéalement positionné pour décoder ces symboles riches en significations. 

Application de la Régression Logistique pour la Détection des Émotions

et nous avons aussi essayé d'utiliser algorithme de Régression logistique de détection d'émotions à partir de textes, nous avons intégré l'approche de la régression logistique, un modèle classique de machine learning. Cette approche a été choisie pour sa simplicité, sa facilité d'implémentation et sa capacité à traiter efficacement des problèmes de classification multiclasse. tout en conservant les caractères spéciaux pour leur potentiel de signification émotionnelle. Nous avons ensuite préparé notre modèle en construisant une pipeline contenant un transformateur CountVectorizer pour convertir les textes en un format exploitable par le modèle, suivi de l'estimateur LogisticRegression. Cette pipeline a été entraînée sur un ensemble de données d'entraînement.
3 l'ensemble de données Emotion
cette dataset se trouve dans kaggle 
Contenu
Chaque enregistrement se compose de deux attributs :
Emotion : joy/sadness...
Text  : 
Après une phase initiale d'analyse exploratoire des données (EDA) où nous avons examiné la distribution des différentes émotions dans notre jeu de données, nous avons procédé à un nettoyage minutieux des textes, en éliminant les éléments non pertinents tels que les identifiants d'utilisateur et les mots vides, 
# Sarcasm
1Description
Cette partie de projet vise à explorer les capacités de détection de sarcasme à l'aide d'un modèle de traitement du langage naturel (NLP) pré-entraîné. Le sarcasme, caractérisé par une divergence entre le sens littéral et le sens implicite des propos, représente un défi majeur dans l'analyse de sentiments et la compréhension contextuelle.
2 Model pré-entraînée utilisée :
Dans la phase de construction du modèle, nous avons intégré une approche hybride combinant les embeddings de mots pré-entraînés GloVe avec un réseau neuronal récurrent de type LSTM (Long Short-Term Memory). 

Le modèle GloVe est couramment utilisé dans le traitement du langage naturel pour convertir des mots en représentations vectorielles capables de capturer la signification sémantique des mots. 
Ensuite, nous avons utilisé une couche LSTM, un type de RNN particulièrement efficace pour traiter les séquences de données en tenant compte de leur contexte temporel. Les LSTMs sont capables de capturer la dépendance à long terme et sont donc particulièrement adaptés à la compréhension du sarcasme, qui nécessite souvent une analyse contextuelle des phrases. 
Cette combinaison de GloVe et de LSTM vise à exploiter à la fois les représentations de mots riches en sémantique et la capacité des LSTMs à comprendre la structure séquentielle et le contexte du langage, éléments clés pour identifier avec précision le sarcasme dans le texte.



# Installation
1. Clone the repository to your local machine:
```
git clone 
```

2. Install the 'requirements.txt':
```
pip install -r requirements.txt
```

3. To run this project :
```
 pyhton -m streamlit run app.py
```

