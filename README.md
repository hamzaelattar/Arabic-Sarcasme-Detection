# Détection du sarcasme en arabe

Projet de classification de tweets arabes avec le jeu de données **ArSarcasm-v2**. Le notebook [`test.ipynb`](test.ipynb) contient l'audit des données, une baseline TF-IDF, les approches zero-shot, AraVec, les modèles de langue gelés et fine-tunés, puis l'évaluation de généralisation.

## Avant de commencer

- Installer **VS Code** avec les extensions [Python et Jupyter](https://code.visualstudio.com/docs/datascience/jupyter-notebooks). Pour calculer sur Google Colab depuis VS Code, installer aussi l'[extension officielle Google Colab](https://github.com/googlecolab/colab-vscode).
- Disposer d'une connexion Internet pour télécharger les données et les modèles. Les étapes de fine-tuning demandent beaucoup de temps et un GPU est recommandé. La baseline TF-IDF peut tourner sans GPU.
- Ouvrir **ce dossier du projet** dans VS Code, puis `test.ipynb`. Le notebook installe les bibliothèques Python manquantes dans sa cellule « Dépendances et configuration » ; redémarrer le noyau si `pip` le demande.

## Exécuter avec Colab depuis VS Code

1. Dans `test.ipynb`, cliquer sur **Select Kernel** (en haut à droite), puis **Colab > Auto Connect** et se connecter au compte Google. Pour choisir un GPU, sélectionner un serveur Colab avec GPU dans le menu du noyau. [Guide de l'extension Colab](https://github.com/googlecolab/colab-vscode).
2. Exécuter la première cellule de vérification. `Colab :` doit afficher une valeur et `Dossier du noyau :` doit indiquer un chemin Linux tel que `/content`. Si `Colab : None` apparaît, le noyau est local : revenir au choix du noyau. Pour vérifier le GPU, exécuter `import torch; print(torch.cuda.is_available())` après l'installation des dépendances.
3. **Avant la cellule de configuration qui définit `ROOT`**, monter Google Drive et choisir un dossier persistant. Dans VS Code, la commande **Colab: Mount Google Drive to Server...** peut préparer le montage ; on peut aussi exécuter cette cellule :

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   import os
   os.environ['TP2_OUTPUT_DIR'] = '/content/drive/MyDrive/TP2_Sarcasme'
   ```

   `TP2_OUTPUT_DIR` et `TP2_Sarcasme` sont les identifiants de chemin déjà utilisés par le notebook et les anciens résultats. Le nom public du projet est **Détection du sarcasme en arabe**. La cellule de configuration doit alors afficher `Sorties : /content/drive/MyDrive/TP2_Sarcasme`.
4. Exécuter les autres cellules dans l'ordre. **Si `ROOT` pointe déjà vers Drive, ignorer la cellule « Sauvegarde des résultats Colab » après la partie E** : les fichiers sont déjà écrits dans Drive. Poursuivre directement avec F puis G.

Une autre possibilité est de laisser `TP2_OUTPUT_DIR` non défini : le notebook écrit alors dans `/content`. Dans ce cas, exécuter la cellule « Sauvegarde des résultats Colab » **après E et avant F** pour copier les sorties vers `Mon Drive/TP2_Sarcasme`. `/content` est temporaire ; faire la copie avant de fermer la session Colab.

## Exécuter localement dans VS Code

1. Ouvrir le dossier du projet et choisir **Select Kernel > Python Environments**, puis un environnement Python local. Installer `ipykernel` dans cet environnement si VS Code le demande.
2. Exécuter la cellule de vérification : `Colab : None` est attendu. Le dossier du noyau doit être `H:\detection_saracasme_arabe` sur cette machine. Si ce n'est pas le cas, exécuter cette cellule **avant** celle qui définit `ROOT` :

   ```python
   import os
   os.environ['TP2_OUTPUT_DIR'] = r'H:\detection_saracasme_arabe'
   ```

3. Exécuter les cellules dans l'ordre, mais **ignorer la cellule « Sauvegarde des résultats Colab »** : elle importe `google.colab` et concerne uniquement le noyau Colab. Les sorties seront dans les sous-dossiers du projet sur `H:`. Sur un autre ordinateur, remplacer `H:\detection_saracasme_arabe` par le chemin réel du projet ; si le noyau démarre déjà dans ce dossier, aucun réglage de chemin n'est nécessaire.

La cellule de configuration contient `RUN_ZERO_SHOT`, `RUN_STATIC`, `RUN_FROZEN`, `RUN_FINETUNE` et `RUN_GENERALIZATION`. Les mettre à `False` pour un premier passage rapide de la baseline, puis les réactiver pour les expériences complètes. Le notebook n'a pas de reprise automatique d'une expérience interrompue ; les fichiers déjà produits restent toutefois sur disque si le dossier de sortie est persistant.

## Où trouver les fichiers

| Dossier | Contenu |
| --- | --- |
| `data/` | CSV officiels train/test ; téléchargés si absents. |
| `results/` | Métriques dans `results.csv`, coûts des modèles et 30 erreurs à annoter. La colonne `categorie_manuelle` doit être vérifiée par une personne. |
| `predictions/` | Prédictions des modèles évalués. |
| `figures/` | Graphiques et matrice de confusion. |
| `models/` | Checkpoints locaux volumineux, exclus de Git. |

**Le noyau détermine l'emplacement des fichiers.** Avec Colab, `H:` n'est pas accessible directement : lire les résultats dans `Mon Drive/TP2_Sarcasme` ou les télécharger depuis Drive pour les placer sur Windows. Avec un noyau local, lire les sous-dossiers de `H:\detection_saracasme_arabe`. Le fichier Word du compte-rendu reste local et est ignoré par Git. Le code et les résultats suivis par Git sont disponibles sur [GitHub](https://github.com/hamzaelattar/Arabic-Sarcasme-Detection).

## Résultat principal

La sélection sur la validation retient E2 AraBERT-twitter multi-tâche (`λ = 0,3`, graine 123, seuil 0,76), avec un F1 sarcastique de 0,7059 sur validation et 0,6001 sur test. Les étiquettes du test officiel ne servent à l'évaluation qu'à partir de la partie F.
