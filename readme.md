# TIPE : Optimisation de Lenia par détection de structures récurrentes

## Objectif

L’objectif est d’utiliser la détection de cycles pour optimiser Lenia.

En exécutant le Jeu de la Vie à la main, on remarque qu’il est souvent possible de réduire les calculs en identifiant des structures stables, comme les oscillateurs. Il est donc raisonnable de se demander si une optimisation similaire peut être appliquée à Lenia.

## Plan

1. Implémentation du Jeu de la Vie  
    - Version naïve  
    - Version avec détection de cycle  
2. Optimisation de Lenia  
    - Version naïve  
    - Version avec FFT  
    - Version parallélisée sur GPU  
    - Version avec détection de cycle  
3. Complétude de Lenia (bonus)  
    - Écrire un algorithme génétique pour trouver un glider gun  
    - Trouver un glider gun  

## Définition

On appelle simulation ce qui est défini par :

- Un champ scalaire, $A : L \mapsto [0, 1]$, où $L$ est un espace euclidien muni de la distance $d$ (cette implémentation utilise un espace euclidien en deux dimensions, mais Lenia peut être implémentée dans n’importe quelle dimension)
- Une fonction de croissance, $G : [0, 1] \mapsto [-1, 1]$
- Un noyau de convolution, $K : \mathbb{R} \mapsto [0, 1]$, tel que  
$$
\int_{0}^{2\pi}\int_{0}^{+\infty} K(r) \,dr \, d\theta = 1
$$

où $K$ associe à une distance $r$ un poids de voisinage — plus $r$ est petit, plus le nœud considéré est un voisin.

- Un pas de temps $\Delta t$

La formule pour calculer l’état de $A(\vec{x})$ est :
$$
A^{t+\Delta t}(\vec{x}) = \text{clip}(A^t(\vec{x}) + \Delta t \cdot G^t(\vec{x}), 0, 1)
$$

où $G^t$ est le résultat de l’application de la fonction de croissance $G$ à la convolution de $K$ et $A^t$.

---

## Jeu de la Vie

Dans le cas du Jeu de la Vie de Conway :

- $
K(r) =
\begin{cases} 
\frac{1}{9}, & \text{si } r = 1 \text{ ou } \sqrt{2} \\ 
0, & \text{sinon}
\end{cases}
$

- $
G(s) =
\begin{cases} 
1, & \text{si } s = 2 \\ 
0, & \text{si } s = 3 \\ 
-1, & \text{sinon}
\end{cases}
$

- $\Delta t = 1$

- On prendra un champ scalaire $A^0$ initialement aléatoire.

### Version naïve

On atteint difficilement les 30 générations par seconde.

### Version avec détection de cycle

Après recherche, il apparaît que parmi toutes les structures oscillantes qui émergent naturellement avec une probabilité supérieure à une sur un milliard, la majorité oscille en seulement deux générations.

On choisit alors de les détecter pour ne pas les recalculer.

Cette optimisation permet un gain significatif en termes de performances, augmentant la vitesse de génération de 30 à 300 générations par seconde.

---

## Lenia

Lenia est une version continue du Jeu de la Vie.

Pour passer à un modèle continu, on utilise des fonctions gaussiennes :

- $
\text{bell}(x) = \exp\left(-\frac{(x - m)^2}{2s^2}\right)
$

- $
G(v) = 2 \cdot \text{bell}(v, 0.15, 0.015) - 1
$

- $
K_R(r) = \text{bell}\left(\frac{r}{R}, 0.5, 0.15\right) \cdot \frac{1}{S_R}
$

avec
$
S_R = \int_0^{2\pi} \int_0^{+\infty} \text{bell}\left(\frac{r}{R}, 0.5, 0.15\right) \, dr \, d\theta
$

- $\Delta t \rightarrow 0$

### Version naïve

Absurdemment lente, même pour des simulations très petites.

### Version avec FFT

Le gain est considérable : on peut alors augmenter la résolution de la simulation par un facteur 16 tout en conservant la même fréquence de rafraîchissement.

#### Convolution

Pour effectuer les convolutions, on utilise l’algorithme FFT 2D qui permet de les calculer en $O(n^2 \log n)$ au lieu de $O(n^4)$ pour une version naïve.  
(explication détaillée à ajouter)

### Version parallélisée sur GPU

#### Problème des calculs inutiles

Pour qu’une cellule passe à un état non nul (vivant), elle doit avoir au moins une voisine active dans son voisinage. Or, en pratique, de nombreuses cellules sont recalculées inutilement.

Une solution possible consiste à ne calculer que les cellules susceptibles d’être modifiées à l’état suivant.

#### Région d’intérêt ← À FAIRE !!!

Sur une grande zone vide, la simulation devrait être rapide. On va donc définir des zones d’intérêt.

Ces zones seront de taille $s$, une puissance de deux (pour optimiser le calcul), et suffisamment grandes pour permettre la convolution sans effets de bord. Pour cela, on ajoute la taille du rayon du noyau sur chaque côté avant d’effectuer la convolution, puis on récupère uniquement les valeurs intéressantes.

#### Utilisation de CUDA pour paralléliser

On atteint ainsi l’état de l’art actuel, mais les performances pour une simulation en $K$ restent très faibles.

### Version avec détection de cycle

(À compléter)

---

## Documentation

- [Vidéo Science Étonnante](https://www.youtube.com/watch?v=PlzV4aJ7iMI)  
- [Tutoriel Lenia initial](https://colab.research.google.com/github/OpenLenia/Lenia-Tutorial/blob/main/Tutorial_From_Conway_to_Lenia.ipynb#scrollTo=ycvjBlAOt6tK)  
- [Lenia and Expanded Universe (Article)](https://arxiv.org/pdf/2005.03742)  
- [Lenia — Biology of Artificial Life (Article)](https://arxiv.org/pdf/1812.05433)
- [Generative Models for Periodicity Detection in
Noisy Signals (Article)](https://arxiv.org/pdf/2201.07896)