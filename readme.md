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


#### Algorithme de Cooley–Tukey (FFT)

la **Transformée de Fourier Discrète (DFT)** peut être vue comme une simple évaluation de polynômes.  

Si l’on considère les données d’entrée comme les coefficients d’un polynôme :

$$
P(X) = \sum_{k=0}^{n-1} a_k X^k ,
$$

alors calculer la DFT revient à évaluer ce polynôme en $n$ points particuliers :  
les **racines $n$-ièmes de l’unité**

$$
\omega_n^j = e^{-\tfrac{2 \pi i}{n} j}, \quad j = 0,1,\dots,n-1.
$$

Pour un calcul efficace on utilise la méthode **diviser pour régner**.  

En effet, on s’aperçoit que pour obtenir les valeurs en $n$ points d’un polynôme, il suffit de séparer en **parties paire et impaire** :

$$
P(X) = \sum_{k=0}^{n-1} a_k X^k 
= P_\text{pair}(X^2) + X \cdot P_\text{impair}(X^2),
$$

où :  
- $P_\text{pair}(X) = a_0 + a_2 X + a_4 X^2 + \cdots$  
- $P_\text{impair}(X) = a_1 + a_3 X + a_5 X^2 + \cdots$

Ainsi, au lieu d’évaluer $P(X)$ en $n$ points, il suffit d’évaluer $P_\text{pair}$ et $P_\text{impair}$ en $n/2$ points chacun.  
Ces points sont reliés par des symétries : les racines de l’unité viennent par **paires conjuguées** ($\omega_n^j$ et $-\omega_n^j$).  

On obtient donc une récurrence :

$$
T(n) = 2\,T\!\left(\tfrac{n}{2}\right) + c n, \qquad T(1) = d.
$$

Écrivons la récurrence sur plusieurs niveaux :

$$
\begin{aligned}
T(n) &= 2\! \left( 2\,T\!\left(\tfrac{n}{2^2}\right)+c\tfrac{n}{2} \right) + c n
     = 2^2 T\!\left(\tfrac{n}{2^2}\right) + c n\left(1+\tfrac{1}{2}\right),\\
T(n) &= 2^2\! \left( 2\,T\!\left(\tfrac{n}{2^3}\right)+c\tfrac{n}{2^2} \right) + c n\left(1+\tfrac{1}{2}\right)
     = 2^3 T\!\left(\tfrac{n}{2^3}\right) + c n\left(1+\tfrac{1}{2}+\tfrac{1}{2^2}\right).
\end{aligned}
$$

Après $k$ niveaux on obtient :

$$
T(n) = 2^k T\!\left(\tfrac{n}{2^k}\right) + c n\sum_{i=0}^{k-1} \tfrac{1}{2^i}.
$$

Choisissons $k$ tel que $\tfrac{n}{2^k}=1$, donc $k=\log_2 n$.  
Alors $2^k = n$ et $T\!\left(\tfrac{n}{2^k}\right)=T(1)=d$. D'où :

$$
T(n) = n\cdot d + c n\sum_{i=0}^{\log_2 n -1} \tfrac{1}{2^i}.
$$

La somme géométrique vaut :

$$
\sum_{i=0}^{\log_2 n -1} \tfrac{1}{2^i} = \frac{1 - \tfrac{1}{2^{\log_2 n}}}{1-\tfrac{1}{2}}
= 2\left(1-\tfrac{1}{n}\right).
$$

Donc :

$$
T(n) = n d + c n \cdot 2\left(1-\tfrac{1}{n}\right)
     = n d + 2c n - 2c.
$$

Pour les grandes valeurs de $n$ (termes dominants) :

$$
T(n) = \Theta(n \log n).
$$

Ainsi la solution de :

$$
T(n)=2T(n/2)+\Theta(n)
$$

est bien :

$$
T(n)=\Theta(n\log n). \quad 🚀
$$

On obtient ainsi une complexité optimale en $O(n \log n)$ pour passer des coefficients aux valeurs et inversement.

---

#### Convolution

Pour effectuer les convolutions, on utilise l'**algorithme FFT 2D** qui est définie comme une FFT selon un axe puis une FFT selon l'autre axe, d’où une complexité en $O(n^2 \log n)$. 

En effet, soit $A$ et $B$ deux polynômes, et soit $C = A \times B$ leur produit. On remarque que les coefficients de $C$ vérifient :

$$
c_k = \sum_{i+j=k} a_i \cdot b_j
$$

On voit donc qu'il s'agit exactement de la **convolution** des coefficients $a_i$ et $b_j$.

Or, par les polynômes d'interpolation de Lagrange, on sait qu'un polynôme de degré $n$ peut être représenté par ses **valeurs en $n$ points distincts**.  

Ainsi, pour multiplier $A$ et $B$ :  
1. On évalue $A$ en $n$ points $(x_0, x_1, \dots, x_{n-1})$ → valeurs $A(x_i)$  
2. On évalue $B$ aux mêmes points → valeurs $B(x_i)$  
3. On multiplie **point par point** :  
   $$
   C(x_i) = A(x_i) \cdot B(x_i)
   $$  
4. On retrouve ensuite les coefficients de $C$ via l'interpolation de Lagrange.

💡 L'idée clé : la FFT permet de transformer la **convolution des coefficients** en une **multiplication point par point** très rapide.

---

Si l'on résout directement le système d'interpolation, la complexité est trop élevée ($O(n^2)$ par étape).  

Pour y remédier, on **choisit astucieusement les points** d'évaluation : les **racines de l'unité**, ce qui permet d'utiliser la **FFT**.

En effet la FFT n’est rien d’autre qu’une méthode efficace pour obtenir les $n$ valeurs  

$$
P(\omega_n^j), \quad j = 0,1,\dots,n-1,
$$

c’est-à-dire l’évaluation simultanée du polynôme $P$ en $n$ points bien choisis.

Pour faire une convolution, on a deux FFT 2D à effectuer et une multiplication point par point en $O(n^2)$.  
On obtient donc la convolution en $O(n^2 \log n)$ également.  

👉 Contre $O(n^4)$ pour la version naïve.

### Version parallélisée sur GPU

#### Problème des calculs inutiles

Pour qu’une cellule passe à un état non nul (vivant), elle doit avoir au moins une voisine active dans son voisinage. Or, en pratique, de nombreuses cellules sont recalculées inutilement.

Une solution possible consiste à ne calculer que les cellules susceptibles d’être modifiées à l’état suivant.

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
- [Generative Models for Periodicity Detection in Noisy Signals (Article)](https://arxiv.org/pdf/2201.07896)
- [FFT on GPU](https://www.kennethmoreland.com/fftgpu/fftgpu.pdf)