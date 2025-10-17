#  MCOT — Mise en Cohérence des Objectifs du TIPE

---

## 1. Positionnements thématiques & mots-clés
- **Positionnements thématiques** (ordre décroissant — 3 max, le premier relevant de ta filière) :
  1. INFORMATIQUE
  2. …
  3. …
- **Mots-clés (français / anglais)** — 5 mots-clés, classés par importance :
  - français : Lenia - 
  - anglais : Lenia - artificial life - cellular automata - 

---

## 2. Bibliographie commentée (≈ 650 mots max)

1. Depuis son invention en 1970 par John Horton Conway, le Game of Life (GoL) est devenu l’un des automates cellulaires les plus emblématiques des mathématiques récréatives et de la simulation complexe. Ce système, régi par des règles simples de naissance, de survie et de mort appliquées sur une grille bidimensionnelle, a révélé une étonnante richesse de comportements émergents. Parmi les structures les plus célèbres figure le glider, une configuration stable capable de se déplacer au fil des générations selon un motif régulier. **[1]**

2. Au fil du temps, de nombreuses généralisations du GoL ont vu le jour, cherchant à étendre ou modifier ses propriétés fondamentales : nombre d’états, topologie du voisinage, ou encore type de règles. Parmi ces extensions, le modèle Larger than Life (LtL), introduit par Evans, remplace le voisinage de Moore classique (8 cases adjacentes) par un voisinage défini par un rayon r, englobant ainsi un nombre potentiellement bien plus élevé de cellules. Cette approche ouvre la voie à des dynamiques plus variées et à la création de structures se déplaçant selon des directions non triviales. **[2]**

3. La généralisation naturelle de LtL consiste à faire tendre le rayon r vers l’infini, conduisant à un modèle continu, où chaque cellule devient un point infinitésimal dans l’espace. Ce passage au continuum a été formalisé par Pivato sous le nom de RealLife, première formulation mathématique d’un automate cellulaire en espace réel. **[3]**

4. Dans la continuité de ces travaux, Lenia, introduit par Chan en 2018, propose une version entièrement continue du Game of Life : l’espace, le temps et les états cellulaires y sont représentés par des valeurs réelles.**[4]** Cette approche donne naissance à une riche diversité de structures auto-organisées, parfois qualifiées de formes de vie artificielles continues. Cependant, cette continuité se traduit par une augmentation considérable de la complexité de calcul, chaque itération nécessitant l’évaluation du voisinage à grande échelle pour chaque point du champ. **[5]**

5. Afin de rendre ces simulations réalisables, les implémentations modernes de Lenia recourent à l’algorithme de la Transformée de Fourier Rapide (FFT) pour accélérer la convolution spatiale entre la matrice d’état et le noyau de voisinage. Cette optimisation, fondée sur la propriété de convolution dans le domaine fréquentiel, permet de réduire la complexité du calcul, rendant ainsi possible l’exploration numérique de dynamiques complexes sur de larges grilles. **[6]**

6. Malgré ces améliorations, la simulation de Lenia demeure coûteuse sur le plan computationnel, notamment lors de l’exploration de longues dynamiques où le système converge vers des comportements quasi-stationnaires ou périodiques.
Chan a montré que de nombreuses formes de vie artificielle dans Lenia présentent ce genre de comportements. **[5]**
Dans ce contexte, la poursuite du calcul au-delà d’un cycle complet conduit à des redondances temporelles sans nouvel apport informatif.

7. Une piste d’optimisation consiste donc à détecter automatiquement la réapparition d’états similaires au cours de la simulation, afin d’interrompre ou de compresser le calcul lorsque le système entre dans une phase cyclique.

---

(parler de hash life)
## 3. Problématique retenue (≈ 50 mots max)

- Dans le but d’identifier des structures complexes et d’optimiser les performances de calcul, je vais chercher à exploiter la détection de cycles dans l’évolution de Lenia afin d’éviter les calculs redondants entre générations.

---

## 4. Objectifs du TIPE (≈ 100 mots max) (à rediger...)
1. Implemetattion Naïve du jeu la vie pour reference.
2. Implementation de la version avec detection de cycle et comparaisons avec la methode naïve.

3. Implemtation naïve de Lenia en C afin d'avoir une refference.
4. Implementation de de la methode de Chan : CUDA + FFT
5. Implementation de detection de cycle et comparaisons 

5. Implementation de detection de cycle et comparaisons + CUDA + FFT et comparaison 

6. Interpretation des resultat. (CUDA c'est rapide...)


- Énoncé précis de ce que tu vises à accomplir dans ton travail, en réponse à la problématique.

---

## 5. Liste des références bibliographiques (2 à 10 références)
  1. Martin Gardner, **Mathematical Games - The fantastic combinations of John Conway's new solitaire game "life"**, Scientific American, 223(4), October 1970 p. 120–123, https://web.stanford.edu/class/sts145/Library/life.pdf
  2. Kellie M. Evans, **Larger than Life: Digital Creatures in a Family of Two-Dimensional Cellular Automata. Discrete Models: Combinatorics, Computation, and Geometry**, DM-CCG 2001, 2001, Paris, France. pp.177-192, https://inria.hal.science/hal-01182968v1/document
  3. Pivato, M., **RealLife: The continuum limit of Larger than Life cellular automata.**, Theoretical Computer Science, 2007, 372(1):46–68. https://arxiv.org/pdf/math/0503504
  4. Chan, B. W.-C. , **Lenia: Biology of artificial life. Complex Systems**, arXiv preprint, 2019, https://arxiv.org/pdf/1812.05433
  5. Chan, B. W.-C. ,  **Lenia and Expanded Universe**, arXiv preprint, 2020, https://arxiv.org/pdf/2005.03742
  6. Open-Source, **Lenia - Mathematical Life Forms**, Repo Github, https://github.com/Chakazul/Lenia