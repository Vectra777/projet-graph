from __future__ import annotations

INF = float("inf")


class Graph:
    """
    Représentation d'un graphe orienté valué par sa matrice d'adjacence.

    Deux formats de fichier sont supportés (détection automatique) :

    Format A : Matrice d'adjacence :
        Ligne 1 : nombre de sommets n
        Lignes 2..n+1 : n valeurs séparées par des espaces
                        (entiers, flottants, ou le mot-clé INF)
        Exemple :
            4
            0   3   INF 7
            8   0   2   INF
            5   INF 0   1
            2   INF INF 0

    Format B : Liste d'arcs :
        Ligne 1 : nombre de sommets n
        Ligne 2 : nombre d'arcs m
        Lignes 3..m+2 : <sommet_source> <sommet_destination> <valeur>
        Exemple :
            4
            5
            3 1 25
            1 0 12
            2 0 -5
            0 1 0
            2 1 7
        Les arcs non listés sont considérés comme inexistants (valeur INF).
        La diagonale (i -> i) vaut 0 par défaut.
    """

    def __init__(
        self,
        size: int,
        matrix: list[list[float]],
        labels: list[str] | None = None,
    ) -> None:
        self.size = size
        self.matrix = matrix
        # Par défaut les sommets sont nommés "0", "1", ..., "n-1"
        self.labels = labels if labels is not None else [str(i) for i in range(size)]

    # ------------------------------------------------------------------
    # Chargement depuis un fichier : point d'entrée public
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, filepath: str) -> "Graph":
        """
        Charge un graphe depuis un fichier texte.
        Le format (matrice ou liste d'arcs) est détecté automatiquement
        à partir du contenu du fichier.
        Lève ValueError si le fichier est mal formé.
        """
        with open(filepath, "r", encoding="utf-8") as fh:
            # On lit toutes les lignes non vides une seule et unique fois
            lines = [line.strip() for line in fh if line.strip()]

        if not lines:
            raise ValueError("Le fichier est vide.")

        # Lecture du nombre de sommets (commun aux deux formats)
        try:
            size = int(lines[0])
        except ValueError:
            raise ValueError(
                f"La première ligne doit être un entier (nombre de sommets), "
                f"obtenu : '{lines[0]}'"
            )

        if size <= 0:
            raise ValueError(
                f"Le nombre de sommets doit être strictement positif, obtenu : {size}"
            )

        # --- Détection automatique du format ----------------------------
        # Heuristique : si la ligne 2 est un entier seul ET que le nombre
        # de lignes restantes correspond à nb_arcs lignes de 3 tokens,
        # c'est le format B (liste d'arcs). Sinon c'est le format A (matrice).
        is_arc_list = cls._detect_arc_list_format(lines, size)

        if is_arc_list:
            return cls._load_arc_list(lines, size)
        else:
            return cls._load_matrix(lines, size)

    # ------------------------------------------------------------------
    # Détection du format
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_arc_list_format(lines: list[str], size: int) -> bool:
        """
        Retourne True si le fichier semble être au format liste d'arcs.

        Critères :
          - Il existe une ligne 2 qui est un entier seul (nb_arcs).
          - Le nombre total de lignes est cohérent avec size + 1 + nb_arcs.
          - La ligne 2 ne ressemble pas à une ligne de matrice
            (une ligne de matrice a `size` tokens, pas 1).
        """
        if len(lines) < 2:
            return False

        tokens_line2 = lines[1].split()

        # Une ligne de matrice aurait `size` tokens ; une ligne de comptage en a 1
        if len(tokens_line2) != 1:
            return False

        try:
            nb_arcs = int(tokens_line2[0])
        except ValueError:
            return False

        # Vérification de cohérence : on doit avoir exactement nb_arcs lignes d'arcs
        expected_total = 1 + 1 + nb_arcs  # ligne size + ligne nb_arcs + arcs
        return len(lines) == expected_total

    # ------------------------------------------------------------------
    # Chargement : Format A : matrice
    # ------------------------------------------------------------------

    @classmethod
    def _load_matrix(cls, lines: list[str], size: int) -> "Graph":
        """
        Construit un Graph à partir du format matrice.
        `lines` est déjà filtré (lignes non vides, stripped).
        """
        if len(lines) < size + 1:
            raise ValueError(
                f"Le fichier déclare {size} sommet(s) mais ne contient que "
                f"{len(lines) - 1} ligne(s) de matrice (attendu : {size})."
            )

        matrix: list[list[float]] = []

        for row_idx in range(size):
            raw = lines[1 + row_idx].split()

            if len(raw) != size:
                raise ValueError(
                    f"La ligne {row_idx + 1} de la matrice contient {len(raw)} valeur(s) "
                    f"au lieu de {size}."
                )

            row: list[float] = []
            for token in raw:
                row.append(_parse_value(token, row_idx, size))
            matrix.append(row)

        return cls(size, matrix)

    # ------------------------------------------------------------------
    # Chargement : Format B : liste d'arcs
    # ------------------------------------------------------------------

    @classmethod
    def _load_arc_list(cls, lines: list[str], size: int) -> "Graph":
        """
        Construit un Graph à partir du format liste d'arcs.
        `lines` est déjà filtré (lignes non vides, stripped).
        """
        # Lecture du nombre d'arcs
        try:
            nb_arcs = int(lines[1])
        except ValueError:
            raise ValueError(
                f"La deuxième ligne doit être un entier (nombre d'arcs), "
                f"obtenu : '{lines[1]}'"
            )

        if nb_arcs < 0:
            raise ValueError(
                f"Le nombre d'arcs ne peut pas être négatif, obtenu : {nb_arcs}"
            )

        expected_lines = 2 + nb_arcs
        if len(lines) < expected_lines:
            raise ValueError(
                f"Le fichier déclare {nb_arcs} arc(s) mais ne contient que "
                f"{len(lines) - 2} ligne(s) d'arcs."
            )

        # Initialisation : INF partout sauf la diagonale qui vaut 0
        matrix: list[list[float]] = [
            [0.0 if i == j else INF for j in range(size)] for i in range(size)
        ]

        for arc_idx in range(nb_arcs):
            line = lines[2 + arc_idx]
            tokens = line.split()

            if len(tokens) != 3:
                raise ValueError(
                    f"L'arc n°{arc_idx + 1} doit contenir 3 valeurs "
                    f"(source destination valeur), obtenu : '{line}'"
                )

            try:
                src = int(tokens[0])
                dst = int(tokens[1])
            except ValueError:
                raise ValueError(
                    f"Les extrémités d'un arc doivent être des entiers, "
                    f"obtenu : '{tokens[0]}' et '{tokens[1]}' (arc n°{arc_idx + 1})"
                )

            if not (0 <= src < size and 0 <= dst < size):
                raise ValueError(
                    f"Sommet hors limites à l'arc n°{arc_idx + 1} : "
                    f"src={src}, dst={dst} (sommets valides : 0 à {size - 1})"
                )

            value = _parse_value(tokens[2], arc_idx, size)

            # En cas d'arcs multiples entre deux mêmes sommets, on garde le plus petit
            if value < matrix[src][dst]:
                matrix[src][dst] = value

        return cls(size, matrix)

    # ------------------------------------------------------------------
    # Affichage de la matrice d'adjacence
    # ------------------------------------------------------------------

    def display(self) -> None:
        """
        Affiche la matrice d'adjacence du graphe de manière lisible,
        avec alignement des colonnes et identification des sommets
        en en-tête de lignes et de colonnes.
        """
        col_w = self._column_width()

        # En-tête des colonnes
        header_pad = " " * (col_w + 3)
        col_headers = "  ".join(lbl.center(col_w) for lbl in self.labels)
        print(f"{header_pad}{col_headers}")

        # Ligne de séparation
        sep_width = (col_w + 3) + (col_w + 2) * self.size
        print("-" * sep_width)

        # Lignes de données
        for i, row in enumerate(self.matrix):
            row_label = self.labels[i].rjust(col_w)
            values = "  ".join(_fmt(v, col_w) for v in row)
            print(f"{row_label} |  {values}")

    # ------------------------------------------------------------------
    # Helper interne
    # ------------------------------------------------------------------

    def _column_width(self) -> int:
        """
        Calcule la largeur de colonne minimale pour que l'affichage
        soit lisible : suffisante pour les labels et pour les valeurs.
        """
        max_label = max(len(lbl) for lbl in self.labels)
        max_value = max(len(_fmt_raw(v)) for row in self.matrix for v in row)
        return max(max_label, max_value, 3)


# ----------------------------------------------------------------------
# Utilitaires de formatage et de parsing (portée module)
# ----------------------------------------------------------------------


def _parse_value(token: str, context_idx: int, size: int) -> float:
    """
    Convertit un token texte en valeur numérique.
    Accepte les entiers, les flottants et le mot-clé INF (insensible à la casse).
    Lève ValueError avec un message explicite en cas d'échec.
    """
    if token.upper() == "INF":
        return INF
    try:
        return float(token)
    except ValueError:
        raise ValueError(
            f"Valeur inattendue '{token}' (contexte : indice {context_idx}, "
            f"graphe à {size} sommet(s)). Attendu : un nombre entier, "
            "un flottant, ou 'INF'."
        )


def _fmt_raw(value: float) -> str:
    """
    Retourne la représentation textuelle d'une valeur sans padding.
    Les valeurs entières sont affichées sans décimales (ex. 3 et non 3.0).
    """
    if value == INF:
        return "INF"
    if value == int(value):
        return str(int(value))
    return str(value)


def _fmt(value: float, width: int) -> str:
    """Retourne la représentation textuelle d'une valeur centrée sur `width` caractères."""
    return _fmt_raw(value).center(width)
