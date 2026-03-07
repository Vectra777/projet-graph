from __future__ import annotations

from graph import INF, Graph, _fmt_raw


class FloydWarshallResult:
    """
    Contient les résultats de l'algorithme de Floyd-Warshall :
      - L : matrice des plus courts chemins (distances minimales)
      - P : matrice des prédécesseurs (pour reconstituer les chemins)
      - has_negative_cycle : True si un circuit absorbant a été détecté
    """

    def __init__(
        self,
        L: list[list[float]],
        P: list[list[int | None]],
        has_negative_cycle: bool,
        labels: list[str],
    ):
        self.L = L
        self.P = P
        self.has_negative_cycle = has_negative_cycle
        self.labels = labels
        self.size = len(L)

    # ------------------------------------------------------------------
    # Reconstitution d'un chemin
    # ------------------------------------------------------------------

    def get_path(self, src: int, dst: int) -> list[int] | None:
        """
        Reconstitue le chemin le plus court entre src et dst.
        Retourne une liste d'indices de sommets, ou None s'il n'existe pas.
        """
        if self.L[src][dst] == INF:
            return None

        path: list[int] = []
        current = dst

        while current != src:
            path.append(current)
            pred = self.P[src][current]
            if pred is None:
                return None
            current = pred

        path.append(src)
        path.reverse()
        return path

    def format_path(self, src: int, dst: int) -> str:
        """
        Retourne une représentation textuelle du chemin le plus court
        entre src et dst.
        """
        path = self.get_path(src, dst)
        if path is None:
            return f"Aucun chemin entre {self.labels[src]} et {self.labels[dst]}."

        steps = " -> ".join(self.labels[v] for v in path)
        cost = _fmt_raw(self.L[src][dst])
        return f"{steps}   (coût : {cost})"


# ------------------------------------------------------------------
# Algorithme principal
# ------------------------------------------------------------------


def floyd_warshall(graph: Graph, verbose: bool = True) -> FloydWarshallResult:
    """
    Exécute l'algorithme de Floyd-Warshall sur le graphe donné.

    Si verbose=True, affiche les matrices L et P à chaque itération.

    Retourne un FloydWarshallResult.
    """
    n = graph.size
    labels = graph.labels

    # --- Initialisation de L (copie de la matrice d'adjacence) ----------
    L: list[list[float]] = [[graph.matrix[i][j] for j in range(n)] for i in range(n)]

    # --- Initialisation de P (matrice des prédécesseurs) ----------------
    # P[i][j] = i  si i != j et qu'il existe un arc direct i->j
    # P[i][j] = None sinon (pas encore de chemin connu)
    P: list[list[int | None]] = [[None] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j and L[i][j] != INF:
                P[i][j] = i

    if verbose:
        print("\n" + "=" * 60)
        print("  INITIALISATION  (k = 0)")
        print("=" * 60)
        _print_L(L, labels, step=0)
        _print_P(P, labels, step=0)

    # --- Boucle principale k = 1..n -------------------------------------
    for k in range(n):
        for i in range(n):
            for j in range(n):
                via = _safe_add(L[i][k], L[k][j])
                if via < L[i][j]:
                    L[i][j] = via
                    P[i][j] = P[k][j]

        if verbose:
            print("\n" + "=" * 60)
            print(f"  ITÉRATION  k = {k + 1}  (sommet pivot : {labels[k]})")
            print("=" * 60)
            _print_L(L, labels, step=k + 1)
            _print_P(P, labels, step=k + 1)

    # --- Détection de circuit absorbant ---------------------------------
    has_negative_cycle = any(L[i][i] < 0 for i in range(n))

    return FloydWarshallResult(L, P, has_negative_cycle, labels)


# ------------------------------------------------------------------
# Affichage des matrices intermédiaires
# ------------------------------------------------------------------


def _print_L(L: list[list[float]], labels: list[str], step: int) -> None:
    """Affiche la matrice des distances L à l'étape donnée."""
    n = len(L)
    col_width = _matrix_col_width(L, labels)

    print(f"\n  Matrice L^{step} (distances minimales) :")
    _print_matrix(
        data=[[_fmt_raw(L[i][j]) for j in range(n)] for i in range(n)],
        labels=labels,
        col_width=col_width,
    )


def _print_P(P: list[list[int | None]], labels: list[str], step: int) -> None:
    """Affiche la matrice des prédécesseurs P à l'étape donnée."""
    n = len(P)
    col_width = _pred_col_width(P, labels)

    print(f"\n  Matrice P^{step} (prédécesseurs) :")

    def _pred_label(val: int | None) -> str:
        if val is None:
            return "-"
        return labels[val]

    _print_matrix(
        data=[[_pred_label(P[i][j]) for j in range(n)] for i in range(n)],
        labels=labels,
        col_width=col_width,
    )


def _print_matrix(data: list[list[str]], labels: list[str], col_width: int) -> None:
    """
    Affiche une matrice générique (valeurs déjà converties en str)
    avec des en-têtes de lignes et de colonnes.
    """
    n = len(labels)
    header_pad = " " * (col_width + 3)
    col_headers = "  ".join(lbl.center(col_width) for lbl in labels)
    print(f"{header_pad}{col_headers}")

    sep_width = (col_width + 3) + (col_width + 2) * n
    print("  " + "-" * (sep_width - 2))

    for i, row in enumerate(data):
        row_label = labels[i].rjust(col_width)
        values = "  ".join(v.center(col_width) for v in row)
        print(f"{row_label} |  {values}")


# ------------------------------------------------------------------
# Utilitaires internes
# ------------------------------------------------------------------


def _safe_add(a: float, b: float) -> float:
    """Addition sûre : renvoie INF si l'un des opérandes est INF."""
    if a == INF or b == INF:
        return INF
    return a + b


def _matrix_col_width(L: list[list[float]], labels: list[str]) -> int:
    max_label = max(len(lbl) for lbl in labels)
    max_value = max(
        len(_fmt_raw(L[i][j])) for i in range(len(L)) for j in range(len(L))
    )
    return max(max_label, max_value, 3)


def _pred_col_width(P: list[list[int | None]], labels: list[str]) -> int:
    max_label = max(len(lbl) for lbl in labels)
    return max(max_label, 3)
