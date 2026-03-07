import os
import sys
from io import StringIO
from pathlib import Path

from floyd_warshall import FloydWarshallResult, floyd_warshall
from graph import Graph

# Répertoire contenant les fichiers de graphes (relatif à ce script)
GRAPHS_DIR = Path(__file__).parent / "graphs"

# Répertoire de sortie pour les traces d'exécution
TRACES_DIR = Path(__file__).parent / "traces"

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║          ALGORITHME DE FLOYD-WARSHALL  :  Projet Graphes     ║
╚══════════════════════════════════════════════════════════════╝
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers d'interface bas niveau
# ──────────────────────────────────────────────────────────────────────────────


def clear() -> None:
    """Efface le terminal uniquement si on est dans un TTY interactif."""
    if os.isatty(0):
        os.system("cls" if os.name == "nt" else "clear")


def separator(char: str = "─", width: int = 64) -> None:
    """Affiche une ligne de séparation horizontale."""
    print(char * width)


def ask(prompt: str) -> str:
    """
    Lit une saisie utilisateur et la renvoie sans espaces superflus.
    Termine proprement le programme si stdin est fermé (EOF).
    """
    try:
        return input(prompt).strip()
    except EOFError:
        print("\n  (Fin de l'entrée standard : arrêt du programme.)")
        raise SystemExit(0)


def ask_yes_no(prompt: str) -> bool:
    """
    Pose une question fermée oui/non.
    Retourne True pour oui, False pour non.
    Accepte : o, oui, y, yes / n, non, no  (insensible à la casse).
    """
    while True:
        answer = ask(f"{prompt} [o/n] : ").lower()
        if answer in ("o", "oui", "y", "yes"):
            return True
        if answer in ("n", "non", "no"):
            return False
        print("  Réponse invalide : veuillez saisir 'o' (oui) ou 'n' (non).")


def ask_int(prompt: str, lo: int, hi: int) -> int:
    """
    Demande un entier compris entre lo et hi inclus.
    Répète la question tant que la saisie est invalide.
    """
    while True:
        raw = ask(prompt)
        try:
            value = int(raw)
        except ValueError:
            print(f"  Entrée invalide : veuillez saisir un entier entre {lo} et {hi}.")
            continue
        if lo <= value <= hi:
            return value
        print(f"  Valeur hors limites : veuillez saisir un entier entre {lo} et {hi}.")


# ──────────────────────────────────────────────────────────────────────────────
# Capture de sortie (pour la génération de trace)
# ──────────────────────────────────────────────────────────────────────────────


class OutputCapture:
    """
    Gestionnaire de contexte qui duplique stdout vers un StringIO interne.
    Tout ce qui est affiché avec print() pendant le bloc `with` est à la fois
    affiché à l'écran ET mémorisé pour pouvoir être sauvegardé dans un fichier.
    """

    def __init__(self) -> None:
        self._buffer = StringIO()
        self._original_stdout = sys.stdout

    def __enter__(self) -> "OutputCapture":
        sys.stdout = _TeeWriter(self._original_stdout, self._buffer)
        return self

    def __exit__(self, *_) -> None:
        sys.stdout = self._original_stdout

    def get_text(self) -> str:
        """Retourne tout ce qui a été capturé depuis l'entrée dans le contexte."""
        return self._buffer.getvalue()


class _TeeWriter:
    """
    Proxy d'écriture qui redirige chaque écriture vers deux destinations :
    l'écran (destination originale) et un tampon mémoire (StringIO).
    """

    def __init__(self, primary, secondary) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, text: str) -> int:
        self._primary.write(text)
        self._secondary.write(text)
        return len(text)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    # Délégation des attributs non définis explicitement (encoding, etc.)
    def __getattr__(self, name: str):
        return getattr(self._primary, name)


# ──────────────────────────────────────────────────────────────────────────────
# Sauvegarde des traces
# ──────────────────────────────────────────────────────────────────────────────


def save_trace(graph_number: int, text: str) -> Path:
    """
    Sauvegarde le contenu `text` dans un fichier de trace nommé
    `trace-graphe-<N>.txt` dans le répertoire TRACES_DIR.
    Crée le répertoire s'il n'existe pas encore.
    Retourne le chemin du fichier créé.
    """
    TRACES_DIR.mkdir(exist_ok=True)
    trace_path = TRACES_DIR / f"trace-graphe-{graph_number}.txt"
    trace_path.write_text(text, encoding="utf-8")
    return trace_path


# ──────────────────────────────────────────────────────────────────────────────
# Listage et chargement des fichiers graphes
# ──────────────────────────────────────────────────────────────────────────────


def list_graph_files() -> list[Path]:
    """
    Retourne la liste triée des fichiers .txt dans le dossier graphs/.
    Le tri est alphabétique, ce qui place graphe-1 avant graphe-2, etc.
    """
    if not GRAPHS_DIR.exists():
        return []
    return sorted(GRAPHS_DIR.glob("*.txt"))


def extract_graph_number(filepath: Path) -> str:
    """
    Tente d'extraire un numéro depuis le nom du fichier.
    Par exemple : 'graphe-3.txt' → '3'.
    Si aucun numéro n'est trouvé, retourne le nom complet sans extension.
    """
    stem = filepath.stem  # nom sans extension
    # On cherche les chiffres à la fin ou après un tiret/underscore
    import re

    match = re.search(r"[-_]?(\d+)$", stem)
    if match:
        return match.group(1)
    return stem


def display_graph_menu(files: list[Path]) -> None:
    """Affiche la liste numérotée des graphes disponibles."""
    separator()
    print("  Graphes disponibles :")
    separator()
    for idx, f in enumerate(files, start=1):
        graph_num = extract_graph_number(f)
        print(f"  {idx:>3}.  Graphe {graph_num:<4}  ({f.name})")
    separator()


def load_graph_interactive() -> tuple[Graph, int] | tuple[None, None]:
    """
    Propose à l'utilisateur de choisir un graphe parmi les fichiers disponibles
    ou de saisir un chemin personnalisé.

    Retourne un tuple (Graph, numéro_graphe) si le chargement réussit,
    ou (None, None) si l'utilisateur annule ou si une erreur survient.
    """
    files = list_graph_files()

    print()
    separator("═")
    print("  CHARGEMENT D'UN GRAPHE")
    separator("═")

    if files:
        display_graph_menu(files)
        print(f"  Entrez un numéro (1-{len(files)}) pour charger un graphe listé,")
        print("  ou appuyez sur Entrée pour saisir un chemin manuellement.")
        print("  Tapez '0' pour annuler.")
        separator()

        raw = ask("  Votre choix : ")

        if raw == "0":
            return None, None

        if raw == "":
            # Saisie manuelle d'un chemin
            filepath = ask("  Chemin du fichier : ")
            if not filepath:
                print("  Chemin vide : chargement annulé.")
                return None, None
            chosen_path = Path(filepath)
        else:
            try:
                idx = int(raw)
            except ValueError:
                print("  Entrée invalide chargement annulé.")
                return None, None
            if not (1 <= idx <= len(files)):
                print("  Numéro hors limites chargement annulé.")
                return None, None
            chosen_path = files[idx - 1]
    else:
        print("  Aucun fichier trouvé dans le dossier 'graphs/'.")
        filepath = ask("  Chemin du fichier graphe : ")
        if not filepath:
            print("  Chemin vide chargement annulé.")
            return None, None
        chosen_path = Path(filepath)

    # Extraction du numéro pour nommer la trace
    graph_number_str = extract_graph_number(chosen_path)
    try:
        graph_number = int(graph_number_str)
    except ValueError:
        graph_number = 0  # Fallback si le nom ne contient pas de numéro

    print()
    print(f"  Chargement de : {chosen_path.name} …")

    try:
        graph = Graph.from_file(str(chosen_path))
    except FileNotFoundError:
        print(f"  ERREUR : Fichier introuvable : '{chosen_path}'")
        return None, None
    except ValueError as exc:
        print(f"  ERREUR lors de la lecture du fichier : {exc}")
        return None, None

    print(f"  Graphe {graph_number_str} chargé avec succès ({graph.size} sommet(s)).")
    return graph, graph_number


# ──────────────────────────────────────────────────────────────────────────────
# Affichage des résultats finaux
# ──────────────────────────────────────────────────────────────────────────────


def display_result_summary(result: FloydWarshallResult) -> None:
    """
    Affiche la matrice finale L des plus courts chemins
    ainsi que le diagnostic de circuit absorbant.
    """
    from floyd_warshall import _print_L

    separator("═")
    print("  RÉSULTAT FINAL : Matrice L des plus courts chemins")
    separator("═")
    _print_L(result.L, result.labels, step=result.size)

    print()
    separator("═")
    if result.has_negative_cycle:
        print("  ⚠  CIRCUIT ABSORBANT DÉTECTÉ")
        print("     (Au moins un L[i][i] < 0 dans la matrice finale)")
        print("     Les plus courts chemins ne sont pas définis.")
    else:
        print("  ✔  Aucun circuit absorbant : les plus courts chemins sont valides.")
    separator("═")


# ──────────────────────────────────────────────────────────────────────────────
# Boucle de consultation des chemins
# ──────────────────────────────────────────────────────────────────────────────


def path_query_loop(result: FloydWarshallResult) -> None:
    """
    Boucle interactive permettant à l'utilisateur de consulter
    les chemins de valeur minimale entre deux sommets quelconques.

    Le schéma est celui demandé par le sujet :
        Chemin ?
        Si oui : Sommet de départ ? → Sommet d'arrivée ? → Affichage → Recommencer ?
        Si non : arrêter
    """
    n = result.size
    labels = result.labels

    print()
    separator("═")
    print("  CONSULTATION DES PLUS COURTS CHEMINS")
    separator("═")
    print("  Sommets disponibles :")
    for idx, lbl in enumerate(labels):
        print(f"    {idx:>3}  →  {lbl}")
    separator()

    while True:
        # "Chemin ?"
        if not ask_yes_no("\n  Voulez-vous afficher un chemin ?"):
            break

        # "Sommet de départ ?"
        print(f"  Sommet de départ  (0 – {n - 1}) :")
        src = ask_int("    Numéro : ", 0, n - 1)

        # "Sommet d'arrivée ?"
        print(f"  Sommet d'arrivée  (0 – {n - 1}) :")
        dst = ask_int("    Numéro : ", 0, n - 1)

        # Affichage du chemin
        print()
        separator()
        if src == dst:
            print(f"  Sommet de départ et d'arrivée identiques : {labels[src]}")
            print("  Coût du chemin trivial : 0")
        else:
            path_str = result.format_path(src, dst)
            print(f"  Chemin  {labels[src]} → {labels[dst]} :")
            print(f"  {path_str}")
        separator()
        # La boucle reboucle sur "Recommencer ?" via le ask_yes_no suivant


# ──────────────────────────────────────────────────────────────────────────────
# Traitement complet d'un graphe
# ──────────────────────────────────────────────────────────────────────────────


def process_graph(graph: Graph, graph_number: int) -> None:
    """
    Enchaîne toutes les étapes de traitement pour un graphe :
      (1) Affichage de la matrice d'adjacence initiale
      (2) Exécution de Floyd-Warshall avec affichage des matrices L et P
          à chaque itération
      (3) Diagnostic circuit absorbant
      (4) Consultation interactive des chemins les plus courts
          (seulement si aucun circuit absorbant)
    Les affichages sont capturés et sauvegardés dans un fichier de trace.
    """
    with OutputCapture() as capture:
        # ── Étape (1) : matrice initiale ───────────────────────────────
        print()
        separator("═")
        print(f"  GRAPHE {graph_number} : MATRICE D'ADJACENCE INITIALE")
        separator("═")
        graph.display()

        # ── Étape (2) : Floyd-Warshall ─────────────────────────────────
        print()
        separator("═")
        print("  EXÉCUTION DE L'ALGORITHME DE FLOYD-WARSHALL")
        separator("═")
        result = floyd_warshall(graph, verbose=True)

        # ── Étape (3) : résumé et diagnostic ──────────────────────────
        print()
        display_result_summary(result)

        # ── Étape (4) : consultation des chemins ───────────────────────
        if not result.has_negative_cycle:
            print()
            path_query_loop(result)
        else:
            print()
            print("  Le graphe contient un circuit absorbant :")
            print("  la consultation des chemins est désactivée.")

    # Sauvegarde de la trace (inclut tout ce qui a été affiché dans le bloc)
    trace_path = save_trace(graph_number, capture.get_text())
    print()
    print(f"  Trace sauvegardée : {trace_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Menu principal et boucle de contrôle
# ──────────────────────────────────────────────────────────────────────────────


def main_menu() -> None:
    """Affiche le menu principal."""
    print()
    separator("═")
    print("  MENU PRINCIPAL")
    separator("═")
    print("  1.  Analyser un graphe")
    print("  2.  Quitter")
    separator("═")


def main() -> None:
    """Point d'entrée principal : boucle de menu permettant d'analyser
    plusieurs graphes successifs sans relancer le programme."""
    clear()
    print(BANNER)

    while True:
        main_menu()
        choice = ask("  Votre choix : ")

        match choice:
            case "1":
                # Chargement du graphe choisi par l'utilisateur
                graph, graph_number = load_graph_interactive()

                if graph is not None and graph_number is not None:
                    process_graph(graph, graph_number)

                print()
                try:
                    input("  Appuyez sur Entrée pour revenir au menu principal…")
                except EOFError:
                    pass
                clear()
                print(BANNER)

            case "2":
                print()
                print("  Au revoir !")
                print()
                break

            case _:
                print("  Choix invalide : veuillez saisir 1 ou 2.")


if __name__ == "__main__":
    raise SystemExit(main())
