"""
MedAssist RAG — scripts/query.py
Script CLI pour interroger MedAssist en ligne de commande.

Usage:
    python scripts/query.py "Quels sont les critères de diagnostic de la tuberculose ?"
    python scripts/query.py --interactive
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.chain import MedAssistChain

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def print_response(response) -> None:
    print(f"\nQuestion : {response.question}")
    print(f"Spécialité détectée : {response.speciality_detected or 'générale'}")
    print(f"\nRéponse :\n{response.answer}")

    if response.sources:
        print(f"\nSources ({len(response.sources)}) :")
        for i, src in enumerate(response.sources, 1):
            print(f"  [{i}] {src['title']} — page {src['page']} (score: {src['score']:.3f})")

    print(f"\nLatence : {response.latency_ms:.0f}ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interroger MedAssist en CLI")
    parser.add_argument("question", nargs="?", help="Question médicale à poser")
    parser.add_argument("--interactive", action="store_true", help="Mode conversation continue")
    parser.add_argument("--k", type=int, default=5, help="Nombre de chunks à récupérer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.question and not args.interactive:
        print("Usage: python scripts/query.py \"votre question\" ou --interactive")
        sys.exit(1)

    chain = MedAssistChain(k=args.k, use_filter=True)

    if args.interactive:
        print("MedAssist — mode interactif (Ctrl+C pour quitter)\n")
        while True:
            try:
                question = input("Question : ").strip()
                if question:
                    print_response(chain.invoke(question))
                    print()
            except KeyboardInterrupt:
                print("\nFin de la session.")
                break
    else:
        print_response(chain.invoke(args.question))


if __name__ == "__main__":
    main()