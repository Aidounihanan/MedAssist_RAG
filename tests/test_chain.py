"""Tests pour src/rag/chain.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.chain import detect_speciality, is_hors_scope


def test_detect_speciality_tuberculose():
    assert detect_speciality("Quels sont les critères de la tuberculose ?") == "infectiologie"


def test_detect_speciality_cardiologie():
    assert detect_speciality("Quelles sont les cibles tensionnelles pour l'HTA ?") == "cardiologie"


def test_detect_speciality_returns_none_for_generic_question():
    assert detect_speciality("Bonjour, comment allez-vous ?") is None


def test_is_hors_scope_detects_prescription_request():
    assert is_hors_scope("Quel médicament dois-je prescrire à ce patient ?") is True


def test_is_hors_scope_detects_diagnostic_request():
    assert is_hors_scope("Ai-je un diagnostic de diabète ?") is True


def test_is_hors_scope_false_for_protocol_question():
    assert is_hors_scope("Quel est le protocole de prise en charge de la tuberculose ?") is False


def test_is_hors_scope_false_for_diagnostic_criteria_question():
    assert is_hors_scope("Quels sont les critères de diagnostic de la tuberculose ?") is False
