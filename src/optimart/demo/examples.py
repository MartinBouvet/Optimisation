"""Needle-in-haystack demos: the fact is buried so naive truncation fails."""

from __future__ import annotations

from typing import Any

_FILLER_OPS = """
Compte-rendu hebdo — équipe ops, 4 mars 2026.
Le sprint a porté sur le monitoring Datadog, le renouvellement des certificats TLS et le nettoyage des dashboards Grafana.
Plusieurs tickets P3 restent ouverts : latence du worker d'export CSV, alerts Slack trop bruyantes, rotation des clés S3.
Alice a demandé un point sur la capacité du cluster Kubernetes avant Black Friday.
Bob a mergé le correctif de pagination sur l'API billing. Carol relance le fournisseur CDN pour une erreur collatérale sur les assets FR.
Le taux d'erreur 5xx est redescendu à 0,12 % après le rollback du déploiement de lundi.
On a aussi parlé du nouvel outil de feature flags, sans décision. La doc interne a 40 pages de procédures d'astreinte que personne ne relit.
Le NPS interne du support est à 41. Les files Zendesk dépassent 6 heures le vendredi. On prévoit un runbook incident pour le trimestre.
""".strip()

_FILLER_LEGAL = """
Annexe RH — politique de télétravail 2025.
Le forfait jours concerne les cadres autonomes. Les notes de frais voyage se déclarent sous 10 jours.
La charte informatique interdit le shadow IT. Le CSE a validé la grille d'astreinte.
Un rappel : les données clients ne quittent pas l'UE. Les audits SOC2 sont planifiés en novembre.
Le siège reste à Lyon Part-Dieu. L'open-space du 4e a été réaménagé. La cantine ferme à 14h30.
""".strip()

_FILLER_NOISE = """
Wikipedia noise: The Nile flows north through Egypt. Kilimanjaro is in Tanzania.
Saturn's rings are ice and rock. Helium is used in balloons. Granite is an igneous rock.
Python is popular in data analysis. Beethoven wrote nine numbered symphonies.
The Pacific is the largest ocean. Mars has two small moons, Phobos and Deimos.
""".strip()


def examples() -> list[dict[str, Any]]:
    renewal = "\n\n".join(
        [
            _FILLER_OPS,
            _FILLER_LEGAL,
            _FILLER_OPS.replace("4 mars 2026", "11 mars 2026"),
            _FILLER_LEGAL,
            "The March renewal was billed at 480 000 euros excluding tax for 24 months.",
            _FILLER_NOISE,
            _FILLER_OPS.replace("Alice", "Diane").replace("Bob", "Éric"),
        ]
    )
    photosynthesis = "\n\n".join(
        [
            _FILLER_NOISE,
            _FILLER_OPS,
            _FILLER_LEGAL,
            _FILLER_NOISE,
            "Plants absorb carbon dioxide during photosynthesis to build sugars.",
            _FILLER_OPS,
        ]
    )
    return [
        {
            "id": "renouvellement",
            "title": "RAG interne — le chiffre est à la fin",
            "question": "What amount was billed for the March renewal?",
            "context": renewal,
        },
        {
            "id": "photosynthesis",
            "title": "Needle in haystack — answer buried",
            "question": "which gas do plants absorb during photosynthesis?",
            "context": photosynthesis,
        },
    ]
