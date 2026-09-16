# Optimart

Filtre local devant ChatGPT (et les API compatibles).  
Tu changes l’adresse du serveur. Optimart enlève les phrases inutiles, puis envoie le reste.

## Résultats (64 tests)

| | Tokens en moins | La bonne info est encore là |
|---|---|---|
| Tout envoyer | 0 % | 98 % |
| Couper bêtement le début ou la fin | ~46 % | 66 % |
| **Optimart** | **~46 %** | **91 %** |

Sur 100 000 appels gpt-4o, ça fait environ **96 $** d’économie sur les mots en entrée.

Ce n’est pas une note ChatGPT : on vérifie si la réponse est encore dans le texte envoyé.

## Utilisation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_proxy.py
```

Le serveur tourne sur `http://127.0.0.1:8000`.

Dans ton code, tu ne changes que ça :

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="ta-cle",  # transmise au vrai fournisseur si le proxy n'est pas en dry-run
)
```

Par défaut le proxy est en **dry-run** : il montre le texte filtré, sans appeler l’API. Pour relayer vraiment :

```bash
python scripts/run_proxy.py --live
```

## Entraîner / mesurer

```bash
python scripts/prepare_dataset.py --config configs/data_hotpot_mini.yaml --datasets hotpotqa --max-examples 800
python scripts/train.py --config-train configs/train_minilm.yaml
python scripts/run_benchmarks.py --checkpoint artifacts/train_minilm/best.pt --hotpot 40
```

Détail des chiffres : [`paper/BENCHMARK.md`](paper/BENCHMARK.md).
