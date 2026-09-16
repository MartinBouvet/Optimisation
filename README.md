# Optimart

Tu colles un pavé RAG. Il **jette les phrases inutiles** et **garde celles qui servent à répondre**.

Pas un résumé ChatGPT : un petit modèle local + un knapsack. Tu changes l’URL de l’API, c’est tout.

## La démo (30 secondes)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_demo.py
```

Ouvre [http://127.0.0.1:8000](http://127.0.0.1:8000), clique **RAG interne**, puis **Couper les tokens**.

Tu dois voir :
- **Optimart** : le `480 000 euros` reste (vert)
- **Coupe bête** : le chiffre disparaît (rose) — on a juste gardé le début du pavé

Sur 64 tests (Hotpot + cas contrôlés), même budget ~46 % de tokens en moins :

| | Tokens en moins | La bonne info est encore là |
|---|---|---|
| Tout envoyer | 0 % | 98 % |
| Couper le début / la fin | ~46 % | 66 % |
| **Optimart** | **~46 %** | **91 %** |

≈ **96 $** d’économie / 100 000 appels gpt-4o (tokens d’entrée). Ce n’est pas une note ChatGPT : on vérifie si la réponse est encore dans le texte envoyé.

## Proxy API

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="ta-cle")
```

Dry-run par défaut (montre le texte filtré). Relais réel : `python scripts/run_proxy.py --live`.

## Entraîner

Sans checkpoint, la démo utilise MiniLM en zéro-shot (cosine question ↔ phrase). Avec un checkpoint local, le scorer fine-tuné prend le relais.

```bash
python scripts/prepare_dataset.py --config configs/data_hotpot_mini.yaml --datasets hotpotqa --max-examples 800
python scripts/train.py --config-train configs/train_minilm.yaml
python scripts/run_benchmarks.py --checkpoint artifacts/train_minilm/best.pt --hotpot 40
```

Détail : [`paper/BENCHMARK.md`](paper/BENCHMARK.md).
