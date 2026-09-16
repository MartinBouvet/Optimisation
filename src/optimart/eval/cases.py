"""Reproducible RAG-style cases. Gold answers sit in known positions among distractors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    answers: list[str]
    supports: list[str]
    distractors: list[str]
    place: str  # head | mid | tail — where supporting facts sit

    def messages(self) -> list[dict[str, str]]:
        if self.place == "head":
            body = self.supports + self.distractors
        elif self.place == "tail":
            body = self.distractors + self.supports
        else:
            cut = max(len(self.distractors) // 2, 1)
            body = self.distractors[:cut] + self.supports + self.distractors[cut:]
        docs = " ".join(body)
        return [
            {
                "role": "system",
                "content": "You are a careful assistant. Answer using only the provided documents. Never invent sources.",
            },
            {"role": "user", "content": f"{docs}\nQuestion: {self.question}"},
        ]


_DISTRACTORS = [
    "The Nile is often listed among the longest rivers on Earth and flows through several African countries.",
    "Mount Kilimanjaro is a dormant volcano in Tanzania and a popular trekking destination.",
    "The Amazon rainforest contains a large share of terrestrial species and spans several countries.",
    "Saturn is a gas giant famous for a ring system made of ice and rock particles.",
    "Python is a high-level programming language widely used in data analysis and automation.",
    "The Pacific Ocean is the largest ocean basin and covers more area than all land combined.",
    "Granite is a coarse-grained igneous rock commonly used in construction and sculpture.",
    "Helium is a noble gas used in balloons because it is lighter than air and nonflammable.",
    "Mars has two small moons named Phobos and Deimos that orbit close to the planet.",
    "Beethoven completed nine numbered symphonies, the last of which includes a choral finale.",
]


def crafted_cases() -> list[EvalCase]:
    return [
        EvalCase(
            "france_mid",
            "What is the capital of France?",
            ["Paris"],
            ["Paris is the capital of France and stands on the river Seine."],
            _DISTRACTORS,
            "mid",
        ),
        EvalCase(
            "hamlet_head",
            "Who wrote Hamlet?",
            ["William Shakespeare", "Shakespeare"],
            ["William Shakespeare wrote Hamlet around 1600 as a revenge tragedy."],
            _DISTRACTORS,
            "head",
        ),
        EvalCase(
            "co2_tail",
            "What gas do plants absorb during photosynthesis?",
            ["carbon dioxide", "CO2"],
            ["Plants absorb carbon dioxide during photosynthesis to build sugars."],
            _DISTRACTORS,
            "tail",
        ),
        EvalCase(
            "everest_mid",
            "Where is Mount Everest?",
            ["Nepal", "China"],
            ["Mount Everest lies on the border of Nepal and China in the Himalayas."],
            _DISTRACTORS[1:],  # drop Kilimanjaro to avoid mountain confusion in supports only
            "mid",
        ),
        EvalCase(
            "canberra_mid",
            "What is the capital of Australia?",
            ["Canberra"],
            ["Canberra is the capital city of Australia, not Sydney or Melbourne."],
            _DISTRACTORS + ["Sydney is the largest city in Australia and a major harbour."],
            "mid",
        ),
        EvalCase(
            "monalisa_tail",
            "Who painted the Mona Lisa?",
            ["Leonardo da Vinci", "Leonardo"],
            ["Leonardo da Vinci painted the Mona Lisa during the Italian Renaissance."],
            _DISTRACTORS + ["Pablo Picasso was a Spanish painter who co-founded Cubism."],
            "tail",
        ),
        EvalCase(
            "armstrong_head",
            "Who was the first person to walk on the Moon?",
            ["Neil Armstrong", "Armstrong"],
            ["Neil Armstrong was the first person to walk on the Moon in 1969."],
            _DISTRACTORS,
            "head",
        ),
        EvalCase(
            "h2o_mid",
            "What is the chemical formula of water?",
            ["H2O", "h2o"],
            ["The chemical formula of water is H2O, two hydrogen atoms and one oxygen atom."],
            _DISTRACTORS,
            "mid",
        ),
        EvalCase(
            "washington_tail",
            "Who was the first president of the United States?",
            ["George Washington", "Washington"],
            ["George Washington was the first president of the United States."],
            _DISTRACTORS,
            "tail",
        ),
        EvalCase(
            "jupiter_head",
            "Which planet is the largest in the Solar System?",
            ["Jupiter"],
            ["Jupiter is the largest planet in the Solar System, a gas giant with a Great Red Spot."],
            _DISTRACTORS,
            "head",
        ),
        EvalCase(
            "curie_mid",
            "Who discovered radium?",
            ["Marie Curie", "Curie"],
            ["Marie Curie isolated radium and polonium while researching radioactivity."],
            _DISTRACTORS,
            "mid",
        ),
        EvalCase(
            "greatwall_tail",
            "In which country is the Great Wall?",
            ["China"],
            ["The Great Wall is a series of fortifications in China built over many centuries."],
            _DISTRACTORS,
            "tail",
        ),
        EvalCase(
            "guido_mid",
            "Who created the Python programming language?",
            ["Guido van Rossum", "van Rossum"],
            ["Guido van Rossum created the Python programming language in the late 1980s."],
            _DISTRACTORS,
            "mid",
        ),
        EvalCase(
            "einstein_head",
            "Who developed the theory of relativity?",
            ["Albert Einstein", "Einstein"],
            ["Albert Einstein developed the special and general theories of relativity."],
            _DISTRACTORS,
            "head",
        ),
        EvalCase(
            "dna_tail",
            "What is the shape of the DNA molecule?",
            ["double helix"],
            ["DNA is shaped like a double helix, two strands wound around each other."],
            _DISTRACTORS,
            "tail",
        ),
        EvalCase(
            "light_mid",
            "What is the approximate speed of light in vacuum?",
            ["299792458", "300000000"],
            ["The speed of light in vacuum is 299792458 metres per second, a defined constant."],
            _DISTRACTORS,
            "mid",
        ),
        EvalCase(
            "tesla_head",
            "Who invented the alternating-current induction motor popularized in the 1880s?",
            ["Nikola Tesla", "Tesla"],
            ["Nikola Tesla invented a practical alternating-current induction motor in the 1880s."],
            _DISTRACTORS,
            "head",
        ),
        EvalCase(
            "kepler_mid",
            "Who formulated three laws of planetary motion?",
            ["Johannes Kepler", "Kepler"],
            ["Johannes Kepler formulated three laws of planetary motion from Tycho Brahe's data."],
            _DISTRACTORS,
            "mid",
        ),
        EvalCase(
            "amazon_river_tail",
            "Which river has the largest discharge into the ocean?",
            ["Amazon"],
            ["The Amazon River has the largest discharge of any river into the ocean."],
            _DISTRACTORS,
            "tail",
        ),
        EvalCase(
            "tokyo_head",
            "What is the capital of Japan?",
            ["Tokyo"],
            ["Tokyo is the capital of Japan and one of the most populous metropolitan areas."],
            _DISTRACTORS,
            "head",
        ),
        EvalCase(
            "sahara_mid",
            "What is the largest hot desert on Earth?",
            ["Sahara"],
            ["The Sahara is the largest hot desert on Earth, covering much of North Africa."],
            _DISTRACTORS,
            "mid",
        ),
        EvalCase(
            "photosynthesis_tail",
            "What energy source do plants use for photosynthesis?",
            ["sunlight", "sun"],
            ["Plants use sunlight as the energy source that drives photosynthesis."],
            _DISTRACTORS,
            "tail",
        ),
        EvalCase(
            "newton_head",
            "Who formulated the law of universal gravitation?",
            ["Isaac Newton", "Newton"],
            ["Isaac Newton formulated the law of universal gravitation in the Principia."],
            _DISTRACTORS,
            "head",
        ),
        EvalCase(
            "louvre_mid",
            "In which city is the Louvre museum?",
            ["Paris"],
            ["The Louvre museum is located in Paris and houses the Mona Lisa."],
            _DISTRACTORS,
            "mid",
        ),
    ]
