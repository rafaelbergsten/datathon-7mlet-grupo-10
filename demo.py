"""Run the five golden-set cases against the saved policy."""

import argparse
import json
import tempfile
from pathlib import Path

from banditlab.data import ROOT
from banditlab.service import DecisionService

CASES = [
    {
        "case": "A",
        "contact": "cellular",
        "eligible": True,
        "consent": True,
        "expected_action": "short_message",
        "expected_reason": None,
        "rationale": "Mensagem curta é a melhor no cenário simulado deste contexto.",
    },
    {
        "case": "B",
        "contact": "telephone",
        "eligible": True,
        "consent": True,
        "expected_action": "guided_explanation",
        "expected_reason": None,
        "rationale": "Explicação guiada é a melhor no cenário simulado deste contexto.",
    },
    {
        "case": "C",
        "contact": "cellular",
        "eligible": False,
        "consent": True,
        "expected_action": None,
        "expected_reason": "ineligible",
        "rationale": "Cliente inelegível: nenhuma mensagem comercial é recomendada.",
    },
    {
        "case": "D",
        "contact": "telephone",
        "eligible": True,
        "consent": False,
        "expected_action": None,
        "expected_reason": "no_consent",
        "rationale": "Sem permissão de contato: nenhuma mensagem comercial é recomendada.",
    },
    {
        "case": "E",
        "contact": "unknown",
        "eligible": True,
        "consent": True,
        "expected_action": None,
        "expected_reason": "unknown_context",
        "rationale": "Contexto desconhecido: encaminhar para validação humana.",
    },
]


def golden_set():
    with tempfile.TemporaryDirectory() as directory:
        service = DecisionService(Path(directory) / "golden.sqlite", seed=42)
        results = []
        for case in CASES:
            response = service.recommend(
                **{k: case[k] for k in ("contact", "eligible", "consent")}, mode="greedy"
            )
            passed = (
                response["action"] == case["expected_action"]
                and response.get("reason") == case["expected_reason"]
            )
            results.append(
                {
                    **case,
                    "actual_action": response["action"],
                    "actual_reason": response.get("reason"),
                    "passed": passed,
                    "response": response,
                }
            )
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact", choices=["cellular", "telephone", "unknown"])
    parser.add_argument("--mode", choices=["sample", "greedy"], default="sample")
    args = parser.parse_args()
    if args.contact:
        service = DecisionService(ROOT / "runtime/decisions.sqlite")
        print(
            json.dumps(service.recommend(args.contact, True, True, args.mode), indent=2, ensure_ascii=False)
        )
    else:
        rows = golden_set()
        (ROOT / "results/golden_set.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        for row in rows:
            print(f"{row['case']}: {row['actual_action'] or row['actual_reason']} | passed={row['passed']}")
        if not all(row["passed"] for row in rows):
            raise SystemExit("Golden-set mismatch: inspect policy learning before presenting")
