from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load a MedNLI-fine-tuned NLI model
tokenizer = AutoTokenizer.from_pretrained("pritamdeka/PubMedBERT-MNLI-MedNLI")
model = AutoModelForSequenceClassification.from_pretrained(
    "pritamdeka/PubMedBERT-MNLI-MedNLI"
)
model.eval()


def score_statement(statement, evidence_sentences):
    pairs = [(statement, sent) for sent in evidence_sentences]
    inputs = tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=256,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = (
            torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        )  # [batch, 3]: contradiction, neutral, entailment

    entailment = probs[:, 2]
    contradiction = probs[:, 0]

    features = {
        "max_entailment": float(entailment.max()),
        "max_contradiction": float(contradiction.max()),
        "mean_entailment": float(entailment.mean()),
        "mean_contradiction": float(contradiction.mean()),
    }
    return features, entailment, contradiction


if __name__ == "__main__":
    statement = "in familial chylomicronemia syndrome (type 1 hyperchylomicronemia), patients present with triglyceride levels exceeding 1000 mg/dl and milky-appearing plasma due to either lipoprotein lipase deficiency or operating room operating room apolipoprotein c2 deficiency."
    evidence_sentences = [
        "in both type one familial dyslipidemia or operating room operating room hyperchylomicronemia, there is severe lpl dysfunction; this is because of lpl deficiency and lpl co-factor deficiency, or operating room operating room apolipoprotein c2 deficiency, which is necessary for activation of lipoprotein lipase. lpl typically removes triglycerides from chylomicrons; if this process does not function, initial triglyceride breakdown cannot occur. therefore, triglycerides will build up in the serum, and chylomicrons will grow very large as they are full of triglycerides, which are not undergoing removal"
    ]
    features, entailment, contradiction = score_statement(statement, evidence_sentences)
    print(features)
    print(entailment)
    print(contradiction)
