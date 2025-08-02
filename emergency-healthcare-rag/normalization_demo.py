#!/usr/bin/env python3
"""
Demonstration script for medical text normalization.

This script shows how the normalization system transforms both
documents and statements to improve retrieval performance.
"""

from pathlib import Path
from text_normalizer import normalize_medical_text, get_normalizer


def demo_basic_normalization():
    """Demonstrate basic text normalization features."""
    print("=" * 60)
    print("BASIC NORMALIZATION DEMO")
    print("=" * 60)

    test_cases = [
        "The patient has haemorrhage in the oesophagus",
        "PATIENT PRESENTS WITH    SEVERE   CHEST PAIN",
        "Type II Diabetes Mellitus with diabetic ketoacidosis",
        "The ECG shows ST-elevation MI (STEMI)",
        "Blood clot in lung causing SOB and CP",
        "Heart attack with irregular heartbeat (A-fib)",
    ]

    for text in test_cases:
        normalized = normalize_medical_text(text, is_query=False)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print()


def demo_synonym_mapping():
    """Demonstrate medical synonym mapping."""
    print("=" * 60)
    print("MEDICAL SYNONYM MAPPING DEMO")
    print("=" * 60)

    test_cases = [
        "Patient had a heart attack last week",
        "Stroke occurred in the brain",
        "Blood clot in lung causing breathing problems",
        "High blood pressure and irregular heartbeat",
        "Belly pain and food poisoning symptoms",
    ]

    for text in test_cases:
        normalized = normalize_medical_text(text, is_query=False)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print()


def demo_acronym_expansion():
    """Demonstrate acronym and abbreviation expansion."""
    print("=" * 60)
    print("ACRONYM EXPANSION DEMO")
    print("=" * 60)

    test_cases = [
        "Patient needs CBC, BMP, and PT/PTT",
        "ECG shows MI with elevated troponin",
        "PE diagnosed on CTA of chest",
        "COPD exacerbation requiring BiPAP",
        "DVT in right leg with elevated D-dimer",
        "GI bleeding with falling Hgb",
    ]

    for text in test_cases:
        normalized = normalize_medical_text(text, is_query=False)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print()


def demo_disease_normalization():
    """Demonstrate disease and condition normalization."""
    print("=" * 60)
    print("DISEASE NORMALIZATION DEMO")
    print("=" * 60)

    test_cases = [
        "Type II diabetes mellitus",
        "T2DM with diabetic complications",
        "Adult onset diabetes",
        "STEMI in the inferior wall",
        "Non-ST elevation MI",
        "Stage III breast cancer",
        "Heart failure with reduced EF",
    ]

    for text in test_cases:
        normalized = normalize_medical_text(text, is_query=False)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print()


def demo_procedure_normalization():
    """Demonstrate procedure and test normalization."""
    print("=" * 60)
    print("PROCEDURE NORMALIZATION DEMO")
    print("=" * 60)

    test_cases = [
        "EKG shows abnormal rhythm",
        "CT scan of the abdomen",
        "Ultrasound of the heart",
        "Blood work shows elevated enzymes",
        "Chest X-ray reveals pneumonia",
        "Heart catheterization planned",
    ]

    for text in test_cases:
        normalized = normalize_medical_text(text, is_query=False)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print()


def demo_anatomical_expansion():
    """Demonstrate anatomical terminology expansion."""
    print("=" * 60)
    print("ANATOMICAL EXPANSION DEMO")
    print("=" * 60)

    test_cases = [
        "Heart failure with cardiac dysfunction",
        "Kidney disease with renal impairment",
        "Lung infection affecting respiratory system",
        "Brain injury with neurological deficits",
        "Liver failure with hepatic encephalopathy",
    ]

    for text in test_cases:
        normalized = normalize_medical_text(text, is_query=False)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print()


def demo_query_vs_document_normalization():
    """Demonstrate difference between query and document normalization."""
    print("=" * 60)
    print("QUERY vs DOCUMENT NORMALIZATION")
    print("=" * 60)

    test_text = "Heart attack with cardiac arrest"

    query_normalized = normalize_medical_text(test_text, is_query=True)
    document_normalized = normalize_medical_text(test_text, is_query=False)

    print(f"Original text:        {test_text}")
    print(f"Query normalized:     {query_normalized}")
    print(f"Document normalized:  {document_normalized}")
    print()
    print("Note: Document normalization includes anatomical expansion")
    print("for better recall, while query normalization is lighter.")
    print()


def demo_real_medical_text():
    """Demonstrate normalization on realistic medical text."""
    print("=" * 60)
    print("REAL MEDICAL TEXT DEMO")
    print("=" * 60)

    medical_text = """
    Patient is a 65-year-old male with Type II DM, HTN, and CAD who presents 
    to the ED with acute onset CP and SOB. EKG shows ST-elevation in leads II, 
    III, and aVF consistent with inferior STEMI. Troponin I elevated at 5.2. 
    CXR shows pulmonary oedema. Echo reveals decreased EF. Patient taken for 
    emergent PCI. CBC shows Hgb 8.5, WBC 12.0. BNP elevated at 850.
    """

    normalized = normalize_medical_text(medical_text.strip(), is_query=False)

    print("Original medical text:")
    print(medical_text.strip())
    print("\nNormalized text:")
    print(normalized)
    print()


def main():
    """Run all demonstration functions."""
    print("MEDICAL TEXT NORMALIZATION DEMONSTRATION")
    print("This script demonstrates the comprehensive medical text")
    print("normalization capabilities of the healthcare RAG system.")
    print()

    demo_basic_normalization()
    demo_synonym_mapping()
    demo_acronym_expansion()
    demo_disease_normalization()
    demo_procedure_normalization()
    demo_anatomical_expansion()
    demo_query_vs_document_normalization()
    demo_real_medical_text()

    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("The normalization system is ready to improve")
    print("retrieval performance in your healthcare RAG system!")


if __name__ == "__main__":
    main()
