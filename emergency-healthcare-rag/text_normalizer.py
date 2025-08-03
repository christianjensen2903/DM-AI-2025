"""
Comprehensive medical text normalization module for healthcare RAG system.

This module provides normalization for medical documents and statements including:
- Basic text normalization (lowercase, whitespace, ASCII folding)
- Medical synonym mapping
- Acronym and abbreviation expansion
- Disease/condition normalization
- Procedure and test name standardization
- Anatomical terminology normalization
- Spelling variants (British/American)
"""

import re
import unicodedata


class MedicalTextNormalizer:
    """Comprehensive medical text normalizer with medical terminology handling."""

    def __init__(self):
        """Initialize the normalizer with medical dictionaries and mappings."""
        self._init_synonym_mappings()
        self._init_acronym_mappings()
        self._init_disease_mappings()
        self._init_procedure_mappings()
        self._init_anatomical_mappings()
        self._init_spelling_variants()
        self._compile_patterns()

    def _init_synonym_mappings(self):
        """Initialize medical synonym mappings."""
        self.medical_synonyms = {
            # Cardiac conditions
            "heart attack": "myocardial infarction",
            "heart failure": "cardiac failure",
            "irregular heartbeat": "arrhythmia",
            "chest pain": "angina",
            # Neurological conditions
            "stroke": "cerebrovascular accident",
            "brain attack": "cerebrovascular accident",
            "seizure": "epileptic seizure",
            "fit": "seizure",
            # Respiratory conditions
            "blood clot in lung": "pulmonary embolism",
            "lung infection": "pneumonia",
            "breathing difficulty": "dyspnea",
            "shortness of breath": "dyspnea",
            "difficulty breathing": "dyspnea",
            # Gastrointestinal
            "stomach ache": "abdominal pain",
            "belly pain": "abdominal pain",
            "food poisoning": "gastroenteritis",
            # General medical terms
            "high blood pressure": "hypertension",
            "low blood pressure": "hypotension",
            "high blood sugar": "hyperglycemia",
            "low blood sugar": "hypoglycemia",
            "blood poisoning": "sepsis",
            "infection in blood": "sepsis",
            "kidney failure": "renal failure",
            "liver failure": "hepatic failure",
            # Emergency conditions
            "shock": "circulatory shock",
            "bleeding": "hemorrhage",
            "broken bone": "fracture",
            "burn": "thermal injury",
        }

    def _init_acronym_mappings(self):
        """Initialize medical acronym and abbreviation mappings."""
        self.medical_acronyms = {
            # Cardiac
            "mi": "myocardial infarction",
            "cad": "coronary artery disease",
            "chf": "congestive heart failure",
            "afib": "atrial fibrillation",
            "a-fib": "atrial fibrillation",
            "vtach": "ventricular tachycardia",
            "vfib": "ventricular fibrillation",
            "cpr": "cardiopulmonary resuscitation",
            "acls": "advanced cardiac life support",
            # Respiratory
            "pe": "pulmonary embolism",  # context-dependent
            "copd": "chronic obstructive pulmonary disease",
            "ards": "acute respiratory distress syndrome",
            "cpap": "continuous positive airway pressure",
            "bipap": "bilevel positive airway pressure",
            # Neurological
            "cva": "cerebrovascular accident",
            "tia": "transient ischemic attack",
            "gcs": "glasgow coma scale",
            "icp": "intracranial pressure",
            "tbi": "traumatic brain injury",
            # Laboratory/Diagnostics
            "cbc": "complete blood count",
            "bmp": "basic metabolic panel",
            "cmp": "comprehensive metabolic panel",
            "pt": "prothrombin time",
            "ptt": "partial thromboplastin time",
            "inr": "international normalized ratio",
            "bnp": "b-type natriuretic peptide",
            "troponin": "cardiac troponin",
            "crp": "c-reactive protein",
            "esr": "erythrocyte sedimentation rate",
            "abg": "arterial blood gas",
            "ua": "urinalysis",
            "lft": "liver function tests",
            # Imaging
            "ct": "computed tomography",
            "mri": "magnetic resonance imaging",
            "cta": "computed tomography angiography",
            "ecg": "electrocardiogram",
            "ekg": "electrocardiogram",
            "echo": "echocardiogram",
            "cxr": "chest x-ray",
            # Procedures
            "pci": "percutaneous coronary intervention",
            "cabg": "coronary artery bypass graft",
            "icd": "implantable cardioverter defibrillator",
            "ppm": "permanent pacemaker",
            # Medical specialties and departments
            "ed": "emergency department",
            "er": "emergency room",
            "icu": "intensive care unit",
            "ccu": "cardiac care unit",
            "or": "operating room",
            # Medications and treatments
            "iv": "intravenous",
            "po": "per os",  # by mouth
            "im": "intramuscular",
            "sc": "subcutaneous",
            "prn": "pro re nata",  # as needed
            "bid": "bis in die",  # twice daily
            "tid": "ter in die",  # three times daily
            "qid": "quater in die",  # four times daily
            # Common conditions
            "dm": "diabetes mellitus",
            "htn": "hypertension",
            "dvt": "deep vein thrombosis",
            "uti": "urinary tract infection",
            "uri": "upper respiratory infection",
            "gi": "gastrointestinal",
            "gu": "genitourinary",
            "msk": "musculoskeletal",
            # Emergency/Trauma
            "gsw": "gunshot wound",
            "mvc": "motor vehicle collision",
            "mva": "motor vehicle accident",
            "loc": "loss of consciousness",
            "sob": "shortness of breath",
            "cp": "chest pain",
            "abd": "abdominal",
        }

    def _init_disease_mappings(self):
        """Initialize disease and condition normalizations."""
        self.disease_normalizations = {
            # Diabetes variants
            "type ii diabetes mellitus": "type 2 diabetes mellitus",
            "t2dm": "type 2 diabetes mellitus",
            "t2d": "type 2 diabetes mellitus",
            "type ii diabetes": "type 2 diabetes mellitus",
            "type 2 diabetes": "type 2 diabetes mellitus",
            "diabetes type 2": "type 2 diabetes mellitus",
            "adult onset diabetes": "type 2 diabetes mellitus",
            "non-insulin dependent diabetes": "type 2 diabetes mellitus",
            "niddm": "type 2 diabetes mellitus",
            "type i diabetes mellitus": "type 1 diabetes mellitus",
            "t1dm": "type 1 diabetes mellitus",
            "t1d": "type 1 diabetes mellitus",
            "type i diabetes": "type 1 diabetes mellitus",
            "type 1 diabetes": "type 1 diabetes mellitus",
            "diabetes type 1": "type 1 diabetes mellitus",
            "juvenile diabetes": "type 1 diabetes mellitus",
            "insulin dependent diabetes": "type 1 diabetes mellitus",
            "iddm": "type 1 diabetes mellitus",
            # Myocardial infarction variants
            "stemi": "st-elevation myocardial infarction",
            "nstemi": "non-st-elevation myocardial infarction",
            "st elevation mi": "st-elevation myocardial infarction",
            "non-st elevation mi": "non-st-elevation myocardial infarction",
            "heart attack": "myocardial infarction",
            # Cancer staging normalization
            "stage iii": "stage 3",
            "stage ii": "stage 2",
            "stage iv": "stage 4",
            "stage i": "stage 1",
            # Heart failure classifications
            "heart failure with reduced ejection fraction": "hfref",
            "heart failure with preserved ejection fraction": "hfpef",
            "systolic heart failure": "hfref",
            "diastolic heart failure": "hfpef",
        }

    def _init_procedure_mappings(self):
        """Initialize procedure and test name normalizations."""
        self.procedure_normalizations = {
            # Imaging equivalences (keep ECG/EKG as is since acronym expansion handles them)
            "ct scan": "computed tomography",
            "cat scan": "computed tomography",
            "mri scan": "magnetic resonance imaging",
            "ultrasound": "ultrasonography",
            "x-ray": "radiography",
            "chest x-ray": "chest radiography",
            # Laboratory tests
            "blood work": "laboratory studies",
            "blood test": "laboratory studies",
            "lab work": "laboratory studies",
            "blood gases": "arterial blood gas analysis",
            "cardiac enzymes": "cardiac biomarkers",
            "liver enzymes": "hepatic enzymes",
            # Procedures
            "heart catheterization": "cardiac catheterization",
            "angioplasty": "percutaneous coronary intervention",
            "balloon angioplasty": "percutaneous coronary intervention",
            "stent placement": "coronary stent insertion",
            "bypass surgery": "coronary artery bypass graft",
            "open heart surgery": "cardiac surgery",
        }

    def _init_anatomical_mappings(self):
        """Initialize anatomical terminology normalizations."""
        self.anatomical_mappings = {
            # Heart/Cardiac
            "heart": ["heart", "cardiac", "coronary"],
            "cardiac": ["heart", "cardiac", "coronary"],
            "coronary": ["heart", "cardiac", "coronary"],
            # Kidney/Renal
            "kidney": ["kidney", "renal", "nephric"],
            "renal": ["kidney", "renal", "nephric"],
            "nephric": ["kidney", "renal", "nephric"],
            # Liver/Hepatic
            "liver": ["liver", "hepatic"],
            "hepatic": ["liver", "hepatic"],
            # Lung/Pulmonary
            "lung": ["lung", "pulmonary", "respiratory"],
            "pulmonary": ["lung", "pulmonary", "respiratory"],
            "respiratory": ["lung", "pulmonary", "respiratory"],
            # Brain/Cerebral/Neurological
            "brain": ["brain", "cerebral", "neurological", "neural"],
            "cerebral": ["brain", "cerebral", "neurological", "neural"],
            "neurological": ["brain", "cerebral", "neurological", "neural"],
            "neural": ["brain", "cerebral", "neurological", "neural"],
            # Stomach/Gastric
            "stomach": ["stomach", "gastric"],
            "gastric": ["stomach", "gastric"],
            # Blood/Hematologic
            "blood": ["blood", "hematologic", "hemic"],
            "hematologic": ["blood", "hematologic", "hemic"],
            "hemic": ["blood", "hematologic", "hemic"],
        }

    def _init_spelling_variants(self):
        """Initialize British/American spelling variants."""
        self.spelling_variants = {
            # British to American
            "haemorrhage": "hemorrhage",
            "haemoglobin": "hemoglobin",
            "haematuria": "hematuria",
            "haematoma": "hematoma",
            "anaemia": "anemia",
            "anaesthetic": "anesthetic",
            "oesophagus": "esophagus",
            "oedema": "edema",
            "diarrhoea": "diarrhea",
            "colour": "color",
            "favour": "favor",
            "behaviour": "behavior",
            "centre": "center",
            "litre": "liter",
            "metre": "meter",
            "organised": "organized",
            "realise": "realize",
            "recognise": "recognize",
            "hospitalisation": "hospitalization",
            "localisation": "localization",
            "traumatised": "traumatized",
            "catheterise": "catheterize",
            "emphasise": "emphasize",
            "analyse": "analyze",
            "paralyse": "paralyze",
            "defence": "defense",
            "licence": "license",
            "practice": "practice",  # as noun in British, verb in American
            "practise": "practice",  # as verb in British
        }

    def _compile_patterns(self):
        """Compile regex patterns for efficient text processing."""
        # Pattern for whitespace normalization
        self.whitespace_pattern = re.compile(r"\s+")

        # Pattern for removing extra punctuation
        self.punctuation_pattern = re.compile(r"[^\w\s\-\.]")

        # Pattern for word boundaries for whole word replacement
        self.word_boundary_pattern = lambda word: re.compile(
            r"\b" + re.escape(word) + r"\b", re.IGNORECASE
        )

    def normalize_text(self, text: str, expand_anatomical: bool = True) -> str:
        """
        Apply comprehensive normalization to medical text.

        Args:
            text: Input text to normalize
            expand_anatomical: Whether to expand anatomical synonyms

        Returns:
            Normalized text
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: ASCII folding / diacritics removal
        text = self._ascii_fold(text)

        # Step 2: Lowercase conversion
        text = text.lower()

        # # Step 3: British/American spelling normalization
        text = self._normalize_spelling_variants(text)

        # # Step 4: Medical synonym mapping
        text = self._apply_synonym_mapping(text)

        # Step 5: Acronym and abbreviation expansion
        text = self._expand_acronyms(text)

        # Step 6: Disease/condition normalization
        text = self._normalize_diseases(text)

        # Step 7: Procedure/test normalization
        text = self._normalize_procedures(text)

        # Step 8: Anatomical terminology expansion (optional)
        if expand_anatomical:
            text = self._expand_anatomical_terms(text)

        # # Step 9: Whitespace normalization (final cleanup)
        # text = self._normalize_whitespace(text)

        return text.strip()

    def _ascii_fold(self, text: str) -> str:
        """Remove diacritics and convert to ASCII."""
        # Normalize to NFD (decomposed form)
        nfd_text = unicodedata.normalize("NFD", text)
        # Filter out combining characters (diacritics)
        ascii_text = "".join(c for c in nfd_text if unicodedata.category(c) != "Mn")
        return ascii_text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace to single spaces."""
        return self.whitespace_pattern.sub(" ", text)

    def _normalize_spelling_variants(self, text: str) -> str:
        """Convert British spellings to American spellings."""
        for british, american in self.spelling_variants.items():
            pattern = self.word_boundary_pattern(british)
            text = pattern.sub(american, text)
        return text

    def _apply_synonym_mapping(self, text: str) -> str:
        """Apply medical synonym mappings."""
        for synonym, canonical in self.medical_synonyms.items():
            pattern = self.word_boundary_pattern(synonym)
            text = pattern.sub(canonical, text)
        return text

    def _expand_acronyms(self, text: str) -> str:
        """Expand medical acronyms and abbreviations."""
        for acronym, expansion in self.medical_acronyms.items():
            # Handle both with and without periods
            for variant in [acronym, acronym.replace(".", "")]:
                pattern = self.word_boundary_pattern(variant)
                # Replace with expansion, keeping original for context
                text = pattern.sub(f"{variant} {expansion}", text)
        return text

    def _normalize_diseases(self, text: str) -> str:
        """Normalize disease and condition names."""
        for variant, canonical in self.disease_normalizations.items():
            pattern = self.word_boundary_pattern(variant)
            text = pattern.sub(canonical, text)
        return text

    def _normalize_procedures(self, text: str) -> str:
        """Normalize procedure and test names."""
        for variant, canonical in self.procedure_normalizations.items():
            pattern = self.word_boundary_pattern(variant)
            text = pattern.sub(canonical, text)
        return text

    def _expand_anatomical_terms(self, text: str) -> str:
        """Expand anatomical terminology with synonyms."""
        # Track expanded terms to avoid redundant expansions
        expanded_terms = set()

        for base_term, synonyms in self.anatomical_mappings.items():
            pattern = self.word_boundary_pattern(base_term)
            if pattern.search(text) and base_term not in expanded_terms:
                # Add unique synonyms that aren't already in the text
                unique_synonyms = []
                for syn in synonyms:
                    if (
                        syn != base_term
                        and syn not in text.lower()
                        and syn not in expanded_terms
                    ):
                        unique_synonyms.append(syn)
                        expanded_terms.add(syn)

                if unique_synonyms:
                    # Only add 1-2 most relevant synonyms to avoid explosion
                    selected_synonyms = unique_synonyms[:2]
                    synonym_text = " ".join(selected_synonyms)
                    text = pattern.sub(f"{base_term} {synonym_text}", text)
                    expanded_terms.add(base_term)
        return text

    def normalize_query(self, query: str) -> str:
        """
        Normalize a query/statement with lighter processing.

        For queries, we want less aggressive expansion to avoid
        over-expanding short queries.
        """
        return self.normalize_text(query, expand_anatomical=False)

    def normalize_document(self, document: str) -> str:
        """
        Normalize a document with full processing.

        For documents, we want full expansion to improve recall.
        """
        return self.normalize_text(document, expand_anatomical=True)


# Global normalizer instance
_normalizer = None


def get_normalizer() -> MedicalTextNormalizer:
    """Get a singleton instance of the medical text normalizer."""
    global _normalizer
    if _normalizer is None:
        _normalizer = MedicalTextNormalizer()
    return _normalizer


def normalize_medical_text(text: str, is_query: bool = False) -> str:
    """
    Convenience function for medical text normalization.

    Args:
        text: Text to normalize
        is_query: Whether this is a query (lighter processing) or document (full processing)

    Returns:
        Normalized text
    """
    normalizer = get_normalizer()
    if is_query:
        return normalizer.normalize_query(text)
    else:
        return normalizer.normalize_document(text)
