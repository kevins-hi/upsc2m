from pydantic import BaseModel
import spacy
import textstat

nlp = spacy.load("en_core_web_sm")

##########################################################################################
###    The following pydantic models are used with OpenAI's structured outputs API.    ###
##########################################################################################

# Obscurity
class ObscurityScore(BaseModel):
    reasoning: str
    facts: list[str]
    obscurities: list[int]

# Ambiguity
class AmbiguityScore(BaseModel):
    reasoning: str
    score: int

# Distractor Quality
class DistractorQuality(BaseModel):
    reasoning: str
    distractor_1_quality: int
    distractor_2_quality: int
    distractor_3_quality: int

###############################################################################
###    The following functions are used to compute the non-LLM features.    ###
###############################################################################

# Reading Difficulty
def compute_reading_difficulty(question: str) -> float:
    return textstat.flesch_reading_ease(question)

# Negation Presence
def infer_negation_presence(question: str) -> int:
    doc = nlp(question.question)
    negation_presence = any(token.dep_ == "neg" for token in doc)
    return int(negation_presence)

# Named Entity Count
useful_labels = {
    "PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "EVENT", 
    "LAW", "PRODUCT", "NORP", "LANGUAGE", "FAC", "DATE",
}

def count_named_entities(question: str) -> int:
    doc = nlp(question)
    seen_texts = set()
    filtered_ents = []
    for ent in doc.ents:
        ent_text = ent.text.strip().lower()
        if ent.label_ in useful_labels and ent_text not in seen_texts:
            seen_texts.add(ent_text)
            filtered_ents.append((ent.text, ent.label_))
    return len(filtered_ents)