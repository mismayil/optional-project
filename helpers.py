import spacy

nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser"])

doc_cache = {}

def get_doc(text):
    if text in doc_cache:
        return doc_cache[text]
    
    doc = nlp(text)
    doc_cache[text] = doc

    return doc

with open("placeholders.txt") as f:
    placeholders = f.read().splitlines()

def has_overlap(specific_text: str, general_text: str, threshold: float) -> bool:
    sp_doc = get_doc(specific_text)
    gl_doc = get_doc(general_text)

    sp_words = set([token.lemma_.lower() for token in sp_doc])
    gl_words = set([token.lemma_.lower() for token in gl_doc if token.text not in placeholders])

    if len(gl_words) > 0:
        return (len(sp_words.intersection(gl_words)) / len(gl_words)) > threshold
    
    return False


def has_overlap_with_story(text: str, story: str, threshold: float) -> bool:
    contexts = story.split(".")
    doc = get_doc(text)
    words = set([token.lemma_.lower() for token in doc])

    for context in contexts:
        context_doc = get_doc(context)
        context_words = set([token.lemma_.lower() for token in context_doc])

        if len(words) > 0 and (len(words.intersection(context_words)) / len(words)) > threshold:
            return True

    return False  