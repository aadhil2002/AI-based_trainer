from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textstat import flesch_kincaid_grade, syllable_count
import logging
from typing import Dict

class PhrasingLanguage:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def analyze(self, text: str) -> Dict[str, float]:
        try:
            words = word_tokenize(text)
            unique_words = set(words) - self.stop_words
            lexical_diversity = len(unique_words) / max(len(words), 1)

            readability = flesch_kincaid_grade(text)
            avg_syllables = sum(syllable_count(word) for word in words) / max(len(words), 1)

            return {
                "language_score": float((lexical_diversity + (1 / (readability + 1))) / 2),
                "lexical_diversity": float(lexical_diversity),
                "readability_grade": float(readability),
                "avg_syllables_per_word": float(avg_syllables)
            }
        except Exception as e:
            logging.error(f"Error in PhrasingLanguage analysis: {str(e)}")
            return {"language_score": 0.5, "lexical_diversity": 0, "readability_grade": 0, "avg_syllables_per_word": 0}