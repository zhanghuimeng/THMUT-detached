#!/usr/bin/env python
# coding=utf-8
# Copyright https://stackoverflow.com/questions/50330455/how-to-detokenize-spacy-text-without-doc-context/59618856
# Detokenize tokens (does not cope with sub tokens)

import spacy
import string


class Detokenizer:
    """ This class is an attempt to detokenize spaCy tokenized sentence """
    def __init__(self, lang="en"):
        # Download the model by spacy if does not work
        if lang == "en":
            self.nlp = spacy.load("en_core_web_sm")
        elif lang == "de":
            self.nlp = spacy.load("de_core_news_sm")
        else:
            raise ValueError("Unknown Language Name '%s'" % lang)

    def get_words(self, tokens : list):
        """ Call this method to get list of detokenized words """
        while self._connect_next_token_pair(tokens):
            pass
        return tokens

    def get_sentence(self, tokens : list) -> str:
        """ call this method to get detokenized sentence """
        return " ".join(self.get_words(tokens))

    def _connect_next_token_pair(self, tokens : list):
        i = self._find_first_pair(tokens)
        if i == -1:
            return False
        tokens[i] = tokens[i] + tokens[i+1]
        tokens.pop(i+1)
        return True

    def _find_first_pair(self,tokens):
        if len(tokens) <= 1:
            return -1
        for i in range(len(tokens)-1):
            if self._would_spaCy_join(tokens,i):
                return i
        return -1

    def _would_spaCy_join(self, tokens, index):
        """
        Check whether the sum of lengths of spaCy tokenized words is equal to the length of joined and then spaCy tokenized words...

        In other words, we say we should join only if the join is reversible.
        eg.:
            for the text ["The","man","."]
            we would joins "man" with "."
            but wouldn't join "The" with "man."
        """
        left_part = tokens[index]
        right_part = tokens[index+1]
        length_before_join = len(self.nlp(left_part)) + len(self.nlp(right_part))
        length_after_join = len(self.nlp(left_part + right_part))
        if self.nlp(left_part)[-1].text in string.punctuation:
            return False
        return length_before_join == length_after_join


if __name__ == "__main__":
    # Test

    for lang in ["en", "de"]:
        dt = Detokenizer(lang)
        if lang == "en":
            sentence = "I am the man, who dont dont know. And who won't. be doing"
            nlp = spacy.load("en_core_web_sm")
        else:
            sentence = "Ich bin der Mann, der kennt nicht. Und der tut nicht."
            nlp = spacy.load("de_core_news_sm")

        spaCy_tokenized = nlp(sentence)

        string_tokens = [a.text for a in spaCy_tokenized]

        list_of_words = dt.get_words(string_tokens)
        detokenized_sentence = dt.get_sentence(string_tokens)

        print(lang)
        print("Original Sentence: " + sentence)
        print("Tokens: "+ str(string_tokens))
        print("Detokenized Words: " + str(list_of_words))
        print("Detokenized Sentence: " + detokenized_sentence)
