# Analysis

## Layer 2, Head 5: Subject-Verb Relationships

This attention head appears to focus on the relationship between subjects and their verbs. Specifically, the tokens representing subjects (such as nouns or pronouns) seem to pay attention to the tokens representing their corresponding verbs.

Example Sentences:

Input: "The dog chased the cat."
Observed: The token "dog" attends strongly to "chased."

Input: "She enjoys reading books in the evening."
Observed: The token "She" attends strongly to "enjoys."

In these examples, the attention mechanism identifies the grammatical relationship between the subject of the sentence and the verb that follows it, demonstrating a basic syntactic understanding.

## Layer 5, Head 8: Modifier-Noun Relationships

This attention head appears to focus on adjectives modifying nouns. Specifically, adjectives strongly attend to the nouns they describe, and vice versa.

Example Sentences:

Input: "The tall building dominates the skyline."
Observed: The token "tall" attends strongly to "building."

Input: "A beautiful painting was displayed in the gallery."
Observed: The token "beautiful" attends strongly to "painting."

This head identifies the semantic relationship between descriptive modifiers (adjectives) and the nouns they modify, contributing to a model's ability to interpret descriptive language.

Relationships identified are not absolute and may not hold for all sentences. However, these attention patterns provide insights into the linguistic capabilities of the BERT model.

Attention heads are probabilistic and may attend to other tokens outside the primary relationship described.
