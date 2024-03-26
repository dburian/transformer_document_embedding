"""Standardized column names.

Datasets should honor it (most of them do so by default). Models/Heads can count on it.
"""


"""String of document's text."""
TEXT = "text"

"""Integer identifier of a document."""
ID = "id"

"""String of first document's text."""
TEXT_0 = "text_0"

"""String of second document's text."""
TEXT_1 = "text_1"

"""Integer identifier of first document."""
ID_0 = "id_0"

"""Integer identifier of first document."""
ID_1 = "id_1"

"""Embedding of contextual model."""
CONTEXTUAL_EMBED = "contextual_embed"

"""Embedding of structural model."""
STRUCTURAL_EMBED = "structural_embed"

"""Supervised label for document."""
LABEL = "label"

"""Embedding produced by an embedding model."""
EMBEDDING = "embedding"

"""Length of document in mpnet tokens."""
LENGTH = "length"

IDS = {
    ID_0,
    ID_1,
    ID,
}

SUPERVISED = IDS | {
    LABEL,
    CONTEXTUAL_EMBED,
    STRUCTURAL_EMBED,
}
