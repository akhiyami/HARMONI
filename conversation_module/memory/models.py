"""
Define the data models for the memory system using Pydantic.
"""
############
# Import necessary libraries
############

from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field

# define here the vocabulary for contextual features 
VOCABULARY = ["emploi", "intérêts", "loisirs", "sports", "musique", "voyages", "technologie", "famille", "amis", "animaux", "nourriture"] 

# Define the patterns for name field (contextual features)
free_pattern = r"^\w+$"
closed_vocaluary_pattern = r"^(" + "|".join(VOCABULARY) + r")$"

###########
# Type Definitions
###########

TagsListType = List[str]
ValueListType = List[str]

# Define Pydantic models for the feature structures
class PrimaryFeature(BaseModel):
    type: Literal["primary"] = Field(
        ...,
        description="Indique que cette caractéristique est primaire, comme le nom, l'âge, le genre ou les préférences de dialogue."
    )
    name: Literal["nom", "age", "genre", "preference_dialogue"] = Field(
        ...,
        description="Décris la catégorie d'une caractéristique utilisateur (nom, âge, genre, préférence de dialogue)."
    )
    description: None = Field(
        None, 
        description="Cette caractéristique est primaire et n'a pas besoin de description."
    )
    tags: None = Field(
        None, 
        description="Cette caractéristique est primaire et n'a pas besoin de tags."
    )
    value: Optional[ValueListType] = Field(
        None, 
        description="Liste de valeurs associées à la caractéristique, pour représenter les différentes facettes ou aspects de cette caractéristique. Peut être None.",
        min_items=0
    )
    embeddings: None = Field(
        None, 
        description="Cette caractéristique est primaire et n'a pas besoin d'embeddings."
    )

class ContextualFeature(BaseModel):
    type: Literal["contextual"] = Field(
        ...,
        description="Indique que cette caractéristique est contextuelle, c'est-à-dire qu'elle n'est pertinente que dans un contexte spécifique."
    )
    name: str = Field(
        ..., 
        pattern=free_pattern, # or closed_vocaluary_pattern,
        description="Doit être un mot unique sans espaces ni caractères spéciaux, décrivant la catégorie d'une caractéristique utilisateur (par exemple: hobby, emploi, intérêts...)."
    )
    description: str = Field(
        ..., 
        description="Description de la caractéristique, pour donner plus de contexte et de détails sur ce qu'elle représente."
    )
    tags: TagsListType = Field(
        ..., 
        description="Liste de mots-clés associés à la caractéristique, pour faciliter la recherche et le filtrage. Doit contenir entre 1 et 3 mots-clés.",
        min_items=1,
        max_items=3
    )
    value: ValueListType = Field(
        ..., 
        description="Liste de valeurs associées à la caractéristique, pour représenter les différentes facettes ou aspects de cette caractéristique.",
        min_items=1
    )
    embeddings: Optional[List[float]] = Field(
        None, 
        description="Représentation vectorielle de la caractéristique, utilisée pour la recherche sémantique et la similarité. Elle sera générée automatiquement plus tard."
    )

Feature = Union[PrimaryFeature, ContextualFeature]


# Define a Pydantic model for the answer with updated memory
class LongTermMemory(BaseModel):
    primary_features: List[Feature] = Field(
        default_factory=list,
        description="Caractéristiques essentielles (nom, genre, âge, ton, etc.)."
    )
    features: List[Feature] = Field(
        default_factory=list,
        description="Caractéristiques secondaires ou contextuelles (goûts, opinions, expériences...)."
    )
