"""
Define the data models for the memory system using Pydantic.
"""

#--------------------------------------- Imports ---------------------------------------#

from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field

import yaml

#--------------------------------------- Config ---------------------------------------#

#load config from YAML file
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# define here the vocabulary for contextual features
vocabulary = config.get("memory", {}).get("vocabulary", [])

# Define the patterns for name field (contextual features)
free_pattern = r"^\w+$"
closed_vocabulary_pattern = r"^(" + "|".join(vocabulary) + r")$"

pattern = closed_vocabulary_pattern if config.get("memory", {}).get("closed_vocabulary", True) else free_pattern

###############

# Define the type aliases for tags and values
TagsListType = List[str]
ValueListType = List[str]

#--------------------------------------- Data Models ---------------------------------------#

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
        pattern=pattern,
        description="Doit être un mot unique sans espaces ni caractères spéciaux, décrivant la catégorie d'une caractéristique utilisateur (par exemple: hobby, emploi, intérêts...)."
    )
    description: str = Field(
        ..., 
        description="Description générale de la caractéristique, indépendante de sa valeur, pour donner plus de détails sur ce qu'elle représente."
    )
    tags: TagsListType = Field(
        ..., 
        description="Liste de mots-clés associés à la caractéristique, pour faciliter la recherche et le filtrage. Doit contenir entre 1 et 3 mots-clés.",
        min_items=1
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

# Define a type alias for Feature, which can be either PrimaryFeature or ContextualFeature
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

class Name(BaseModel):
    name: str = Field(
        ..., 
        pattern=pattern,
        description="Doit être un mot unique sans espaces ni caractères spéciaux, décrivant la catégorie d'une caractéristique utilisateur (par exemple: hobby, emploi, intérêts...)."
    )

class FeaturesNames(BaseModel):
    Modify: List[str] = Field(
        ...,
        description="Liste des noms de caractéristiques à modifier."
    )
    Add: List[Name] = Field(
        ...,
        description="Liste des noms de caractéristiques à ajouter."
    )
