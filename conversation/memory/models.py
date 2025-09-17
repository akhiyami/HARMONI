"""
Define the data models for the memory system using Pydantic.
"""

#--------------------------------------- Imports ---------------------------------------#

from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field, model_validator

import yaml

#--------------------------------------- Config ---------------------------------------#

#load config from YAML file
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# define here the vocabulary for contextual features
vocabulary = config.get("memory", {}).get("vocabulary", {})

# Define the patterns for name field (contextual features)
free_pattern = r"^\w+$"
closed_vocabulary_pattern = r"^(" + "|".join(list(vocabulary.keys())) + r")$"

pattern = closed_vocabulary_pattern if config.get("memory", {}).get("closed_vocabulary", True) else free_pattern

primary_features = {
    "nom": "Les prénoms et noms de l'utilisateur.",
    "age": "L'âge de l'utilisateur.",
    "genre": "Le genre de l'utilisateur (masculin, féminin, non-binaire, etc.).",
    "preference_dialogue": "Les préférences de dialogue de l'utilisateur : le ton, comment il souhaite se faire appeler..."
}

name_pattern = free_pattern if not config.get("memory", {}).get("closed_vocabulary", True) else r"^(" + "|".join(list(vocabulary.keys()) + list(primary_features.keys())) + r")$"

###############

# Define the type aliases for tags and values
TagsListType = List[str]
ValueListType = List[str]
EmbeddingType = List[float]  # embedding for a single value
EmbeddingListType = List[EmbeddingType]  # embeddings per value

#--------------------------------------- Data Models ---------------------------------------#

class PrimaryFeature(BaseModel):
    type: Literal["primary"] = Field(
        ...,
        description="Indique que cette caractéristique est primaire, comme le nom, l'âge, le genre ou les préférences de dialogue."
    )
    name: Literal["nom", "age", "genre", "preference_dialogue"] = Field(
        ...,
        description="Décrit la catégorie d'une caractéristique utilisateur (nom, âge, genre, préférence de dialogue)."
    )
    description: str = Field(
        ...,
        description="Description textuelle de la caractéristique, définie automatiquement selon le nom."
    )
    value: Optional[ValueListType] = Field(
        None, 
        description="Liste de valeurs associées à la caractéristique.",
        min_items=0
    )
    embeddings: None = Field(
        None,
        description="Les caractéristiques primaires n'ont pas d'embeddings."
    )

    @model_validator(mode="after")
    def set_description_from_name(self):
        expected_desc = primary_features[self.name]
        object.__setattr__(self, "description", expected_desc)
        return self


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
    description: Optional[str] = Field(
        None,
        description="Description textuelle de la caractéristique, expliquant son contexte ou sa signification. Peut être None.",
        min_length=10,
        max_length=200
    )
    value: ValueListType = Field(
        ..., 
        description="Liste de valeurs associées à la caractéristique, pour représenter les différentes facettes ou aspects de cette caractéristique.",
        min_items=1
    )
    embeddings: Optional[EmbeddingListType] = Field(
        None,
        description="Liste d'embeddings correspondants aux valeurs contextuelles."
    )
    @model_validator(mode="after")
    def set_description_from_name(self):
        if self.name not in vocabulary:
            raise ValueError(f"Unknown name '{self.name}'. Must be one of {list(vocabulary)}")
        expected_desc = vocabulary[self.name]
        object.__setattr__(self, "description", expected_desc)  # enforce binding
        return self

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
        pattern=name_pattern,
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
