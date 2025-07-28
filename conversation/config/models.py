"""
This module contains usefull models and configurations for the conversation module.
"""

#--------------------------------------- Imports ---------------------------------------#

from transformers import SiglipVisionModel, SiglipImageProcessor


#--------------------------------------- Models ---------------------------------------#

# Load the user retriever model and processor
model_name = "hamedrahimi/ULIP-p16"
USER_RETRIEVER_MODEL = SiglipVisionModel.from_pretrained(model_name)
USER_RETRIEVER_PROCESSOR = SiglipImageProcessor.from_pretrained(model_name)