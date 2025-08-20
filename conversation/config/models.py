"""
This module contains usefull models and configurations for the conversation module.
"""

#--------------------------------------- Imports ---------------------------------------#

from transformers import SiglipVisionModel, SiglipImageProcessor
import dotenv
import os
from insightface.model_zoo.model_zoo import get_model as insightface_get_model

from conversation.config.utils import suppress_stdout

################

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

#--------------------------------------- Models ---------------------------------------#

# ULIP-p16 
model_name = "hamedrahimi/ULIP-p16"
USER_RETRIEVER_MODEL = SiglipVisionModel.from_pretrained(model_name)
USER_RETRIEVER_PROCESSOR = SiglipImageProcessor.from_pretrained(model_name)

# INSIGHTFACE
with suppress_stdout():
    INSIGHTFACE_MODEL = insightface_get_model('buffalo_l', download=True) #FaceAnalysis(name='buffalo_l')
    INSIGHTFACE_MODEL.prepare(ctx_id=0)  # or -1 for CPU