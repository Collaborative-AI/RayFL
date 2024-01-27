# Use huggingface/accelerate-gpu as the base image
FROM huggingface/accelerate-cpu
# FROM huggingface/accelerate-gpu

# Install the required Python packages
USER root

COPY src /src
WORKDIR /src

RUN pip install torchvision dataset evaluate seqeval accelerate torchinfo peft rouge_score tensorboard diffusers[torch] transformers