from dataclasses import dataclass, field
from typing import (
    Dict,
    Optional,
    List
)

@dataclass
class Model:
    # Internal and unique name of the model
    name: str
    # Pretty name to show
    pretty_name: str
    # Provider of the model, e.g., "OpenAI", "Anthropic", "DeepSeek"
    provider: str
    # Parameters of the model, e.g., {"temperature": 0.7, "max_tokens": 1000}
    parameters: dict = field(default_factory=dict)


    def __post_init__(self):
        if not self.name:
            raise ValueError("Model name cannot be empty.")
        if not self.pretty_name:
            raise ValueError("Model pretty name cannot be empty.")
        if not self.provider:
            raise ValueError("Model provider cannot be empty.")
        if not isinstance(self.parameters, dict):
            raise TypeError("Model parameters must be a dictionary.")
    def __str__(self):
        return f"{self.pretty_name} ({self.provider})"
    def __repr__(self):
        return f"Model(name={self.name}, pretty_name={self.pretty_name}, provider={self.provider}, parameters={self.parameters})"


# List of available models. gpt-4.1, o3, deepseek-reasoner, claude-4-sonnet
MODELS = [
    Model(name="gpt-5-2025-08-07", pretty_name="GPT-5", provider="OpenAI", parameters={"temperature": 0.7, "max_tokens": 1024}),
    Model(name="gpt-4.1", pretty_name="GPT-4.1", provider="OpenAI", parameters={"temperature": 0.7, "max_tokens": 1024}),
    Model(name="gpt-4.1-mini", pretty_name="GPT-4.1 Mini", provider="OpenAI", parameters={"temperature": 0.7, "max_tokens": 1000}),
    Model(name="o3", pretty_name="O3", provider="OpenAI", parameters={"temperature": 0.7, "max_tokens": 1000}),
    Model(name="deepseek-reasoner", pretty_name="DeepSeek Reasoner", provider="DeepSeek", parameters={"temperature": 0.7, "max_tokens": 1000}),
    Model(name="claude-4-sonnet", pretty_name="Claude 4 Sonnet", provider="Anthropic", parameters={"temperature": 0.7, "max_tokens": 1000})
]


def get_model_by_name(name: str) -> Model:
    """Get a model by its internal name."""
    for model in MODELS:
        if model.name == name:
            return model
    raise ValueError(f"Model with name '{name}' not found.")


def get_available_models() -> List[str]:
    """Get a list of available models."""
    return [model.name for model in MODELS]


def get_model_pretty_name(name: str) -> str:
    """Get the pretty name of a model by its internal name."""
    model = get_model_by_name(name)
    return model.pretty_name if model else None


DEFAULT_MODEL = get_model_by_name("gpt-4.1")
