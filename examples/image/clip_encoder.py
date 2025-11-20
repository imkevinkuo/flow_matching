"""
CLIP encoder utility for text caption encoding.
"""

import torch
import clip


class CLIPEncoder:
    """CLIP text encoder for caption conditioning"""

    def __init__(self, device, model_name="ViT-B/32"):
        """
        Initialize CLIP encoder.

        Args:
            device: Device to load model on
            model_name: CLIP model variant (default: ViT-B/32)
        """
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load(model_name, device=device)
        self.clip_model.eval()

        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        print(f"Loaded CLIP model: {model_name}")

    @torch.no_grad()
    def encode_captions(self, captions):
        """
        Encode text captions using CLIP text encoder.

        Args:
            captions: List of text captions (batch_size,)

        Returns:
            text_features: Normalized CLIP embeddings (batch_size, 512)

        Raises:
            ValueError: If any caption is empty
        """
        if not captions:
            return None

        # Validate that all captions are non-empty
        for i, caption in enumerate(captions):
            if not caption:
                raise ValueError(
                    f"Empty caption found at index {i}. "
                    f"All captions must be non-empty when --captions is enabled."
                )

        # Tokenize and encode
        text_tokens = clip.tokenize(captions, truncate=True).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)

        # L2 normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features  # Shape: (batch_size, 512)
