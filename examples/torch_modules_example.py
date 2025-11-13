"""Example of using PyTorch modules from formed.

This example demonstrates how to use embedders, encoders, and other
building blocks from formed.integrations.torch.modules.
"""

import torch

from formed.integrations.torch.modules import (
    BagOfEmbeddingsSequenceVectorizer,
    FeedForward,
    LSTMSequenceEncoder,
    TokenEmbedder,
)


def main():
    # Create a simple token embedder
    embedder = TokenEmbedder(vocab_size=1000, embedding_dim=128)

    # Create a sequence encoder
    encoder = LSTMSequenceEncoder(input_dim=128, hidden_dim=256, num_layers=2, bidirectional=True)

    # Create a vectorizer for pooling
    vectorizer = BagOfEmbeddingsSequenceVectorizer(pooling="mean")

    # Create a feed-forward classifier
    classifier = FeedForward(input_dim=512, hidden_dim=128, output_dim=3, num_layers=2, dropout=0.1)

    # Create dummy data
    batch_size, seq_len = 4, 10
    token_ids = torch.randint(0, 1000, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # Create a simple batch object
    class Batch:
        def __init__(self, ids, mask):
            self.ids = ids
            self.mask = mask

    batch = Batch(token_ids, mask)

    # Forward pass
    # 1. Embed tokens
    embedder_output = embedder(batch)
    print(f"Embeddings shape: {embedder_output.embeddings.shape}")  # (4, 10, 128)

    # 2. Encode sequence
    encoded = encoder(embedder_output.embeddings, mask=embedder_output.mask)
    print(f"Encoded shape: {encoded.shape}")  # (4, 10, 512) - bidirectional doubles the dimension

    # 3. Vectorize to fixed size
    pooled = vectorizer(encoded, mask=embedder_output.mask)
    print(f"Pooled shape: {pooled.shape}")  # (4, 512)

    # 4. Classify
    logits = classifier(pooled)
    print(f"Logits shape: {logits.shape}")  # (4, 3)

    print("\n✅ All modules work correctly!")


if __name__ == "__main__":
    main()
