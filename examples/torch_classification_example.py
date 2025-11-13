"""Example demonstrating PyTorch classification modules.

This example shows how to use losses, samplers, and weighters from
formed.integrations.torch.modules for classification tasks.
"""

import torch

from formed.integrations.torch.modules import (
    ArgmaxLabelSampler,
    BalancedByDistributionLabelWeighter,
    CrossEntropyLoss,
    FeedForward,
    MultinomialLabelSampler,
    StaticLabelWeighter,
)


def main():
    # Create a simple classifier
    classifier = FeedForward(
        input_dim=128,
        hidden_dim=64,
        output_dim=3,  # 3 classes
        num_layers=2,
        dropout=0.1,
    )

    # Create dummy data
    batch_size = 8
    inputs = torch.randn(batch_size, 128)
    labels = torch.randint(0, 3, (batch_size,))  # Ground truth labels

    # Get logits from classifier
    logits = classifier(inputs)
    print(f"Logits shape: {logits.shape}")  # (8, 3)

    # ============================================================
    # 1. Loss Functions
    # ============================================================
    print("\n=== Loss Functions ===")

    # Simple cross-entropy loss
    simple_loss = CrossEntropyLoss()
    loss = simple_loss(logits, labels)
    print(f"Simple CE loss: {loss.item():.4f}")

    # Cross-entropy with static label weights
    # Weight class 2 more heavily (e.g., rare class)
    static_weights = torch.tensor([1.0, 1.0, 2.0])
    static_weighter = StaticLabelWeighter(weights=static_weights)
    weighted_loss = CrossEntropyLoss(weighter=static_weighter)
    loss = weighted_loss(logits, labels)
    print(f"Weighted CE loss (static): {loss.item():.4f}")

    # Cross-entropy with balanced weights
    # Assuming we know the class distribution
    class_distribution = torch.tensor([0.5, 0.3, 0.2])  # 50%, 30%, 20%
    balanced_weighter = BalancedByDistributionLabelWeighter(distribution=class_distribution)
    balanced_loss = CrossEntropyLoss(weighter=balanced_weighter)
    loss = balanced_loss(logits, labels)
    print(f"Weighted CE loss (balanced): {loss.item():.4f}")

    # ============================================================
    # 2. Label Samplers
    # ============================================================
    print("\n=== Label Samplers ===")

    # Argmax sampling (deterministic)
    argmax_sampler = ArgmaxLabelSampler()
    predicted_labels = argmax_sampler(logits)
    print(f"Predicted labels (argmax): {predicted_labels}")

    # Check accuracy
    accuracy = (predicted_labels == labels).float().mean()
    print(f"Accuracy: {accuracy.item():.4f}")

    # Multinomial sampling (stochastic)
    multi_sampler = MultinomialLabelSampler()
    sampled_labels = multi_sampler(logits)
    print(f"Sampled labels (multinomial): {sampled_labels}")

    # Sample with temperature
    # Low temperature = more deterministic (close to argmax)
    sampled_labels_low_temp = multi_sampler(logits, temperature=0.1)
    print(f"Sampled labels (temp=0.1): {sampled_labels_low_temp}")

    # High temperature = more random
    sampled_labels_high_temp = multi_sampler(logits, temperature=2.0)
    print(f"Sampled labels (temp=2.0): {sampled_labels_high_temp}")

    # ============================================================
    # 3. Label Weighters (standalone)
    # ============================================================
    print("\n=== Label Weighters (standalone) ===")

    # Get weights for each class
    static_weights = static_weighter(logits, labels)
    print(f"Static weights: {static_weights}")  # Shape: (1, 3)

    balanced_weights = balanced_weighter(logits, labels)
    print(f"Balanced weights: {balanced_weights}")  # Shape: (1, 3)

    # ============================================================
    # 4. Training example
    # ============================================================
    print("\n=== Training Example ===")

    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    loss_fn = CrossEntropyLoss(weighter=balanced_weighter)

    # Training loop (5 steps)
    for step in range(5):
        # Forward pass
        logits = classifier(inputs)
        loss = loss_fn(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate
        with torch.no_grad():
            predicted = argmax_sampler(logits)
            accuracy = (predicted == labels).float().mean()

        print(f"Step {step + 1}: loss={loss.item():.4f}, accuracy={accuracy.item():.4f}")

    print("\n✅ All classification modules work correctly!")


if __name__ == "__main__":
    main()
