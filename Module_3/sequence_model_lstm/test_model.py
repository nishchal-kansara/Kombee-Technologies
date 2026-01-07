import torch
from model import LSTMClassifier

def test_output_shape():
    model = LSTMClassifier(
        vocab_size=100,
        embed_dim=16,
        hidden_dim=32,
        num_classes=3
    )

    x = torch.randint(0, 100, (4, 8))
    y = model(x)

    assert y.shape == (4, 3)
    print("Output shape test passed")

def test_forward_pass():
    model = LSTMClassifier(
        vocab_size=20,
        embed_dim=8,
        hidden_dim=16,
        num_classes=2
    )

    x = torch.randint(0, 20, (2, 5))
    y = model(x)

    assert not torch.isnan(y).any()
    print("Forward pass test passed")

def test_batch_consistency():
    model = LSTMClassifier(
        vocab_size=30,
        embed_dim=10,
        hidden_dim=20,
        num_classes=4
    )

    x1 = torch.randint(0, 30, (1, 6))
    x2 = torch.randint(0, 30, (3, 6))

    y1 = model(x1)
    y2 = model(x2)

    assert y1.shape[0] == 1
    assert y2.shape[0] == 3
    print("Batch handling test passed")

if __name__ == "__main__":
    test_output_shape()
    test_forward_pass()
    test_batch_consistency()
