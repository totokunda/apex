"""Example usage of the memory management system."""

import torch
import torch.nn as nn
import time
from typing import Optional

from .memory_manager import MemoryManager, auto_manage_model
from .config import MemoryConfig


def example_basic_usage():
    """Basic example of wrapping a model with memory management."""
    print("=== Basic Memory Management Example ===")

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    nn.Linear(1024, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 10),
                ]
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = SimpleModel()

    # Use auto_manage_model for simplest setup
    manager = auto_manage_model(model)

    # Run some forward passes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for i in range(5):
        x = torch.randn(32, 1024, device=device)
        output = model(x)
        print(f"Forward pass {i+1}, output shape: {output.shape}")

        # Show memory summary
        summary = manager.get_memory_summary()
        print(f"GPU usage: {summary['gpu']['usage_ratio']:.2%}")
        print(f"Managed modules: {summary['system']['managed_modules']}")

        time.sleep(1)

    # Cleanup
    manager.cleanup()
    print("Basic example completed\n")


def example_custom_config():
    """Example using custom memory configuration."""
    print("=== Custom Configuration Example ===")

    # Create low-memory configuration
    config = MemoryConfig.for_low_memory()
    print(f"GPU offload threshold: {config.gpu_offload_threshold:.1%}")
    print(f"CPU offload threshold: {config.cpu_offload_threshold:.1%}")

    # Create model
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
    )

    # Create manager with custom config
    with MemoryManager(config) as manager:
        manager.start()

        # Wrap the model
        wrapped_model = manager.wrap_model(model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wrapped_model = wrapped_model.to(device)

        # Run inference
        x = torch.randn(16, 512, device=device)
        output = wrapped_model(x)

        # Get detailed memory info
        summary = manager.get_memory_summary()
        print(f"Module summary: {summary['modules']}")

    print("Custom configuration example completed\n")


def example_manual_control():
    """Example of manual memory management control."""
    print("=== Manual Control Example ===")

    # Create model with large layers
    model = nn.Sequential(
        nn.Linear(2048, 4096),
        nn.Linear(4096, 4096),
        nn.Linear(4096, 2048),
        nn.Linear(2048, 1000),
    )

    # Create manager with auto-management disabled initially
    config = MemoryConfig()
    manager = MemoryManager(config)
    manager.set_auto_management(False)
    manager.start()

    # Wrap individual modules for fine control
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            wrapped = manager.wrap_module(module, module_id=name)
            linear_layers.append(wrapped)

    print(f"Wrapped {len(linear_layers)} linear layers")

    # Manually offload some layers
    print("Manually offloading first two layers...")
    for layer in linear_layers[:2]:
        layer.offload_all()

    # Show memory state
    summary = manager.get_memory_summary()
    for module_id, info in summary["modules"]["modules"].items():
        print(f"Module {module_id}: {info['offload_ratio']:.1%} offloaded")

    # Re-enable auto-management
    print("Enabling auto-management...")
    manager.set_auto_management(True)

    # Run forward passes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for i in range(3):
        x = torch.randn(8, 2048, device=device)
        output = model(x)
        print(f"Forward pass {i+1} completed")
        time.sleep(2)  # Allow auto-management to work

    manager.cleanup()
    print("Manual control example completed\n")


def example_transformer_like_model():
    """Example with a transformer-like model."""
    print("=== Transformer-like Model Example ===")

    class TransformerBlock(nn.Module):
        def __init__(self, d_model=512, nhead=8, dim_feedforward=2048):
            super().__init__()
            self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

            # Feed-forward network
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, d_model),
            )

        def forward(self, x):
            # Self-attention
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)

            # Feed-forward
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)

            return x

    class TransformerModel(nn.Module):
        def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.blocks = nn.ModuleList(
                [TransformerBlock(d_model, nhead) for _ in range(num_layers)]
            )
            self.output_proj = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            x = self.embedding(x)
            for block in self.blocks:
                x = block(x)
            return self.output_proj(x)

    # Create model
    model = TransformerModel(num_layers=4)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Use high-performance config
    config = MemoryConfig.for_high_performance()

    # Auto-manage the model, targeting Linear layers
    manager = auto_manage_model(model, config, layer_types=[nn.Linear])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Simulate training/inference
    seq_length = 128
    batch_size = 16

    for step in range(3):
        # Create input tokens
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=device)

        # Forward pass
        logits = model(input_ids)
        print(f"Step {step+1}: output shape {logits.shape}")

        # Show memory stats
        summary = manager.get_memory_summary()
        print(f"GPU usage: {summary['gpu']['usage_ratio']:.2%}")
        print(f"Offloaded modules: {summary['modules']['offloaded_parameters']}")
        print(f"Total offloads: {summary['system']['total_offloads']}")
        print()

        time.sleep(1)

    manager.cleanup()
    print("Transformer example completed\n")


def example_sentiment_analysis_model():
    """Example optimized for sentiment analysis with <100ms latency requirement."""
    print("=== Sentiment Analysis Model Example ===")

    class SentimentClassifier(nn.Module):
        def __init__(self, vocab_size=10000, embedding_dim=256, hidden_dim=512):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 3),  # positive, negative, neutral
            )

        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, (hidden, _) = self.lstm(embedded)
            # Use last hidden state
            return self.classifier(hidden[-1])

    model = SentimentClassifier()

    # Configure for low latency - keep more in GPU memory
    config = MemoryConfig(
        gpu_offload_threshold=0.95,  # Only offload when very full
        gpu_emergency_threshold=0.98,
        memory_check_interval=0.05,  # Check every 50ms
        prefetch_enabled=True,
        offload_batch_size=5,  # Small batches for faster response
    )

    manager = auto_manage_model(model, config, layer_types=[nn.Linear])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Simulate real-time sentiment analysis
    print("Simulating real-time sentiment analysis...")

    for i in range(10):
        start_time = time.time()

        # Simulate input sentence (variable length)
        seq_len = torch.randint(10, 50, (1,)).item()
        input_ids = torch.randint(0, 1000, (1, seq_len), device=device)

        # Forward pass
        logits = model(input_ids)
        predicted_class = torch.argmax(logits, dim=1).item()

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        sentiment = ["positive", "negative", "neutral"][predicted_class]
        print(f"Sample {i+1}: {sentiment} (latency: {latency_ms:.1f}ms)")

        # Check if we're meeting latency requirement
        if latency_ms > 100:
            print("⚠️  Latency exceeded 100ms requirement!")

        time.sleep(0.1)  # Small delay between requests

    # Final performance summary
    summary = manager.get_memory_summary()
    print(f"\nFinal GPU usage: {summary['gpu']['usage_ratio']:.2%}")
    print(
        f"Total operations: {summary['system']['total_offloads'] + summary['system']['total_reloads']}"
    )

    manager.cleanup()
    print("Sentiment analysis example completed\n")


def run_all_examples():
    """Run all examples."""
    print("Running Memory Management Examples")
    print("=" * 50)

    try:
        example_basic_usage()
        example_custom_config()
        example_manual_control()
        example_transformer_like_model()
        example_sentiment_analysis_model()

        print("All examples completed successfully!")

    except Exception as e:
        print(f"Example failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
