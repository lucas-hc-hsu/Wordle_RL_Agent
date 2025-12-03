"""
Benchmark script to test GPU VRAM usage with different numbers of parallel environments.
Finds optimal number of parallel games while keeping VRAM under a specified limit.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import gc
import time
import argparse
from model.ppo import ActorCritic
from utils.word_list import load_word_list


def get_vram_usage():
    """Get current VRAM usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        return allocated, reserved
    return 0, 0


def clear_cuda_cache():
    """Clear CUDA cache and force garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class VectorizedWordleEnvBenchmark:
    """Simplified vectorized environment for benchmarking."""

    def __init__(self, word_list, device, num_envs=32):
        self.word_list = word_list
        self.device = device
        self.num_envs = num_envs
        self.state_size = 26 * 3

        # Allocate GPU memory for states
        self.states = torch.zeros(num_envs, self.state_size, device=device)
        self.dones = torch.zeros(num_envs, dtype=torch.bool, device=device)


def benchmark_num_envs(num_envs, word_list, device, mini_batch_size=256):
    """
    Benchmark memory usage for a specific number of environments.

    Returns:
        dict: Memory usage info and whether it fits within limits
    """
    clear_cuda_cache()

    state_size = 26 * 3
    action_size = len(word_list)
    steps_per_update = 128

    try:
        # Create environment
        env = VectorizedWordleEnvBenchmark(word_list, device, num_envs)

        # Create model
        model = ActorCritic(state_size, action_size, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        # Simulate rollout buffer allocation (worst case)
        # Buffer stores: states, actions, rewards, log_probs, values, dones
        # for steps_per_update steps
        buffer_states = torch.zeros(steps_per_update, num_envs, state_size, device=device)
        buffer_actions = torch.zeros(steps_per_update, num_envs, dtype=torch.long, device=device)
        buffer_rewards = torch.zeros(steps_per_update, num_envs, device=device)
        buffer_log_probs = torch.zeros(steps_per_update, num_envs, device=device)
        buffer_values = torch.zeros(steps_per_update, num_envs, device=device)
        buffer_dones = torch.zeros(steps_per_update, num_envs, dtype=torch.bool, device=device)

        # Simulate a forward pass
        states = env.states
        with torch.no_grad():
            action_logits, values = model(states)

        # Simulate backward pass with mini-batch
        batch_states = torch.zeros(mini_batch_size, state_size, device=device)
        action_logits, values = model(batch_states)
        loss = action_logits.mean() + values.mean()
        loss.backward()
        optimizer.step()

        # Synchronize to ensure all operations complete
        torch.cuda.synchronize()

        allocated, reserved = get_vram_usage()

        # Clean up
        del env, model, optimizer
        del buffer_states, buffer_actions, buffer_rewards
        del buffer_log_probs, buffer_values, buffer_dones
        del states, action_logits, values, loss, batch_states
        clear_cuda_cache()

        return {
            "num_envs": num_envs,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "success": True,
            "error": None
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            clear_cuda_cache()
            return {
                "num_envs": num_envs,
                "allocated_gb": 0,
                "reserved_gb": 0,
                "success": False,
                "error": str(e)
            }
        raise


def find_optimal_num_envs(word_list, device, max_vram_gb=15.0, start=32, max_envs=4096):
    """
    Find the optimal number of environments that fits within VRAM limit.
    Uses binary search for efficiency.
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking GPU VRAM Usage")
    print(f"{'='*60}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"Max VRAM limit: {max_vram_gb} GB")
    print(f"{'='*60}\n")

    results = []

    # Test specific values
    test_values = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    test_values = [v for v in test_values if v <= max_envs]

    optimal_envs = 0
    optimal_vram = 0

    for num_envs in test_values:
        print(f"Testing {num_envs} parallel environments...", end=" ")
        result = benchmark_num_envs(num_envs, word_list, device)
        results.append(result)

        if result["success"]:
            vram = result["reserved_gb"]
            print(f"VRAM: {vram:.2f} GB", end="")

            if vram <= max_vram_gb:
                print(f" ✓ (within limit)")
                if num_envs > optimal_envs:
                    optimal_envs = num_envs
                    optimal_vram = vram
            else:
                print(f" ✗ (exceeds limit)")
        else:
            print(f"FAILED - {result['error'][:50]}...")
            break

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Num Envs':<12} {'Allocated GB':<15} {'Reserved GB':<15} {'Status'}")
    print(f"{'-'*60}")
    for r in results:
        status = "✓ OK" if r["success"] and r["reserved_gb"] <= max_vram_gb else "✗ Over limit" if r["success"] else "FAILED"
        print(f"{r['num_envs']:<12} {r['allocated_gb']:<15.3f} {r['reserved_gb']:<15.3f} {status}")

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION")
    print(f"{'='*60}")
    print(f"Optimal number of parallel environments: {optimal_envs}")
    print(f"Expected VRAM usage: {optimal_vram:.2f} GB")
    print(f"{'='*60}")

    return optimal_envs, optimal_vram


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU VRAM usage for parallel Wordle training")
    parser.add_argument("--max-vram", type=float, default=15.0, help="Maximum VRAM in GB (default: 15.0)")
    parser.add_argument("--max-envs", type=int, default=4096, help="Maximum environments to test (default: 4096)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Cannot benchmark GPU.")
        return

    device = torch.device("cuda")

    # Load word list
    word_list = load_word_list()
    print(f"Loaded {len(word_list)} words")

    # Find optimal number of environments
    optimal_envs, optimal_vram = find_optimal_num_envs(
        word_list,
        device,
        max_vram_gb=args.max_vram,
        max_envs=args.max_envs
    )

    return optimal_envs, optimal_vram


if __name__ == "__main__":
    main()
