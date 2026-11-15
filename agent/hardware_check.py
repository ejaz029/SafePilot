import torch
import bitsandbytes as bnb


def check_hardware():
    print("SafePilot hardware check")
    print("-" * 40)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("RESULT: CUDA GPU unavailable.")
        return False

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    properties = torch.cuda.get_device_properties(0)
    total_memory = properties.total_memory / (1024 ** 3)
    print(f"GPU memory: {total_memory:.2f} GB")

    try:
        layer = bnb.nn.Linear4bit(
            input_features=128,
            output_features=64,
            bias=False,
            compute_dtype=torch.float16,
            quant_type="nf4",
            compress_statistics=True,
        ).cuda()

        inputs = torch.randn(
            2, 128,
            device="cuda",
            dtype=torch.float16,
        )

        with torch.inference_mode():
            output = layer(inputs)

        torch.cuda.synchronize()

        print(f"Output shape: {tuple(output.shape)}")
        print("4-bit CUDA operation: SUCCESS")

        del layer, inputs, output
        torch.cuda.empty_cache()

        return True

    except Exception as exc:
        print("4-bit CUDA operation: FAILED")
        print(f"{type(exc).__name__}: {exc}")
        return False


if __name__ == "__main__":
    check_hardware()
