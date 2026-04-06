import torch
from nanovllm.utils.quant import TurboQuantMSEKVQuantizer, TurboQuantProdKVQuantizer

def test_quantizer(name, quantizer_class, bits=4, num_vectors=100, dim=128):
    print(f"========== Testing {name} ({bits}-bit) ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the KV quantizer
    quantizer = quantizer_class(bits=bits)
    
    # Generate 100 random vectors
    # Using float16/bfloat16 as it's common for LLM activations
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    x = torch.randn(num_vectors, dim, device=device, dtype=dtype)
    
    # 1. Quantize
    q, scales = quantizer.quantize(x)
    
    # 2. Dequantize
    x_hat = quantizer.dequantize(q, scales, dtype=dtype)
    
    # Compute Quantization Error
    mse = torch.nn.functional.mse_loss(x.float(), x_hat.float()).item()
    mae = torch.nn.functional.l1_loss(x.float(), x_hat.float()).item()
    print(f"Quantization Error over {num_vectors} vectors:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    # Print comparison for the first 3 vectors
    print(f"\nComparing first 3 vectors (Showing first 5 dimensions of each):")
    for i in range(3):
        print(f"Vector {i}:")
        print(f"  Original:   {x[i, :5].tolist()}")
        print(f"  Quantized:  {x_hat[i, :5].tolist()}")
    print("\n")


if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducibility
    
    test_quantizer(
        "TurboQuant MSE", 
        TurboQuantMSEKVQuantizer, 
        bits=4, 
        num_vectors=100, 
        dim=128
    )
    
    test_quantizer(
        "TurboQuant Prod", 
        TurboQuantProdKVQuantizer, 
        bits=4, 
        num_vectors=100, 
        dim=128
    )
