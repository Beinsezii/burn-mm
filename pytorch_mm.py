import time

import torch

for dtype in [torch.float32, torch.bfloat16, torch.float16]:
    for size in [256, 512, 1024, 2048, 4096]:
        flops = size * size * size * 2

        a = torch.rand((size, size), dtype=dtype, device="cuda")
        b = torch.rand((size, size), dtype=dtype, device="cuda")

        def timed_mm(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
            clock = time.perf_counter()
            _ = lhs.matmul(rhs)
            torch.cuda.synchronize()
            return time.perf_counter() - clock

        duration = min(timed_mm(a, b) for _ in range(20))
        print(f"{flops * 1e-12 / duration:6.2f}\t{size:4}\t{dtype}")
