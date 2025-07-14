import time

import torch

print("backend\ttflops\tshape\tdtype")

devices: list[torch.device] = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda"))
    # torch.cuda.tunable.enable(True)

for device in devices:
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        for size in [256, 512, 1024, 2048, 4096]:
            if device.type == "cpu" and size > 1024:
                continue

            flops = size * size * size * 2

            a = torch.rand((size, size), dtype=dtype, device=device)
            b = torch.rand((size, size), dtype=dtype, device=device)

            def timed_mm(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
                clock = time.perf_counter()
                _ = lhs.matmul(rhs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                return time.perf_counter() - clock

            duration = min(timed_mm(a, b) for _ in range(20))
            print(f"pytorch-{device.type}\t{flops * 1e-12 / duration:.2f}\t{size}\t{dtype}")
