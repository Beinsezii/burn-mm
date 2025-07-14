use burn::prelude::*;
use burn::tensor;
use std::time::Instant;

macro_rules! bench_mm {
    ($label:expr, $backend:ty, $device:expr, $dtypes:expr) => {
        for dtype in $dtypes {
            for n in [256, 512, 1024, 2048, 4096] {
                let flops = n * n * n * 2;
                let lhs =
                    Tensor::<$backend, 2>::random([n, n], tensor::Distribution::Default, $device)
                        .cast(dtype);
                let rhs =
                    Tensor::<$backend, 2>::random([n, n], tensor::Distribution::Default, $device)
                        .cast(dtype);
                let tflops = flops as f64 * 1e-12
                    / (0..20)
                        .map(|_| {
                            let clock = Instant::now();
                            let _ = lhs.clone().matmul(rhs.clone()); // can't use refs?
                            <$backend>::sync($device);
                            Instant::now() - clock
                        })
                        .min()
                        .unwrap()
                        .as_secs_f64();
                println!("{}\t{tflops:.2}\t{n}\t{dtype:?}", $label);
            }
        }
    };
}

fn main() {
    println!("backend\ttflops\tshape\tdtype");

    #[cfg(feature = "cuda")]
    bench_mm!(
        "cuda",
        burn::backend::cuda::Cuda,
        &burn::backend::cuda::CudaDevice::default(),
        [tensor::DType::F32, tensor::DType::BF16, tensor::DType::F16]
    );

    #[cfg(feature = "libtorch")]
    bench_mm!(
        "libtorch",
        burn::backend::libtorch::LibTorch,
        &burn::backend::libtorch::LibTorchDevice::Cpu,
        [tensor::DType::F32, tensor::DType::BF16, tensor::DType::F16]
    );

    #[cfg(feature = "libtorch-cuda")]
    bench_mm!(
        "libtorch-cuda",
        burn::backend::libtorch::LibTorch,
        &burn::backend::libtorch::LibTorchDevice::Cuda(0),
        [tensor::DType::F32, tensor::DType::BF16, tensor::DType::F16]
    );

    #[cfg(feature = "ndarray")]
    bench_mm!(
        if cfg!(feature = "openblas") {
            "openblas"
        } else {
            "ndarray"
        },
        burn::backend::ndarray::NdArray,
        &burn::backend::ndarray::NdArrayDevice::Cpu,
        [tensor::DType::F32]
    );

    #[cfg(feature = "rocm")]
    bench_mm!(
        "rocm",
        burn::backend::rocm::Rocm,
        &burn::backend::rocm::RocmDevice::default(),
        [tensor::DType::F32, tensor::DType::BF16, tensor::DType::F16]
    );

    #[cfg(all(feature = "wgpu", not(feature = "vulkan")))]
    bench_mm!(
        "wgpu",
        burn::backend::wgpu::Wgpu,
        &burn::backend::wgpu::WgpuDevice::DefaultDevice,
        [tensor::DType::F32, tensor::DType::F16]
    );

    #[cfg(feature = "vulkan")]
    bench_mm!(
        "vulkan",
        burn::backend::wgpu::Vulkan,
        &burn::backend::wgpu::WgpuDevice::DefaultDevice,
        [tensor::DType::F32, tensor::DType::F16]
    );
}
