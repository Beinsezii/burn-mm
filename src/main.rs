use burn::prelude::*;
use burn::tensor;
use std::time::Instant;

macro_rules! bench_mm {
    ($backend:ty, $device:expr, $dtypes:expr) => {
        for dtype in $dtypes {
            for n in [256, 512, 1024, 2048, 4096] {
                let flops = n * n * n * 2;
                let lhs =
                    Tensor::<$backend, 2>::random([n, n], tensor::Distribution::Default, $device)
                        .cast(dtype);
                let rhs =
                    Tensor::<$backend, 2>::random([n, n], tensor::Distribution::Default, $device)
                        .cast(dtype);
                let duration = flops as f64 * 1e-12
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
                println!("{duration:6.2}\t{n:4}\t{dtype:?}");
            }
        }
    };
}

fn main() {
    #[cfg(feature = "cuda")]
    bench_mm!(
        burn::backend::cuda::Cuda,
        &burn::backend::cuda::CudaDevice::default(),
        [
            // tensor::DType::F64,
            tensor::DType::F32,
            tensor::DType::BF16,
            tensor::DType::F16,
        ]
    );

    #[cfg(feature = "ndarray")]
    bench_mm!(
        burn::backend::ndarray::NdArray,
        &burn::backend::ndarray::NdArrayDevice::Cpu,
        [
            tensor::DType::F64,
            tensor::DType::F32,
            // tensor::DType::BF16,
            // tensor::DType::F16,
        ]
    );

    #[cfg(feature = "rocm")]
    bench_mm!(
        burn::backend::rocm::Rocm,
        &burn::backend::rocm::RocmDevice::default(),
        [
            // tensor::DType::F64,
            tensor::DType::F32,
            tensor::DType::BF16,
            tensor::DType::F16,
        ]
    );

    #[cfg(feature = "wgpu")]
    bench_mm!(
        burn::backend::wgpu::Wgpu,
        &burn::backend::wgpu::WgpuDevice::DefaultDevice,
        [
            // tensor::DType::F64,
            tensor::DType::F32,
            // tensor::DType::BF16,
            tensor::DType::F16,
        ]
    );

    #[cfg(feature = "vulkan")]
    bench_mm!(
        burn::backend::wgpu::Vulkan,
        &burn::backend::wgpu::WgpuDevice::DefaultDevice,
        [
            // tensor::DType::F64,
            tensor::DType::F32,
            // tensor::DType::BF16,
            tensor::DType::F16,
        ]
    );
}
