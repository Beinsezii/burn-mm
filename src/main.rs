use burn::prelude::*;
use burn::tensor;
use std::time::Instant;

fn main() {
    type Backend = burn::backend::rocm::Rocm;
    let device = burn::backend::rocm::HipDevice::new(0);
    for dtype in [
        tensor::DType::F32, // page faults sometimes
        // tensor::DType::BF16, // page faults always
        tensor::DType::F16,
    ] {
        for n in [256, 512, 1024, 2048, 4096] {
            let flops = n * n * n * 2;
            let lhs = Tensor::<Backend, 2>::random([n, n], tensor::Distribution::Default, &device)
                .cast(dtype);
            let rhs = Tensor::<Backend, 2>::random([n, n], tensor::Distribution::Default, &device)
                .cast(dtype);
            let duration = flops as f64 * 1e-12
                / (0..20)
                    .map(|_| {
                        let clock = Instant::now();
                        let _ = lhs.clone().matmul(rhs.clone()); // can't use refs?
                        Backend::sync(&device);
                        Instant::now() - clock
                    })
                    .min()
                    .unwrap()
                    .as_secs_f64();
            println!("{duration:6.2}\t{n:4}\t{dtype:?}");
        }
    }
}
