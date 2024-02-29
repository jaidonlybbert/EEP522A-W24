use rand::Rng;
use std::time::Duration;
use std::io::Write;
use std::time::Instant;

fn main() {
    println!("Performing Naive CPU GEMM");

    benchmark_gemm_compute_bound();
    //benchmark_gemm_memory_bound();
    //benchmark_matrix_add_cpu();
    benchmark_tiled_gemm_cpu();
}


fn benchmark_tiled_gemm_cpu() {
    // Prepare a file to write the results
    let mut file = std::fs::File::create("tiled_gemm_benchmark.csv").unwrap();
    writeln!(file, "Matrix Size,Execution Time").unwrap();

    // Iterate over matrix sizes from 10x10 to 1000x1000
    for size in (16..=200).step_by(16) {
        let time = measure_tiled_gemm(size, size, size).as_millis();
        println!("Size: {}x{}, Time: {} ms", size, size, time);
        writeln!(file, "{},{}", size, time).unwrap();
    }

    println!("Benchmark results written to tiled_gemm_benchmark.csv");
}

fn benchmark_matrix_add_cpu() {
    // Prepare a file to write the results
    let mut file = std::fs::File::create("matrix_add_benchmark.csv").unwrap();
    writeln!(file, "Matrix Size,Execution Time").unwrap();

    // Iterate over matrix sizes from 10x10 to 1000x1000
    for size in (1..=400).step_by(10) {
        for _ in 0..5 {
            let time = measure_matrix_add(size, size, size).as_micros();
            println!("Size: {}x{}, Time: {} us", size, size, time);
            writeln!(file, "{},{}", size, time).unwrap();
        }
    }

    println!("Benchmark results written to matrix_add_benchmark.csv");
}


fn benchmark_gemm_memory_bound() {
    // Prepare a file to write the results
    let mut file = std::fs::File::create("gemm_benchmark_memory.csv").unwrap();
    writeln!(file, "Matrix Size,Execution Time").unwrap();

    // Iterate over matrix sizes from 10x10 to 1000x1000
    for size in (1..=50).step_by(1) {
        for _ in 0..10 {
            let time = measure_gemm(size, size, size).as_micros();
            println!("Size: {}x{}, Time: {} us", size, size, time);
            writeln!(file, "{},{}", size, time).unwrap();
        }
    }

    println!("Benchmark results written to gemm_benchmark_memory.csv");
}


fn benchmark_gemm_compute_bound() {
    // Prepare a file to write the results
    let mut file = std::fs::File::create("gemm_benchmark.csv").unwrap();
    writeln!(file, "Matrix Size,Execution Time").unwrap();

    // Iterate over matrix sizes from 10x10 to 1000x1000
    for size in (16..=200).step_by(16) {
        let time = measure_gemm(size, size, size).as_millis();
        println!("Size: {}x{}, Time: {} ms", size, size, time);
        writeln!(file, "{},{}", size, time).unwrap();
    }

    println!("Benchmark results written to gemm_benchmark.csv");
}

fn gemm(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>, c: &mut Vec<Vec<f64>>) {
    let n = a.len();
    let m = b[0].len();
    let p = a[0].len();

    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..p {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
}

fn tiled_gemm(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>, c: &mut Vec<Vec<f64>>) {
    let n = a.len();
    let m = b[0].len();
    let p = a[0].len();

    let tile_width = 16;
    let tiles_y = m / tile_width;
    let tiles_x = n / tile_width; 
    let phases = p / tile_width;

    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            for phase in 0..phases {
                for i in 0..tile_width {
                    for j in 0..tile_width {
                        let output_xidx = tile_x * tile_width + j;
                        let output_yidx = tile_y * tile_width + i;
                        let mut sum = 0.0;
                        for k in 0..tile_width {
                            sum += a[output_yidx][k + phase * tile_width] * 
                                        b[k + phase * tile_width][output_xidx];
                        }
                        c[output_yidx][output_xidx] += sum;
                    }
                }
            }
        }
    }
}

fn measure_matrix_add(m: usize, n: usize, p: usize) -> Duration {
    let a = create_random_matrix(m, p);
    let b = create_random_matrix(p, n);
    let mut c = vec![vec![0.0; n]; m];

    let start = Instant::now();
    add_matrices(&a, &b, &mut c);
    let duration = start.elapsed();

    duration
}

fn measure_tiled_gemm(m: usize, n: usize, p: usize) -> Duration {
    assert!(m % 16 == 0);
    assert!(n % 16 == 0);
    assert!(p % 16 == 0);
    let a = create_random_matrix(m, p);
    let b = create_random_matrix(p, n);
    let mut c = vec![vec![0.0; n]; m];

    let start = Instant::now();
    tiled_gemm(&a, &b, &mut c);
    let duration = start.elapsed();

    duration
}

fn measure_gemm(m: usize, n: usize, p: usize) -> Duration {
    let a = create_random_matrix(m, p);
    let b = create_random_matrix(p, n);
    let mut c = vec![vec![0.0; n]; m];

    let start = Instant::now();
    gemm(&a, &b, &mut c);
    let duration = start.elapsed();

    duration
}

fn create_random_matrix(m: usize, n: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let mut matrix = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            matrix[i][j] = rng.gen::<f64>();
        }
    }
    matrix
}

fn add_matrices(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>, c: &mut Vec<Vec<f64>>) {
    let m = a.len();
    let n = a[0].len();

    for i in 0..m {
        for j in 0..n {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}
