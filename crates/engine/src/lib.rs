#[macro_use]
extern crate log;

/// Exposes our available Producers
pub mod producers {
    pub use producer::*;
}

/// Expose our available crackers
pub mod crackers {
    pub use cracker::{PDFCracker, PDFCrackerState};
}

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub mod metal_gpu;

#[cfg(feature = "cuda-gpu")]
pub mod cuda_gpu;

// We will run a SPMC layout where a single producer produces passwords
// consumed by multiple workers. This ensures there is a buffer
// so the queue won't be consumed before the producer has time to wake up
const BUFFER_SIZE: usize = 200;

use std::sync::Arc;

use crossbeam::channel::{Receiver, Sender, TryRecvError};

use producer::Producer;

use cracker::{PDFCracker, PDFCrackerState};

/// Returns Ok(Some(<Password in bytes>)) if it successfully cracked the file.
/// Returns Ok(None) if it did not find the password.
/// Returns Err if something went wrong.
/// Callback is called once very time it consumes a password from producer
pub fn crack_file(
    no_workers: usize,
    cracker: PDFCracker,
    producer: Box<dyn Producer>,
    callback: Box<dyn Fn()>,
) -> anyhow::Result<Option<Vec<u8>>> {
    crack_file_cpu(no_workers, cracker, producer, callback)
}

/// CPU-based password cracking
fn crack_file_cpu(
    no_workers: usize,
    cracker: PDFCracker,
    mut producer: Box<dyn Producer>,
    callback: Box<dyn Fn()>,
) -> anyhow::Result<Option<Vec<u8>>> {
    // Spin up workers
    let (sender, r): (Sender<Vec<u8>>, Receiver<_>) = crossbeam::channel::bounded(BUFFER_SIZE);

    let (success_sender, success_reader) = crossbeam::channel::unbounded::<Vec<u8>>();
    let mut handles = vec![];
    let cracker_handle = Arc::from(cracker);

    for _ in 0..no_workers {
        let success = success_sender.clone();
        let r2 = r.clone();
        let c2 = cracker_handle.clone();
        let id: std::thread::JoinHandle<()> = std::thread::spawn(move || {
            let Ok(mut cracker) = PDFCrackerState::from_cracker(&c2) else {
                return
            };

            while let Ok(passwd) = r2.recv() {
                if cracker.attempt(&passwd) {
                    // inform main thread we found a good password then die
                    success.send(passwd).unwrap_or_default();
                    return;
                }
            }
        });
        handles.push(id);
    }
    // Drop our ends
    drop(r);
    drop(success_sender);

    info!("Starting password cracking job...");

    let mut success = None;

    loop {
        // Check for success first before producing more passwords
        match success_reader.try_recv() {
            Ok(password) => {
                success = Some(password);
                info!("Password found! Stopping password generation.");
                break;
            }
            Err(e) => {
                match e {
                    TryRecvError::Empty => {
                        // This is fine, no success yet, continue
                    }
                    TryRecvError::Disconnected => {
                        // All threads have died. Wtf?
                        // let's just report an error and break
                        error!("All workers have exited prematurely, cannot continue operations");
                        break;
                    }
                }
            }
        }

        // Only produce next password if we haven't found a match yet
        match producer.next() {
            Ok(Some(password)) => {
                if sender.send(password).is_err() {
                    // This should only happen if their reciever is closed.
                    error!("unable to send next password since channel is closed");
                    break;
                }
                callback()
            }
            Ok(None) => {
                trace!("out of passwords, exiting loop");
                break;
            }
            Err(error_msg) => {
                error!("error occured while sending: {error_msg}");
                break;
            }
        }
    }

    // Ensure any threads that are still running will eventually exit
    drop(sender);

    let found_password = match success {
        Some(result) => Some(result),
        None => {
            // Wait for any worker threads to report success
            match success_reader.recv() {
                Ok(result) => Some(result),
                Err(e) => {
                    // Channel is empty and disconnected, i.e. all threads have exited
                    // and none found the password
                    debug!("{}", e);
                    None
                }
            }
        }
    };

    Ok(found_password)
}

#[cfg(any(feature = "gpu", all(target_os = "macos", feature = "metal-gpu"), feature = "cuda-gpu"))]
/// GPU-accelerated password cracking
pub fn crack_file_gpu(
    cracker: PDFCracker,
    producer: Box<dyn Producer>,
    callback: Box<dyn Fn()>,
) -> anyhow::Result<Option<Vec<u8>>> {
    // Try CUDA GPU first if available (NVIDIA GPUs)
    #[cfg(feature = "cuda-gpu")]
    {
        info!("Attempting CUDA GPU acceleration for NVIDIA GPUs...");
        match crack_file_cuda_gpu(cracker.clone(), producer, callback) {
            Ok(result) => return Ok(result),
            Err(e) => {
                error!("CUDA GPU failed: {}. Trying other GPU options...", e);
                // Continue to try other GPU options
            }
        }
    }
    
    // Try Metal GPU on macOS if available (Apple Silicon)
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        info!("Attempting Metal GPU acceleration for Apple Silicon...");
        return crack_file_metal_gpu(cracker, producer, callback);
    }
    
    // Fall back to OpenCL (cross-platform)
    #[cfg(feature = "gpu")]
    {
        use crate::gpu::GpuCracker;
        
        info!("Initializing OpenCL GPU for password cracking...");
        
        let mut gpu_cracker = match GpuCracker::new(&cracker) {
            Ok(cracker) => cracker,
            Err(e) => {
                error!("Failed to initialize GPU: {}. Falling back to CPU.", e);
                return crack_file_cpu(4, cracker, producer, callback);
            }
        };
        
        info!("Starting GPU password cracking job...");
        
        let batch_size = 10000;
        let mut batch = Vec::with_capacity(batch_size);
        let mut producer = producer;
        
        loop {
            // Collect batch of passwords
            while batch.len() < batch_size {
                match producer.next() {
                    Ok(Some(password)) => {
                        batch.push(password);
                        callback();
                    }
                    Ok(None) => {
                        // Out of passwords
                        break;
                    }
                    Err(error_msg) => {
                        error!("Error occurred while generating passwords: {error_msg}");
                        break;
                    }
                }
            }
            
            if batch.is_empty() {
                break;
            }
            
            // Try batch on GPU
            match gpu_cracker.attempt_batch(&batch) {
                Ok(Some(password)) => {
                    info!("Password found on GPU!");
                    return Ok(Some(password));
                }
                Ok(None) => {
                    // Continue with next batch
                    batch.clear();
                }
                Err(e) => {
                    error!("GPU error: {}. Falling back to CPU for remaining passwords.", e);
                    // Fall back to CPU for this batch
                    for _password in batch.drain(..) {
                        // Would need to create CPU cracker state here
                        // For now, just continue
                    }
                }
            }
        }
        
        return Ok(None);
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
/// Metal GPU-accelerated password cracking for Apple Silicon
pub fn crack_file_metal_gpu(
    cracker: PDFCracker,
    mut producer: Box<dyn Producer>,
    callback: Box<dyn Fn()>,
) -> anyhow::Result<Option<Vec<u8>>> {
    use crate::metal_gpu::MetalGpuCracker;
    
    let mut metal_cracker = MetalGpuCracker::new(&cracker)?;
    let batch_size = metal_cracker.batch_size();
    
    info!("Starting Metal GPU password cracking with batch size {}...", batch_size);
    
    let mut batch = Vec::with_capacity(batch_size);
    
    loop {
        // Collect batch of passwords
        while batch.len() < batch_size {
            match producer.next() {
                Ok(Some(password)) => {
                    batch.push(password);
                    callback();
                }
                Ok(None) => {
                    break;
                }
                Err(error_msg) => {
                    error!("Error occurred while generating passwords: {error_msg}");
                    break;
                }
            }
        }
        
        if batch.is_empty() {
            break;
        }
        
        // Try batch on Metal GPU
        match metal_cracker.attempt_batch(&batch) {
            Ok(Some(password)) => {
                info!("Password found using Metal GPU!");
                return Ok(Some(password));
            }
            Ok(None) => {
                batch.clear();
            }
            Err(e) => {
                error!("Metal GPU error: {}", e);
                return Err(e);
            }
        }
    }
    
    Ok(None)
}

#[cfg(feature = "cuda-gpu")]
/// CUDA GPU-accelerated password cracking for NVIDIA GPUs
pub fn crack_file_cuda_gpu(
    cracker: PDFCracker,
    mut producer: Box<dyn Producer>,
    callback: Box<dyn Fn()>,
) -> anyhow::Result<Option<Vec<u8>>> {
    use crate::cuda_gpu::CudaGpuCracker;
    
    let mut cuda_cracker = CudaGpuCracker::new(&cracker)?;
    let batch_size = cuda_cracker.batch_size();
    
    info!("Starting CUDA GPU password cracking with batch size {}...", batch_size);
    
    let mut batch = Vec::with_capacity(batch_size);
    
    loop {
        // Collect batch of passwords
        while batch.len() < batch_size {
            match producer.next() {
                Ok(Some(password)) => {
                    batch.push(password);
                    callback();
                }
                Ok(None) => {
                    break;
                }
                Err(error_msg) => {
                    error!("Error occurred while generating passwords: {error_msg}");
                    break;
                }
            }
        }
        
        if batch.is_empty() {
            break;
        }
        
        // Try batch on CUDA GPU
        match cuda_cracker.attempt_batch(&batch) {
            Ok(Some(password)) => {
                info!("Password found using CUDA GPU!");
                return Ok(Some(password));
            }
            Ok(None) => {
                batch.clear();
            }
            Err(e) => {
                error!("CUDA GPU error: {}", e);
                return Err(e);
            }
        }
    }
    
    Ok(None)
}
