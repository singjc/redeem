use candle_core::{Result, Tensor};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use sysinfo::System;
use tqdm::tqdm;
use tqdm::Tqdm;

/// A thread-safe progress bar implementation using `tqdm`.
///
/// This struct manages a progress bar that updates asynchronously in a separate thread.
/// It ensures safe concurrent updates by using an atomic counter and a message-passing
/// channel to send update signals.
///
/// # Fields
/// * `total` - The total number of steps in the progress bar.
/// * `progress` - A shared, mutex-protected `tqdm` progress bar.
/// * `count` - An atomic counter to track progress updates.
/// * `sender` - A channel sender to send progress update messages.
/// * `progress_thread` - An optional handle for the background progress update thread.
/// * `description` - A description displayed alongside the progress bar.
pub struct Progress {
    total: usize,
    progress: Arc<Mutex<Tqdm<Range<usize>>>>,
    count: AtomicUsize,          // Atomic counter for tracking progress
    sender: mpsc::Sender<usize>, // Channel to send updates
    progress_thread: Option<thread::JoinHandle<()>>, // Background thread to update tqdm
    description: String,         // Description for the progress bar
}

impl Progress {
    /// Creates a new progress bar with the given total steps and description.
    ///
    /// This function initializes a `tqdm` progress bar and spawns a background thread
    /// to update the bar asynchronously.
    ///
    /// # Arguments
    /// * `total` - The total number of steps for the progress bar.
    /// * `description` - A string describing the progress bar.
    ///
    /// # Returns
    /// * A new `Progress` instance.
    ///
    /// # Example
    /// ```
    /// let progress = Progress::new(100, "Processing data");
    /// ```
    pub fn new(total: usize, description: &str) -> Self {
        let progress = Arc::new(Mutex::new(tqdm(0..total).desc(Some(description)))); // Initialize Tqdm
        let count = AtomicUsize::new(0);

        let (tx, rx) = mpsc::channel();
        let progress_clone = Arc::clone(&progress);

        // Spawn a thread to handle progress updates
        let handle = thread::spawn(move || {
            for _ in rx {
                let _ = progress_clone.lock().unwrap().pbar.update(1); // Always update by 1
            }
        });

        Self {
            total,
            progress,
            count,
            sender: tx,
            progress_thread: Some(handle),
            description: description.to_string(),
        }
    }

    /// Increments the progress counter safely and updates the progress bar.
    ///
    /// This function uses an atomic counter to track updates and sends an update signal
    /// to the progress thread via a channel.
    ///
    /// If the total progress exceeds the expected `total`, a warning is printed to avoid overflow.
    ///
    /// # Example
    /// ```
    /// let progress = Progress::new(100, "Loading");
    /// progress.inc();
    /// ```
    pub fn inc(&self) {
        let new_count = self.count.fetch_add(1, Ordering::AcqRel) + 1;

        if new_count > self.total {
            log::trace!("⚠️ WARNING: Progress logger received and extra update! This is likely because the logger was initialized with an incorrect total counter, and the process is iterating beyond that counter.");
            return; // Prevent overflow
        }

        let _ = self.sender.send(1); // Always send 1 instead of new_count
    }

    /// Updates the progress bar's description.
    pub fn update_description(&self, new_desc: &str) {
        let mut progress = self.progress.lock().unwrap();
        progress.set_desc(Some(new_desc)); // Update the description dynamically
    }

    /// Finalizes the progress bar by ensuring all updates are completed.
    ///
    /// This function drops the sender to close the channel and waits for the background thread
    /// to finish execution, ensuring no updates are lost.
    ///
    /// # Example
    /// ```
    /// let progress = Progress::new(100, "Downloading files");
    /// for _ in 0..100 {
    ///     progress.inc();
    /// }
    /// progress.finish();
    /// ```
    pub fn finish(self) {
        drop(self.sender); // Close the channel to signal the thread to finish
        if let Some(handle) = self.progress_thread {
            let _ = handle.join(); // Wait for the progress thread to exit
        }
    }
}

/// Returns the current Resident Set Size (RSS) memory usage of the process in bytes.
///
/// The RSS represents the amount of memory currently used by the process, including code,
/// data, and shared libraries but excluding swap space.
///
/// # Returns
/// * `u64` - The used memory in kilobytes (KB).
///
/// # Example
/// ```
/// let mem_usage = get_rss_memory();
/// println!("Current memory usage: {} KB", mem_usage / 1024);
/// ```
pub fn get_rss_memory() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.used_memory()
}

pub fn print_tensor(
    tensor: &Tensor,
    decimal_places: usize,
    max_rows: Option<usize>,
    max_cols: Option<usize>,
) -> Result<()> {
    let shape = tensor.shape().dims();
    if shape.len() != 3 {
        return Err(candle_core::Error::Msg("Expected a 3D tensor".to_string()));
    }

    let (batch, seq_len, features) = (shape[0], shape[1], shape[2]);
    let data = tensor.to_vec3::<f32>()?;

    let max_cols = max_cols.unwrap_or(features);

    println!("tensor([");
    for b in 0..batch {
        println!("  [");
        let rows_to_print = max_rows.unwrap_or(seq_len).min(seq_len);
        for s in 0..rows_to_print {
            print!("    [");
            let cols_to_print = max_cols.min(features);
            if cols_to_print * 2 < features {
                for f in 0..cols_to_print {
                    print!("{:.*e}", decimal_places, data[b][s][f]);
                    if f < cols_to_print - 1 {
                        print!(", ");
                    }
                }
                print!(", ..., ");
                for f in (features - cols_to_print)..features {
                    print!("{:.*e}", decimal_places, data[b][s][f]);
                    if f < features - 1 {
                        print!(", ");
                    }
                }
            } else {
                for f in 0..features {
                    print!("{:.*e}", decimal_places, data[b][s][f]);
                    if f < features - 1 {
                        print!(", ");
                    }
                }
            }
            if s < rows_to_print - 1 {
                println!("],");
            } else {
                print!("]");
            }
        }
        if rows_to_print < seq_len {
            println!(",");
            println!("    ...");
        }
        println!("  ],");
    }
    println!("])");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_progress_it() {
        let total = 24;
        let pbar = Progress::new(total, "Testing Rayon");
        let v = vec![1; total];
        let mapped: Vec<i32> = v
            .into_par_iter()
            .enumerate()
            .map(|(i, x)| {
                pbar.inc();
                pbar.update_description(&format!("Processing item {}", i));
                x * 100
            })
            .collect();
        pbar.finish();
        println!("{:?}", mapped);
    }

    #[test]
    fn test_get_rss_memory() {
        let mem_usage = get_rss_memory();

        // mem_usage is in bytes, convert to MB
        println!("Memory usage: {} MB", mem_usage / 1024 / 1024);
        // print mem_usage in GB
        println!("Memory usage: {} GB", mem_usage / 1024 / 1024 / 1024);

        // Ensure memory usage is a reasonable positive value
        assert!(
            mem_usage > 0,
            "Memory usage should be greater than zero, but got {}",
            mem_usage
        );
    }
}
