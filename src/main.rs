use tab_pfn_rs::tabpfn::settings::settings;

fn main() {
    println!("TabPFN-rs starting...");
    
    // Load settings to verify basic functionality
    let config = settings();
    println!("Model cache dir: {:?}", config.tabpfn.model_cache_dir);
    println!("Allow CPU large dataset: {}", config.tabpfn.allow_cpu_large_dataset);
    
    println!("TabPFN-rs initialized successfully!");
}
