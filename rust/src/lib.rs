//! # kenlm
//!
//! Rust bindings for the KenLM language model library.
//!
//! This crate provides a safe, high-level Rust API for KenLM.
//!
//! ## Usage
//!
//! ```no_run
//! use kenlm::{Model, Config, LoadMethod};
//!
//! fn main() {
//!     let mut config = Config::new();
//!     config.load_method(LoadMethod::LAZY);
//!     let model = Model::load("path/to/model.arpa", &config).unwrap();
//!
//!     let score = model.score("this is a sentence", true, true);
//!     println!("Score: {}", score);
//!
//!     let perplexity = model.perplexity("this is a sentence");
//!     println!("Perplexity: {}", perplexity);
//! }
//! ```
//!
//! ## Testing
//!
//! To run the tests, you need to have a C++ compiler and CMake installed.
//! The tests require the KenLM source code to be present in the parent directory.
//!
//! ```bash
//! cargo test --manifest-path rust/Cargo.toml
//! ```
//!
//! ## Current Limitations
//!
//! - The incremental API using `State` objects is not yet implemented.

#[cxx::bridge(namespace = "kenlm_cxx")]
mod ffi {
    // C++ types and functions exposed to Rust
    unsafe extern "C++" {
        include!("rust/src/kenlm_cxx.hh");

        type Model;
        type State;
        type Config;

        fn new_config() -> UniquePtr<Config>;
        fn set_load_method(config: &mut Config, method: LoadMethod);

        fn load_model(path: &str, config: &Config) -> Result<UniquePtr<Model>>;

        fn get_order(model: &Model) -> u32;
        fn vocab_contains(model: &Model, word: &str) -> bool;

        fn score(model: &Model, sentence: &str, bos: bool, eos: bool) -> f32;

        #[derive(Debug)]
        struct FullScoreResult {
            log_prob: f32,
            ngram_length: i32,
            oov: bool,
        }

        fn full_scores(model: &Model, sentence: &str, bos: bool, eos: bool) -> Vec<FullScoreResult>;

        fn begin_sentence_write(model: &Model, state: &mut State);
        fn null_context_write(model: &Model, state: &mut State);

        fn base_score(model: &Model, in_state: &mut State, word: &str, out_state: &mut State) -> f32;
    }

    // Enums
    #[namespace = "lm::ngram"]
    enum LoadMethod {
        LAZY,
        POPULATE_OR_LAZY,
        POPULATE_OR_READ,
        READ,
        PARALLEL_READ,
    }
}

pub use ffi::{LoadMethod, FullScoreResult};

pub struct Config {
    inner: cxx::UniquePtr<ffi::Config>,
}

impl Config {
    pub fn new() -> Self {
        Self {
            inner: ffi::new_config(),
        }
    }

    pub fn load_method(&mut self, method: LoadMethod) {
        ffi::set_load_method(&mut self.inner, method);
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Model {
    ptr: cxx::UniquePtr<ffi::Model>,
}

impl Model {
    pub fn load(path: &str, config: &Config) -> Result<Self, cxx::Exception> {
        let model = ffi::load_model(path, &config.inner)?;
        Ok(Self { ptr: model })
    }

    pub fn order(&self) -> u32 {
        ffi::get_order(&self.ptr)
    }

    pub fn perplexity(&self, sentence: &str) -> f32 {
        let words = sentence.split_whitespace().count() + 1; // For </s>
        10.0f32.powf(-self.score(sentence, true, true) / words as f32)
    }

    pub fn contains(&self, word: &str) -> bool {
        ffi::vocab_contains(&self.ptr, word)
    }

    pub fn score(&self, sentence: &str, bos: bool, eos: bool) -> f32 {
        ffi::score(&self.ptr, sentence, bos, eos)
    }

    pub fn full_scores(&self, sentence: &str, bos: bool, eos: bool) -> Vec<FullScoreResult> {
        ffi::full_scores(&self.ptr, sentence, bos, eos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score() {
        let mut config = Config::new();
        config.load_method(LoadMethod::LAZY);
        let model = Model::load("../lm/test.arpa", &config).unwrap();
        let score = model.score("a b c", true, true);
        assert!((score - (-4.322589)).abs() < 0.001);
    }

    #[test]
    fn test_order() {
        let mut config = Config::new();
        config.load_method(LoadMethod::LAZY);
        let model = Model::load("../lm/test.arpa", &config).unwrap();
        assert_eq!(model.order(), 3);
    }

    #[test]
    fn test_contains() {
        let mut config = Config::new();
        config.load_method(LoadMethod::LAZY);
        let model = Model::load("../lm/test.arpa", &config).unwrap();
        assert!(model.contains("a"));
        assert!(!model.contains("zyxw"));
    }
}
