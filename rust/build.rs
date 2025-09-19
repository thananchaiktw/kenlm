use std::path::Path;

fn main() {
    // Build the KenLM C++ library
    let dst = cmake::build("..");

    // Tell cargo to look for libraries in the specified directory
    println!("cargo:rustc-link-search=native={}/lib", dst.display());

    // Tell cargo to link the KenLM library
    println!("cargo:rustc-link-lib=static=kenlm");
    println!("cargo:rustc-link-lib=static=kenlm_util");
    println!("cargo:rustc-link-lib=static=double-conversion");
    println!("cargo:rustc-link-lib=static=kenlm_builder");
    println!("cargo:rustc-link-lib=static=kenlm_filter");
    println!("cargo:rustc-link-lib=static=kenlm_interpolate");


    // Tell cargo to link against the C++ standard library
    if let Some(tool) = cxx_build::get_compiler() {
        if tool.is_like_msvc() {
            // Nothing to do
        } else {
            println!("cargo:rustc-link-lib=stdc++");
        }
    }


    // Build the CXX bridge
    let mut build = cxx_build::bridge("src/lib.rs");
    build
        .file("src/kenlm_cxx.cc")
        .include(Path::new(".."))
        .include(Path::new("../util"))
        .include(Path::new("../lm"))
        .flag_if_supported("-std=c++11")
        .compile("kenlm-cxx");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/kenlm_cxx.cc");
    println!("cargo:rerun-if-changed=src/kenlm_cxx.hh");
}
