
# PYP Launcher

PYP Launcher is a program that launches PYP inside an apptainer container depending on the settings in a TOML file.

The goal is to bootstrap the problem of launching PYP, which runs inside a container, but the path to that
container is in a config file, but the libraries need to read the config file are also inside the container.
It's a circular dependency. We break the dependency by building a launcher program (this one) that lives outside
of the container, but can still access libraries to read the config file, to eventually launch the container.

The launcher is written in Rust, compiles to a native binary, and statically links all its runtime dependencies,
so it's as self-contained as possible. That way, we can run the launcher to read the config file without needing
any special setup other than copying the launcher executable onto the filesystem.


## Building

### Prerequisites

1. Install Rust \
   https://www.rust-lang.org/tools/install


### Cargo

`cargo` is the build tool for Rust programs.
To build the launcher, run the following command:

```shell
RUSTFLAGS="-C target-feature=+crt-static" cargo build --release --target x86_64-unknown-linux-gnu
```

The executable will appear at `target/x86_64-unknown-linux-gnu/release/launcher`.

**NOTE**: We have to put the `rustc` compile flags in the environment variable
for now (because `.cargo/config.toml` only applies to the current crate,
not any dependency crates), but we could someday move them into `Cargo.toml`
after `cargo` moves the `profile-rustflags` feature to the stable branch.
See https://github.com/rust-lang/cargo/issues/10271 for the tracking issue.
After that, the build command could just be `cargo build --release`.

**NOTE**: The explicit compilation target (eg `x86_64-unknown-linux-gnu`)
is needed to work around an issue where we want to compile the final binary
with purely static linking, but we want to compile the proc macro
(used by the `thiserror` crate) on the host with regular settings.
Specifying a target explicitly puts `cargo`/`rustc` into cross-compilation
mode with handles compiling proc macros correctly. See the Rust GitHub issue
for more details:
https://github.com/rust-lang/rust/issues/78210


## Testing

To run the test suite, run the following command:

```shell
cargo test -- --test-threads=1
```
