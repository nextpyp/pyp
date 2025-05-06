
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
cargo build --release
```

The executable will appear at `target/release/launcher`.

**NOTE**: While this will build a working executable for your system,
building an executable that will work on other systems takes some extra work.
The strategy here is to dynamically link against the oldest GNU libc
we can find. The sibling `next` project has a CentOS 7 container
for building Rust applications against a very old GNU libc.
Run the Gradle task `vmBuildPypLauncher` to build inside the
container (inside the dev VM) and then look for the built executable
in the `build/pypLauncher/release` folder. That executable should run on any relatively
modern linux distribution.


## Testing

To run the test suite, run the following command:

```shell
cargo test -- --test-threads=1
```
