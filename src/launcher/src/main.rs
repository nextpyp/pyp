
#[cfg(test)]
mod test;

mod errors;
mod exit;
mod config;
mod launcher;


use std::collections::HashMap;
use std::env;

use errors::LauncherError;
use exit::Exit;
use launcher::Launcher;


pub const ENVNAME_HOME: &str = "HOME";
pub const ENVNAME_CONFIG: &str = "PYP_CONFIG";
pub const ENVNAME_SINGULARITY: &str = "SINGULARITY_CONTAINER";
pub const ENVNAME_APPTAINER: &str = "APPTAINER_CONTAINER";
pub const ENVNAME_VERBOSE: &str = "PYP_LAUNCHER_VERBOSE";

pub const ENVNAME_CONTAINERS: [&str; 2] = [
	ENVNAME_SINGULARITY,
	ENVNAME_APPTAINER
];

pub const ALL_ENVNAMES: [&str; 5] = [
	ENVNAME_HOME,
	ENVNAME_CONFIG,
	ENVNAME_SINGULARITY,
	ENVNAME_APPTAINER,
	ENVNAME_VERBOSE
];

const DEFAULT_CONFIG_PATH: &str = ".pyp/config.toml";
const DEFAULT_CONTAINER_CMD: &str = "singularity";
const PYP_PATH_INTERNAL: &str = "/opt/pyp";
const PYP_BIN_RELPATH: &str = "bin/run";


fn main() -> Exit {

	// read all the program arguments now
	let args = env::args_os()
		.map(|s| s.to_string_lossy().to_string())
		.collect::<Vec<String>>();

	// read all the environment variables now
	let mut envvars = HashMap::<String,String>::new();
	for name in ALL_ENVNAMES {
		if let Some(val) = env::var_os(name) {
			envvars.insert(name.to_string(),val.to_string_lossy().to_string());
		}
	}

	// launch the process and forward the exit code
	let launcher = match Launcher::from(args, envvars) {
		Ok(l) => l,
		Err(e) => return Exit::Error(e)
	};
	let exit_status = match launcher.launch() {
		Ok(s) => s,
		Err(e) => return Exit::Error(e)
	};
	Exit::Launched(exit_status)
}
