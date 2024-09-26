
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use crate::{DEFAULT_CONFIG_PATH, DEFAULT_CONTAINER_CMD, ENVNAME_CONFIG, ENVNAME_CONTAINERS, ENVNAME_HOME, ENVNAME_VERBOSE, LauncherError, PYP_BIN_RELPATH, PYP_PATH_INTERNAL};
use crate::config::{ConfigReader, ContainerExec};


#[derive(PartialEq, Eq, Debug)]
pub struct Launcher {
	pub setup_command: Option<LaunchCommand>,
	pub command: LaunchCommand,
	pub verbose: bool
}

impl Launcher {

	pub fn new(setup_command: Option<LaunchCommand>, command: LaunchCommand) -> Self {
		Self {
			setup_command,
			command,
			verbose: false
		}
	}

	pub fn unwrapped(pyp_command: LaunchCommand) -> Self {
		Self::new(None, pyp_command)
	}

	pub fn wrapped(pyp_command: LaunchCommand, setup_command: Option<LaunchCommand>, container_command: LaunchCommand) -> Self {

		// wrap the pyp command with the container command
		let command = LaunchCommand::from(&container_command)
			.append(&pyp_command);

		Self::new(setup_command, command)
	}

	pub fn from(args: Vec<String>, envvars: HashMap<String,String>) -> Result<Self,LauncherError> {

		// check for verbosity
		let verbose = envvars.get(ENVNAME_VERBOSE) == Some(&"1".to_string());
		if verbose {
			println!("PYP Launcher is running in verbose mode");
		}

		// read the pyp command
		let caller_path = PathBuf::from(
			args.get(0)
				.ok_or_else(|| LauncherError::NoCallerPath)?
		);
		let pyp_program = caller_path.file_stem()
			.ok_or_else(|| LauncherError::EmptyCallerPath)?
			.to_string_lossy()
			.to_string();
		let pyp_command = LaunchCommand::new(
				PathBuf::from(PYP_PATH_INTERNAL)
					.join(PYP_BIN_RELPATH)
					.join(&pyp_program)
					.to_string_lossy().to_string()
			)
			.args(args[1 ..].to_vec());
		println!("Launching {} ...", pyp_program);

		if verbose {
			println!("PYP command: {} {}", pyp_command.program, pyp_command.args.join(" "));
		}

		// are we running inside a container already?
		let container_names = ENVNAME_CONTAINERS
			.into_iter()
			.filter(|&name| {
				match envvars.get(name) {
					Some(val) => !val.is_empty(),
					None => false
				}
			})
			.collect::<Vec<_>>();
		if !container_names.is_empty() {

			if verbose {
				println!("Already inside container: {}", container_names.join(", "));
			}

			// yup, just pass through the command
			return Ok(Launcher::unwrapped(pyp_command).verbose(verbose));
		}

		// nope, wrap the command in a container

		// find the home directory, if any
		let home = envvars.get(ENVNAME_HOME)
			.map(|s| PathBuf::from(s))
			.ok_or_else(|| LauncherError::NoEnvVar(ENVNAME_HOME.to_string()))?;

		// get the path to the configuration file
		let config_path_default = home.join(DEFAULT_CONFIG_PATH);
		let config_path: PathBuf = match envvars.get(ENVNAME_CONFIG) {
			Some(p) => PathBuf::from(p),
			None => config_path_default.clone()
		};
		if !config_path.exists() {
			return Err(LauncherError::NoConfig {
				path: config_path_default.to_string_lossy().to_string(),
				envname: ENVNAME_CONFIG.to_string()
			});
		}

		// read the config file
		println!("Reading configuration file at: {}", config_path.to_string_lossy());
		let config = ConfigReader::read(&config_path)?;
		let config_pyp = config.get_section("pyp")?;
		let container_path = config_pyp.get_file("container")?;
		let binds = config_pyp.get_optional_folders("binds")?;
		let scratch = config_pyp.get_optional_unchecked_folder("scratch")?;
		let container_exec = config_pyp.get_container_exec("containerExec", "module", "exec")?;
		let pyp_sources = config_pyp.get_optional_folder("sources")?;

		if verbose {
			println!("Config: container: {:?}", container_path);
			if let Some(binds) = &binds {
				for bind in binds {
					println!("Config: bind: {:?}", bind);
				}
			} else {
				println!("Config: no binds");
			}
			println!("Config: scratch: {:?}", scratch);
			println!("Config: containerExec: {:?}", container_exec);
			println!("Config: sources: {:?}", pyp_sources);
		}

		// build the container commands
		let (setup_command, container_command) = build_container_commands(
			&home,
			&config_path,
			container_path,
			&binds,
			&scratch,
			&container_exec,
			&pyp_sources,
			verbose
		);

		Ok(Launcher::wrapped(pyp_command, setup_command, container_command).verbose(verbose))
	}

	pub fn verbose(mut self, val: bool) -> Self {
		self.verbose = val;
		self
	}

	pub fn launch(&self) -> Result<ExitStatus,LauncherError> {

		// run setup if needed
		if let Some(setup_command) = &self.setup_command {
			setup_command.launch(self.verbose)?;
		}

		// finally, launch the actual command
		self.command.launch(self.verbose)
	}
}


pub fn build_container_commands(
	home: &Path,
	config_path: &Path,
	container_path: &Path,
	binds: &Option<Vec<PathBuf>>,
	scratch: &Option<PathBuf>,
	container_exec: &Option<ContainerExec>,
	pyp_sources: &Option<PathBuf>,
	verbose: bool
) -> (Option<LaunchCommand>,LaunchCommand) {

	// load a module if needed
	let setup_command = match container_exec {

		Some(ContainerExec::Module { module_name, .. }) => {
			Some(
				LaunchCommand::new("module")
					.args(["load", module_name])
			)
		}

		_ => None
	};

	// get the container executable
	let mut container_command = LaunchCommand::new(match container_exec {

		Some(ContainerExec::Module { module_name, exec_name, .. }) => {
			match exec_name {

				// use an explicit executable name, if given
				Some(name) => name.as_str(),

				// othewise, use the module name by default
				None => module_name.as_str()
			}
		}

		Some(ContainerExec::Executable { name }) => {
			name.as_str()
		}

		None => &DEFAULT_CONTAINER_CMD
	});

	// start with the static container binds
	let mut all_binds = Vec::<ContainerBind>::new();
	for path in [".ssh", ".config", ".cache"] {
		all_binds.push(ContainerBind::Identity {
			path: home.join(path)
		});
	}

	// add the container binds from the config
	if let Some(scratch) = scratch {
		all_binds.push(ContainerBind::Identity {
			path: scratch.to_path_buf()
		});
	}
	if let Some(binds) = binds {
		for bind in binds {
			all_binds.push(ContainerBind::Identity {
				path: bind.to_path_buf()
			});
		}
	}
	if let Some(pyp_sources) = pyp_sources {
		all_binds.push(ContainerBind::Map {
			path_external: pyp_sources.to_path_buf(),
			path_internal: PathBuf::from(PYP_PATH_INTERNAL)
		});
	}

	// bind the config file itself, in case it's not in any other bind,
	// but make sure it's the last bind, or it could conflict with other binds
	all_binds.push(ContainerBind::Identity {
		path: config_path.to_path_buf()
	});

	// build all the arguments
	if !verbose {
		for arg in ["--quiet", "--silent"] {
			container_command.args.push(arg.to_string());
		}
	}
	for arg in ["exec", "--no-home"] {
		container_command.args.push(arg.to_string());
	}
	for bind in all_binds {
		container_command.args.push("--bind".to_string());
		container_command.args.push(bind.to_string());
	}
	container_command.args.push(container_path.to_string_lossy().to_string());

	(setup_command, container_command)
}


#[derive(PartialEq, Eq, Debug, Clone)]
pub struct LaunchCommand {
	pub program: String,
	pub args: Vec<String>
}

impl LaunchCommand {

	pub fn new<S:AsRef<str>>(program: S) -> Self {
		Self {
			program: program.as_ref().to_string(),
			args: vec![]
		}
	}

	pub fn args<I, S>(mut self, args: I) -> Self
	where
		I: IntoIterator<Item = S>,
		S: AsRef<str>,
	{
		self.args = args.into_iter()
			.map(|s| s.as_ref().to_string())
			.collect::<Vec<_>>();
		self
	}

	pub fn from(other: &LaunchCommand) -> Self {
		other.clone()
	}

	pub fn append(mut self, other: &LaunchCommand) -> Self {
		self.args.push(other.program.clone());
		for arg in &other.args {
			self.args.push(arg.clone());
		}
		self
	}

	pub fn launch(&self, verbose: bool) -> Result<ExitStatus,LauncherError> {

		if verbose {
			println!("{} {}", self.program, self.args.join(" "));
		}

		Ok(Command::new(&self.program)
			.args(&self.args)
			.spawn()
			.map_err(|e| LauncherError::LaunchFailure(e.to_string()))?
			.wait()
			.map_err(|e| LauncherError::LaunchFailure(e.to_string()))?
		)
	}
}


enum ContainerBind {

	/// map the same path from outside the container to inside the container
	Identity {
		path: PathBuf
	},

	/// map a path outside the container to a (possibly) different path inside the container
	Map {
		path_external: PathBuf,
		path_internal: PathBuf
	}
}

impl ToString for ContainerBind {
	fn to_string(&self) -> String {
		match self {

			ContainerBind::Identity { path } => format!("{}", path.to_string_lossy()),

			ContainerBind::Map { path_external, path_internal } => format!("{}:{}",
				path_external.to_string_lossy(),
				path_internal.to_string_lossy()
			)
		}
	}
}
