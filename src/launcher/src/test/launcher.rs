
use std::collections::HashMap;
use std::path::PathBuf;

use crate::{ENVNAME_APPTAINER, ENVNAME_CONFIG, ENVNAME_HOME, ENVNAME_SINGULARITY, ENVNAME_VERBOSE, Launcher, LauncherError, PYP_BIN_RELPATH, PYP_PATH_INTERNAL};
use crate::launcher::LaunchCommand;
use crate::test::TempFolder;


/// the simplest kind of container launch
#[test]
fn launch_base() {

	let setup = TempSetup::new();
	setup.temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = "{}"
		"#,
		setup.container_path()
	));

	assert_eq!(setup.launcher(&["pyp"]), Ok(Launcher::new(
		None,
		LaunchCommand::new("singularity").test_args(&setup, vec![], vec![
			&format!("{}/{}/pyp", PYP_PATH_INTERNAL, PYP_BIN_RELPATH)
		])
	)));
}

#[test]
fn launch_pyp_args() {

	let setup = TempSetup::new();
	setup.temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = "{}"
		"#,
		setup.container_path()
	));

	assert_eq!(setup.launcher(&["pyp", "--do-things"]), Ok(Launcher::new(
		None,
		LaunchCommand::new("singularity").test_args(&setup, vec![], vec![
			&format!("{}/{}/pyp", PYP_PATH_INTERNAL, PYP_BIN_RELPATH) as &str,
			"--do-things"
		])
	)));
}

#[test]
fn launch_binds() {

	let setup = TempSetup::new();
	let bind_foo = setup.temp_folder.make_subfolder("foo");
	let bind_bar = setup.temp_folder.make_subfolder("bar");
	setup.temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = '{}'
			binds = ['{}', '{}']
		"#,
		setup.container_path(),
		bind_foo.to_string_lossy().to_string(),
		bind_bar.to_string_lossy().to_string()
	));

	assert_eq!(setup.launcher(&["pyp"]), Ok(Launcher::new(
		None,
		LaunchCommand::new("singularity").test_args(&setup, vec![
			"--bind", &bind_foo.to_string_lossy().to_string(),
			"--bind", &bind_bar.to_string_lossy().to_string(),
		], vec![
			&format!("{}/{}/pyp", PYP_PATH_INTERNAL, PYP_BIN_RELPATH)
		])
	)));
}

#[test]
fn launch_scratch() {

	let setup = TempSetup::new();
	let scratch_path = setup.temp_folder.make_subfolder("scratch");
	setup.temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = '{}'
			scratch = '{}'
		"#,
		setup.container_path(),
		scratch_path.to_string_lossy().to_string()
	));

	assert_eq!(setup.launcher(&["pyp"]), Ok(Launcher::new(
		None,
		LaunchCommand::new("singularity").test_args(&setup, vec![
			"--bind", &scratch_path.to_string_lossy().to_string()
		], vec![
			&format!("{}/{}/pyp", PYP_PATH_INTERNAL, PYP_BIN_RELPATH)
		])
	)));
}

#[test]
fn launch_scratch_envvar() {

	let setup = TempSetup::new();
	let scratch_path = setup.temp_folder.make_subfolder("scratch");
	setup.temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = '{}'
			scratch = '{}/$SOME_VAR'
		"#,
		setup.container_path(),
		scratch_path.to_string_lossy().to_string()
	));

	assert_eq!(setup.launcher(&["pyp"]), Ok(Launcher::new(
		None,
		LaunchCommand::new("singularity").test_args(&setup, vec![
			"--bind", &scratch_path.to_string_lossy().to_string()
		], vec![
			&format!("{}/{}/pyp", PYP_PATH_INTERNAL, PYP_BIN_RELPATH)
		])
	)));
}

#[test]
fn launch_scratch_not_envvar() {

	let setup = TempSetup::new();
	let scratch_path = setup.temp_folder.make_subfolder("scratch\\$notavar");
	setup.temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = '{}'
			scratch = '{}'
		"#,
		setup.container_path(),
		scratch_path.to_string_lossy().to_string()
	));

	assert_eq!(setup.launcher(&["pyp"]), Ok(Launcher::new(
		None,
		LaunchCommand::new("singularity").test_args(&setup, vec![
			"--bind", &scratch_path.to_string_lossy().to_string()
		], vec![
			&format!("{}/{}/pyp", PYP_PATH_INTERNAL, PYP_BIN_RELPATH)
		])
	)));
}

#[test]
fn launch_pyp_sources() {

	let setup = TempSetup::new();
	let pyp_path = setup.temp_folder.make_subfolder("pyp");
	setup.temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = '{}'
			sources = '{}'
		"#,
		setup.container_path(),
		pyp_path.to_string_lossy().to_string()
	));

	assert_eq!(setup.launcher(&["pyp"]), Ok(Launcher::new(
		None,
		LaunchCommand::new("singularity").test_args(&setup, vec![
			"--bind", &format!("{}:{}", &pyp_path.to_string_lossy(), PYP_PATH_INTERNAL)
		], vec![
			&format!("{}/{}/pyp", PYP_PATH_INTERNAL, PYP_BIN_RELPATH)
		])
	)));
}

#[test]
fn launch_inside_singularity() {

	let mut setup = TempSetup::new();
	setup.envvars.push((ENVNAME_SINGULARITY.to_string(), "sure".to_string()));

	assert_eq!(setup.launcher(&["pyp"]), Ok(Launcher::new(
		None,
		LaunchCommand::new(&format!("{}/{}/pyp", PYP_PATH_INTERNAL, PYP_BIN_RELPATH))
	)));
}

#[test]
fn launch_inside_singularity_pyp_args() {

	let mut setup = TempSetup::new();
	setup.envvars.push((ENVNAME_SINGULARITY.to_string(), "sure".to_string()));

	assert_eq!(setup.launcher(&["pyp", "--do-things"]), Ok(Launcher::new(
		None,
		LaunchCommand::new(&format!("{}/{}/pyp", PYP_PATH_INTERNAL, PYP_BIN_RELPATH))
			.args(vec!["--do-things"])
	)));
}

#[test]
fn launch_inside_apptainer() {

	let mut setup = TempSetup::new();
	setup.envvars.push((ENVNAME_APPTAINER.to_string(), "why not".to_string()));

	assert_eq!(setup.launcher(&["pyp"]), Ok(Launcher::new(
		None,
		LaunchCommand::new(&format!("{}/{}/pyp", PYP_PATH_INTERNAL, PYP_BIN_RELPATH))
	)));
}

#[test]
fn launch_container_exec_name() {

	let setup = TempSetup::new();
	setup.temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = '{}'
			containerExec = 'runme'
		"#,
		setup.container_path()
	));

	assert_eq!(setup.launcher(&["pyp"]), Ok(Launcher::new(
		None,
		LaunchCommand::new("runme").test_args(&setup, vec![], vec![
			&format!("{}/{}/pyp", PYP_PATH_INTERNAL, PYP_BIN_RELPATH)
		])
	)));
}

#[test]
fn launch_container_exec_module() {

	let setup = TempSetup::new();
	setup.temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = '{}'
			containerExec = {{ module = 'mod' }}
		"#,
		setup.container_path()
	));

	assert_eq!(setup.launcher(&["pyp"]), Ok(Launcher::new(
		Some(LaunchCommand::new("module").args(vec!["load", "mod"])),
		LaunchCommand::new("mod").test_args(&setup, vec![], vec![
			&format!("{}/{}/pyp", PYP_PATH_INTERNAL, PYP_BIN_RELPATH)
		])
	)));
}

#[test]
fn launch_container_exec_module_exec() {

	let setup = TempSetup::new();
	setup.temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = '{}'
			containerExec = {{ module = 'mod', exec = 'runme' }}
		"#,
		setup.container_path()
	));

	assert_eq!(setup.launcher(&["pyp"]), Ok(Launcher::new(
		Some(LaunchCommand::new("module").args(vec!["load", "mod"])),
		LaunchCommand::new("runme").test_args(&setup, vec![], vec![
			&format!("{}/{}/pyp", PYP_PATH_INTERNAL, PYP_BIN_RELPATH)
		])
	)));
}

#[test]
fn launch_verbose() {

	let mut setup = TempSetup::new();
	setup.envvars.push((ENVNAME_VERBOSE.to_string(), "1".to_string()));
	setup.temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = "{}"
		"#,
		setup.container_path()
	));

	assert_eq!(setup.launcher(&["pyp"]), Ok(Launcher::new(
		None,
		LaunchCommand::new("singularity").test_args_ex(&setup, true, vec![], vec![
			&format!("{}/{}/pyp", PYP_PATH_INTERNAL, PYP_BIN_RELPATH)
		])
	).verbose(true)));
}


/// Temporary folder and files that get deleted when the variable goes out of scope
struct TempSetup {
	temp_folder: TempFolder,
	container_path: PathBuf,
	pub envvars: Vec<(String,String)>
}

impl TempSetup {

	fn new() -> Self {

		let temp_folder = TempFolder::new();
		let container_path = temp_folder.write_file("container", vec![1, 2, 3]);

		// make the default env vars
		let envvars = vec![
			(ENVNAME_HOME.to_string(), "/home/user".to_string()),
			(ENVNAME_CONFIG.to_string(), temp_folder.config_path().to_string_lossy().to_string())
		];

		Self {
			temp_folder,
			container_path,
			envvars
		}
	}

	fn container_path(&self) -> String {
		self.container_path.to_string_lossy().to_string()
	}

	fn launcher(&self, args: &[&str]) -> Result<Launcher,LauncherError> {

		let args = args
			.iter()
			.map(|s| s.to_string())
			.collect();

		let envvars = HashMap::<String,String>::from_iter(self.envvars.clone().into_iter());

		Launcher::from(args, envvars)
	}
}


trait TestLaunchCommand {

	fn test_args<I, S>(self, setup: &TempSetup, args: I, pyp_args: I) -> Self
		where
			I: IntoIterator<Item = S>,
			S: AsRef<str>;

	fn test_args_ex<I, S>(self, setup: &TempSetup, verbose: bool, args: I, pyp_args: I) -> Self
		where
			I: IntoIterator<Item = S>,
			S: AsRef<str>;
}

impl TestLaunchCommand for LaunchCommand {

	fn test_args<I, S>(self, setup: &TempSetup, args: I, pyp_args: I) -> Self
		where
			I: IntoIterator<Item = S>,
			S: AsRef<str>,
	{
		self.test_args_ex(setup, false, args, pyp_args)
	}

	fn test_args_ex<I, S>(self, setup: &TempSetup, verbose: bool, args: I, pyp_args: I) -> Self
		where
			I: IntoIterator<Item = S>,
			S: AsRef<str>,
	{
		let mut all_args = Vec::<String>::new();

		// start with the static args

		if !verbose {
			for arg in vec!["--quiet", "--silent"] {
				all_args.push(arg.to_string());
			}
		}

		for arg in vec![
			"exec", "--no-home",
			"--bind", "/home/user/.ssh",
			"--bind", "/home/user/.config",
			"--bind", "/home/user/.cache",
		] {
			all_args.push(arg.to_string());
		}

		// add the test-specific args
		for arg in args {
			all_args.push(arg.as_ref().to_string());
		}

		// add the config file bind last
		all_args.push("--bind".to_string());
		all_args.push(setup.temp_folder.config_path().to_string_lossy().to_string());

		// then the container
		all_args.push(setup.container_path.to_string_lossy().to_string());

		// add the pyp args
		for arg in pyp_args {
			all_args.push(arg.as_ref().to_string());
		}

		self.args(all_args)
	}
}
