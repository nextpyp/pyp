use std::collections::HashMap;
use std::fs::Permissions;
use std::os::unix::fs::PermissionsExt;

use crate::{ENVNAME_CONFIG, ENVNAME_HOME, Launcher, LauncherError};

use super::TempFolder;


#[test]
fn no_caller_path() {

	let args = [];
	let envvars = [];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::NoCallerPath));
}

#[test]
fn empty_caller_path() {

	let args = [""];
	let envvars = [];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::EmptyCallerPath));
}

#[test]
fn no_home() {

	let args = ["pyp"];
	let envvars = [];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::NoEnvVar(ENVNAME_HOME.to_string())));
}

#[test]
fn no_config() {

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user")
	];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::NoConfig {
		envname: ENVNAME_CONFIG.to_string(),
		path: "/home/user/.pyp/config.toml".to_string()
	}));
}

#[test]
#[cfg(target_os = "linux")]
fn config_read_failed() {

	let temp_folder = TempFolder::new();
	let config_path = temp_folder.write_config("");

	// remove our read privileges so the file read fails
	std::fs::set_permissions(&config_path, Permissions::from_mode(0))
		.expect("can't set file mode");

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	let result = launcher(&args, &envvars);
	println!("result: {:?}", result);
	assert!(matches!(result, Err(LauncherError::ConfigReadFailed(..))));
}

#[test]
fn config_parse_failed() {

	let temp_folder = TempFolder::new();
	let config_path = temp_folder.write_config(r#"
		this isn't valid TOML
	"#);

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	let result = launcher(&args, &envvars);
	println!("result: {:?}", result);
	assert!(matches!(result, Err(LauncherError::ConfigParseFailed(..))));
}

#[test]
fn config_no_pyp() {

	let temp_folder = TempFolder::new();
	let config_path = temp_folder.write_config(r#"
		hello = "world"
	"#);

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::ConfigSectionNotFound("pyp".to_string())));
}

#[test]
fn config_pyp_not_table() {

	let temp_folder = TempFolder::new();
	let config_path = temp_folder.write_config(r#"
		pyp = "not a table"
	"#);

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::ConfigSectionNotTable("pyp".to_string())));
}

#[test]
fn config_container_not_found() {

	let temp_folder = TempFolder::new();
	let config_path = temp_folder.write_config(r#"
		[pyp]
	"#);

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::ConfigValueNotFound {
		section: "pyp".to_string(),
		key: "container".to_string()
	}));
}

#[test]
fn config_container_not_string() {

	let temp_folder = TempFolder::new();
	let config_path = temp_folder.write_config(r#"
		[pyp]
		container = 5
	"#);

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::ConfigValueNotString {
		section: "pyp".to_string(),
		key: "container".to_string()
	}));
}

#[test]
fn config_container_file_not_found() {

	let temp_folder = TempFolder::new();
	let config_path = temp_folder.write_config(r#"
		[pyp]
		container = "not a file"
	"#);

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::ConfigFileNotFound {
		section: "pyp".to_string(),
		key: "container".to_string(),
		path: "not a file".to_string()
	}));
}

#[test]
fn config_binds_not_array() {

	let temp_folder = TempFolder::new();
	let container_path = temp_folder.write_file("container", vec![1, 2, 3]);
	let config_path = temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = "{}"
			binds = ""
		"#,
		container_path.to_str().unwrap()
	));

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::ConfigValueNotArray {
		section: "pyp".to_string(),
		key: "binds".to_string()
	}));
}

#[test]
fn config_binds_not_strings() {

	let temp_folder = TempFolder::new();
	let container_path = temp_folder.write_file("container", vec![1, 2, 3]);
	let config_path = temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = "{}"
			binds = ["", 2, 3]
		"#,
		container_path.to_str().unwrap()
	));

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::ConfigValuesNotStrings {
		section: "pyp".to_string(),
		key: "binds".to_string()
	}));
}

#[test]
fn config_binds_not_folders() {

	let temp_folder = TempFolder::new();
	let container_path = temp_folder.write_file("container", vec![1, 2, 3]);
	let config_path = temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = "{}"
			binds = ["foo"]
		"#,
		container_path.to_str().unwrap()
	));

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::ConfigValuesNotFolders {
		section: "pyp".to_string(),
		key: "binds".to_string(),
		path: "foo".to_string()
	}));
}

#[test]
fn config_scratch_not_string() {

	let temp_folder = TempFolder::new();
	let container_path = temp_folder.write_file("container", vec![1, 2, 3]);
	let config_path = temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = "{}"
			binds = ["./"]
			scratch = 5
		"#,
		container_path.to_str().unwrap()
	));

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::ConfigValueNotString {
		section: "pyp".to_string(),
		key: "scratch".to_string()
	}));
}

#[test]
fn config_scratch_not_folder() {

	let temp_folder = TempFolder::new();
	let container_path = temp_folder.write_file("container", vec![1, 2, 3]);
	let config_path = temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = "{}"
			binds = ["./"]
			scratch = "foo"
		"#,
		container_path.to_str().unwrap()
	));

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	// this is no longer an error
	assert!(matches!(launcher(&args, &envvars), Ok(..)));
}

#[test]
fn config_container_exec_not_type() {

	let temp_folder = TempFolder::new();
	let container_path = temp_folder.write_file("container", vec![1, 2, 3]);
	let config_path = temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = "{}"
			binds = ["./"]
			scratch = "./"
			containerExec = 5
		"#,
		container_path.to_str().unwrap()
	));

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::ConfigValueNotContainerExec {
		section: "pyp".to_string(),
		key: "containerExec".to_string()
	}));
}

#[test]
fn config_container_exec_module_not_found() {

	let temp_folder = TempFolder::new();
	let container_path = temp_folder.write_file("container", vec![1, 2, 3]);
	let config_path = temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = "{}"
			binds = ["./"]
			scratch = "./"
			containerExec = {{ foo = "bar" }}
		"#,
		container_path.to_str().unwrap()
	));

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::ConfigValueNotFound {
		section: "pyp.containerExec".to_string(),
		key: "module".to_string()
	}));
}

#[test]
fn config_container_exec_module_not_string() {

	let temp_folder = TempFolder::new();
	let container_path = temp_folder.write_file("container", vec![1, 2, 3]);
	let config_path = temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = "{}"
			binds = ["./"]
			scratch = "./"
			containerExec = {{ module = 5 }}
		"#,
		container_path.to_str().unwrap()
	));

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::ConfigValueNotString {
		section: "pyp.containerExec".to_string(),
		key: "module".to_string()
	}));
}

#[test]
fn config_container_exec_exec_not_string() {

	let temp_folder = TempFolder::new();
	let container_path = temp_folder.write_file("container", vec![1, 2, 3]);
	let config_path = temp_folder.write_config(&format!(
		r#"
			[pyp]
			container = "{}"
			binds = ["./"]
			scratch = "./"
			containerExec = {{ module = "foo", exec = 5 }}
		"#,
		container_path.to_str().unwrap()
	));

	let args = ["pyp"];
	let envvars = [
		(ENVNAME_HOME, "/home/user"),
		(ENVNAME_CONFIG, config_path.to_str().unwrap())
	];

	assert_eq!(launcher(&args, &envvars), Err(LauncherError::ConfigValueNotString {
		section: "pyp.containerExec".to_string(),
		key: "exec".to_string()
	}));
}


fn launcher(args: &[&str], envvars: &[(&str,&str)]) -> Result<Launcher,LauncherError> {

	let args = args
		.iter()
		.map(|s| s.to_string())
		.collect();

	let envvars = HashMap::<String,String>::from_iter(
		envvars
			.iter()
			.map(|(key,val)| (key.to_string(), val.to_string()))
	);

	Launcher::from(args, envvars)
}
