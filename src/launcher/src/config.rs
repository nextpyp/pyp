
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use toml::{Table, Value};

use crate::LauncherError;


pub struct ConfigReader {
	toml: Table
}

impl ConfigReader {

	pub fn read(path: &Path) -> Result<Self,LauncherError> {
		Ok(Self {
			toml: std::fs::read_to_string(&path)
				.map_err(|e| LauncherError::ConfigReadFailed(e.to_string()))?
				.parse::<Table>()
				.map_err(|e| LauncherError::ConfigParseFailed(e))?
		})
	}

	pub fn get_section(&self, name: &'static str) -> Result<ConfigSectionReader,LauncherError> {
		Ok(ConfigSectionReader {
			name,
			toml: self.toml
				.get("pyp")
				.ok_or_else(|| LauncherError::ConfigSectionNotFound(name.to_string()))?
				.as_table()
				.ok_or_else(|| LauncherError::ConfigSectionNotTable(name.to_string()))?
		})
	}
}


pub struct ConfigSectionReader<'a> {
	name: &'static str,
	toml: &'a Table
}

impl<'a> ConfigSectionReader<'a> {

	pub fn get_optional_str(&self, key: &'static str) -> Result<Option<&str>,LauncherError> {
		Ok(match self.toml.get(key) {
			None => None,
			Some(val) => Some(val
				.as_str()
				.ok_or_else(|| LauncherError::ConfigValueNotString {
					section: self.name.to_string(),
					key: key.to_string()
				})?
			)
		})
	}

	pub fn _get_str(&self, key: &'static str) -> Result<&str,LauncherError> {
		Ok(self.get_optional_str(key)?
			.ok_or_else(|| LauncherError::ConfigValueNotFound {
				section: self.name.to_string(),
				key: key.to_string()
			})?
		)
	}

	pub fn get_optional_strs(&self, key: &'static str) -> Result<Option<Vec<&str>>,LauncherError> {
		Ok(match self.toml.get(key) {
			None => None,
			Some(val) => Some(val
				.as_array()
				.ok_or_else(|| LauncherError::ConfigValueNotArray {
					section: self.name.to_string(),
					key: key.to_string()
				})?
				.iter()
				.map(|val| {
					val
						.as_str()
						.ok_or_else(|| LauncherError::ConfigValuesNotStrings {
							section: self.name.to_string(),
							key: key.to_string()
						})
				})
				.collect::<Result<Vec<&str>,LauncherError>>()? // NOTE: collect() here will push the first Error outside the Vec
			)
		})
	}

	pub fn _get_strs(&self, key: &'static str) -> Result<Vec<&str>,LauncherError> {
		Ok(self.get_optional_strs(key)?
			.ok_or_else(|| LauncherError::ConfigValueNotFound {
				section: self.name.to_string(),
				key: key.to_string()
			})?
		)
	}

	pub fn get_optional_file(&self, key: &'static str) -> Result<Option<&Path>,LauncherError> {
		match self.get_optional_str(key)? {
			None => Ok(None),
			Some(val) => {
				let path = Path::new(val);
				if path.is_file() {
					Ok(Some(path))
				} else {
					Err(LauncherError::ConfigFileNotFound {
						section: self.name.to_string(),
						key: key.to_string(),
						path: path.to_string_lossy().to_string()
					})
				}
			}
		}
	}

	pub fn get_file(&self, key: &'static str) -> Result<&Path,LauncherError> {
		Ok(self.get_optional_file(key)?
			.ok_or_else(|| LauncherError::ConfigValueNotFound {
				section: self.name.to_string(),
				key: key.to_string()
			})?
		)
	}

	pub fn get_optional_folder(&self, key: &'static str) -> Result<Option<PathBuf>,LauncherError> {
		match self.get_optional_str(key)? {
			None => Ok(None),
			Some(val) => {
				let path = truncate_path_vars(Path::new(val));
				if path.is_dir() {
					Ok(Some(path))
				} else {
					Err(LauncherError::ConfigFolderNotFound {
						section: self.name.to_string(),
						key: key.to_string(),
						path: path.to_string_lossy().to_string()
					})
				}
			}
		}
	}

	pub fn _get_folder(&self, key: &'static str) -> Result<PathBuf,LauncherError> {
		Ok(self.get_optional_folder(key)?
			.ok_or_else(|| LauncherError::ConfigValueNotFound {
				section: self.name.to_string(),
				key: key.to_string()
			})?
		)
	}

	pub fn get_optional_folders(&self, key: &'static str) -> Result<Option<Vec<PathBuf>>,LauncherError> {
		Ok(match self.get_optional_strs(key)? {
			None => None,
			Some(val) => Some(val.iter()
				.map(|s| {
					let path = truncate_path_vars(Path::new(s));
					if path.is_dir() {
						Ok(path)
					} else {
						Err(LauncherError::ConfigValuesNotFolders {
							section: self.name.to_string(),
							key: key.to_string(),
							path: path.to_string_lossy().to_string()
						})
					}
				})
				.collect::<Result<Vec<PathBuf>,LauncherError>>()? // NOTE: collect() here will push the first Error outside the Vec
			)
		})
	}

	pub fn _get_folders(&self, key: &'static str) -> Result<Vec<PathBuf>,LauncherError> {
		Ok(self.get_optional_folders(key)?
			.ok_or_else(|| LauncherError::ConfigValueNotFound {
				section: self.name.to_string(),
				key: key.to_string()
			})?
		)
	}

	pub fn get_container_exec(&self, key: &'static str, key_module: &'static str, key_exec: &'static str) -> Result<Option<ContainerExec>,LauncherError> {

		let val = match self.toml.get(key) {
			Some(v) => v,
			None => return Ok(None)
		};

		match val {

			Value::String(name) => {
				Ok(Some(ContainerExec::Executable {
					name: name.clone()
				}))
			}

			Value::Table(table) => {

				// read the module name from the table
				let module_name = table.get(key_module)
					.ok_or_else(|| LauncherError::ConfigValueNotFound {
						section: format!("{}.{}", self.name, key),
						key: key_module.to_string()
					})?
					.as_str()
					.ok_or_else(|| LauncherError::ConfigValueNotString {
						section: format!("{}.{}", self.name, key),
						key: key_module.to_string()
					})?
					.to_string();

				// read the executable name from the table, if any
				let exec_name = match table.get(key_exec) {

					Some(val) => Some(
						val.as_str()
							.ok_or_else(|| LauncherError::ConfigValueNotString {
								section: format!("{}.{}", self.name, key),
								key: key_exec.to_string()
							})?
							.to_string()
					),

					None => None
				};

				Ok(Some(ContainerExec::Module {
					module_name,
					exec_name
				}))
			}

			_ => Err(LauncherError::ConfigValueNotContainerExec {
				section: self.name.to_string(),
				key: key.to_string()
			})
		}
	}
}

/// truncates the path at the first component containing a shell variable,
/// returning the non-shell-variabled parent subpath
fn truncate_path_vars(path: &Path) -> PathBuf {
	path.components()
		.take_while(|c| !has_shell_var(c.as_os_str()))
		.collect::<PathBuf>()
}

fn has_shell_var(s: &OsStr) -> bool {

	// need to convert to UTF-8 before we can do any processing on the string
	let s = s.to_string_lossy()
		.chars()
		.collect::<Vec<_>>();

	// look for a $ that is not preceded by a \
	for i in 0 .. s.len() {
		if s[i] == '$' && (i == 0 || s[i-1] != '\\') {
			return true;
		}
	}
	return false;
}


#[derive(Debug)]
pub enum ContainerExec {

	Module {
		module_name: String,
		exec_name: Option<String>
	},

	Executable {
		/// the name (or absolute path) to the executable
		name: String
	}
}
