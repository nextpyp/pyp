
use thiserror::Error;


#[derive(Error, Debug, PartialEq, Eq)]
pub enum LauncherError {

	#[error("Unable to determine PYP command: no caller path.")]
	NoCallerPath,

	#[error("Unable to determine PYP command: caller path is empty.")]
	EmptyCallerPath,

	#[error("No ${0} environment variable set.")]
	NoEnvVar(String),

	#[error("Please create a PYP configuration file in {path} or set ${envname} to the location of your config.toml.")]
	NoConfig {
		path: String,
		envname: String
	},

	#[error("Failed to read config file. {0}")]
	ConfigReadFailed(String),

	#[error("Failed to parse config file. {0}")]
	ConfigParseFailed(#[from] toml::de::Error),

	#[error("No [{0}] section was found in the config file.")]
	ConfigSectionNotFound(String),

	#[error("The [{0}] section in config file must be a table.")]
	ConfigSectionNotTable(String),

	#[error("No `{key}` value was set in the [{section}] section in the config file.")]
	ConfigValueNotFound {
		section: String,
		key: String
	},

	#[error("`{key}` in the [{section}] section in the config file must be a string.")]
	ConfigValueNotString {
		section: String,
		key: String
	},

	#[error("`{key}` in the [{section}] section in the config file must be an array.")]
	ConfigValueNotArray {
		section: String,
		key: String
	},

	#[error("Every entry of `{key}` in the [{section}] section of the config file must be a string.")]
	ConfigValuesNotStrings {
		section: String,
		key: String
	},

	#[error("`{key}` in the [{section}] section of the config file must refer to a file that exists. {path} was not found or is not a file.")]
	ConfigFileNotFound {
		section: String,
		key: String,
		path: String
	},

	#[error("`{key}` in the [{section}] section of the config file must refer to a folder that exists. {path} was not found or is not a folder.")]
	ConfigFolderNotFound {
		section: String,
		key: String,
		path: String
	},

	#[error("Every entry of `{key}` in the [{section}] section of the config file must refer to a folder that exists. {path} was not found or is not a folder.")]
	ConfigValuesNotFolders {
		section: String,
		key: String,
		path: String
	},

	#[error("`{key}` in the [{section}] section in the config file must either a string, or a table with key `module` holding a string.")]
	ConfigValueNotContainerExec {
		section: String,
		key: String
	},

	#[error("Failed to launch. {0}")]
	LaunchFailure(String)
}
