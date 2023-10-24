
mod errors;
mod launcher;


use std::env;
use std::path::PathBuf;


/// A temporary directory that gets deleted when the value goes out of scope
pub struct TempFolder {
	path: PathBuf
}

impl TempFolder {

	pub fn new() -> Self {
		let path = env::temp_dir().join("pyp-launcher");
		std::fs::create_dir_all(&path)
			.expect(&format!("can't create temp folder: {:?}", path));
		Self {
			path
		}
	}

	pub fn config_path(&self) -> PathBuf {
		self.path.join("config.toml")
	}

	pub fn write_file<C:AsRef<[u8]>>(&self, filename: &str, content: C) -> PathBuf {
		let path = self.path.join(filename);
		std::fs::write(&path, content)
			.expect("can't write file");
		path
	}

	pub fn write_config(&self, toml: &str) -> PathBuf {
		let path = self.config_path();
		std::fs::write(&path, toml)
			.expect("can't write config file");
		path
	}

	pub fn make_subfolder(&self, name: &str) -> PathBuf {
		let path = self.path.join(name);
		std::fs::create_dir(&path)
			.expect("can't create sub-folder");

		path
	}
}

impl Drop for TempFolder {
	fn drop(&mut self) {
		if let Err(_) = std::fs::remove_dir_all(&self.path) {
			println!("WARNING: failed to remove temp folder: {:?}", self.path)
		}
	}
}
