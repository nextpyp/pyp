
use std::process::{ExitCode, ExitStatus, Termination};

use crate::LauncherError;


/// a custom return type for main() that lets us forward the exit code of a launched process
pub enum Exit {
	Launched(ExitStatus),
	Error(LauncherError)
}

impl Termination for Exit {

	fn report(self) -> ExitCode {
		match self {

			Exit::Launched(status) => match status.code() {
				Some(code) => ExitCode::from(code as u8),
				None => ExitCode::FAILURE // terminated by signal, no exit code
			},

			Exit::Error(e) => {
				eprintln!("{}", e.to_string());
				// TODO: return a unique code for each error?
				ExitCode::FAILURE
			}
		}
	}
}
