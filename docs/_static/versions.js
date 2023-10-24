
async function main() {

	// find the bits of the DOM that correspond to the left sidebar
	const navElem = document.querySelector('#site-navigation');
	const brandElem = navElem.querySelector('div.navbar-brand-box');

	// make an area we can write to and put in the left sidebar
	const elem = document.createElement('div');
	elem.setAttribute('class', 'versions');
	brandElem.after(elem);

	const labelElem = document.createElement('span');
	labelElem.append(document.createTextNode('Version:'));
	labelElem.setAttribute('for', 'versions-dropdown');
	labelElem.setAttribute('class', 'label');
	elem.append(labelElem);

	// get the version list set by the versions-info script
	const versions = window.nextpypVersions || [];

	// detect the current version, if any
	const pathParts = window.location.pathname.split('/');
	const pathIndex = pathParts
		.findIndex(pathPart => versions.includes(pathPart))
	const currentVersion = pathIndex === -1 ? undefined : pathParts[pathIndex];

	// populate the dropdown with versions
	if (currentVersion === undefined) {

		// if we can't detect the current version, that means we can't navigate to other versions either
		// so just show a generic error message
		elem.append(document.createTextNode("Can't detect version!"));

	} else {

		// make a dropdown with all the versions
		const dropdownElem = document.createElement('select');
		dropdownElem.setAttribute('id', 'versions-dropdown');
		elem.append(dropdownElem);

		for (const version of versions) {
			const optionElem = document.createElement('option');
			optionElem.append(document.createTextNode(version));
			optionElem.setAttribute('value', version);
			if (version === currentVersion) {
				optionElem.setAttribute('selected', '');
			}
			dropdownElem.append(optionElem);
		}

		// wire up events
		dropdownElem.addEventListener("change", () => {

			// get the new version, if it's different from the current version
			const newVersion = dropdownElem.options[dropdownElem.options.selectedIndex].value;
			if (newVersion === currentVersion) {
				return;
			}

			// rebuild the current URL, but with the new version
			const newPathParts = [... pathParts];
			newPathParts[pathIndex] = newVersion;
			const newPath = newPathParts.join('/');

			window.location = newPath;
		});
	}
}

document.addEventListener("DOMContentLoaded", main);
