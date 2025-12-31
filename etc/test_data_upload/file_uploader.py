import argparse
import toolviper
import json
import shutil
import pathlib
import subprocess

parser = argparse.ArgumentParser(
    description=f"Upload test files to R2 cloudfare (Tokens are fetched from user Env)",
    formatter_class=argparse.RawTextHelpFormatter,
)

parser.add_argument("file", type=str, help="File to be uploaded")

parser.add_argument(
    "-p",
    "--project",
    type=str,
    default="astrohack",
    help="To which project folder to upload in cloudfare",
)

parser.add_argument(
    "-t",
    "--telescope",
    type=str,
    default="VLA",
    help="Telescope with which data was taken",
)

parser.add_argument(
    "-m",
    "--observation-mode",
    type=str,
    default="Holography",
    help="Observation mode",
)

parser.add_argument(
    "-d",
    "--data-type",
    type=str,
    default="CASA MS V2",
    help="Data type",
)

parser.add_argument(
    "-u",
    "--update-version",
    action="store_true",
    default=False,
    help="Update manifest version",
)

parser.add_argument(
    "-a",
    "--manifest-path",
    type=str,
    default=None,
    help="Path to manifest file, if None fetches it from toolviper",
)


def execute_shell_command(command):
    results = subprocess.run(command, capture_output=True, text=True)
    return results.stdout


def download_manifest(args, manifest_filename=".manifest.json"):
    if args.manifest_path is None:
        downloaded_manifest = (
            toolviper.utils.data.__file__.split("__")[0]
            + ".cloudflare/file.download.json"
        )
        toolviper.utils.data.update()
        shutil.copyfile(downloaded_manifest, manifest_filename)
    else:
        manifest_filename = args.manifest_path

    with open(manifest_filename, "r") as manifest:
        manifest = json.load(manifest)
    return manifest_filename, manifest


def prepare_data_for_upload(args):
    is_dir = pathlib.Path(args.file).is_dir()
    if is_dir:
        meta_name = args.file + ".zip"
        execute_shell_command(["zip", "-r", meta_name, args.file])
    else:
        meta_name = args.file

    sha256sum = execute_shell_command(["sha256sum", meta_name]).split()[0]
    file_path = pathlib.Path(meta_name)
    file_size_bytes = file_path.stat().st_size

    file_properties = {
        "file": meta_name,
        "path": args.project,
        "dtype": args.data_type,
        "telescope": args.telescope,
        "size": f"{file_size_bytes:d}",
        "mode": args.observation_mode,
        "hash": sha256sum,
    }
    return file_properties


def update_manifest_version(manifest):
    current = manifest["version"]
    rev, major, minor = current.split(".")
    minor = f"{int(minor)+1}"
    manifest["version"] = f"{rev}.{major}.{minor}"
    print(f"Updating manifest version from {current} to {manifest['version']}")


def add_data_to_manifest(args, manifest, file_properties, manifest_filename):
    if args.update_version:
        update_manifest_version(manifest)

    manifest["metadata"][args.file] = file_properties

    with open(manifest_filename, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=4)


def main():
    args = parser.parse_args()

    manifest_filename, manifest = download_manifest(args)

    file_properties = prepare_data_for_upload(args)

    add_data_to_manifest(
        args,
        manifest,
        file_properties,
        manifest_filename,
    )

    return


if __name__ == "__main__":
    main()
