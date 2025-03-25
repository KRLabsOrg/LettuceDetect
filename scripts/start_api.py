#!/usr/bin/env python3

import argparse
import os
import pathlib
import subprocess


def _argparse() -> dict:
    parser = argparse.ArgumentParser(description="Start lettucedetect Web API.")
    parser.add_argument(
        "mode",
        help=(
            'Choose "dev" for development or "prod" for production environments. The serve script '
            'uses "fastapi dev" for "dev" or "fastapi run" for "prod" to start the web server. '
            'Additionally when choosing the "dev" mode, python modules can be directly imported '
            "from the repositroy without installing the package."
        ),
        choices=["prod", "dev"],
    )
    parser.add_argument(
        "--model",
        help='Path or huggingface URL to the model. The default value is "KRLabsOrg/lettucedect-base-modernbert-en-v1".',
        default="KRLabsOrg/lettucedect-base-modernbert-en-v1",
    )
    parser.add_argument(
        "--method",
        help='Hallucination detection method. The default value is "transformer".',
        choices=["transformer"],
        default="transformer",
    )
    return parser.parse_args()


def _run_fastapi(args: dict) -> None:
    scripts_folder = pathlib.Path(__file__).parent.resolve()
    repo_folder = scripts_folder.parent
    api_folder = repo_folder / "lettucedetect_api"
    env = os.environ.copy()
    env["LETTUCEDETECT_MODEL"] = args.model
    env["LETTUCEDETECT_METHOD"] = args.method
    if args.mode == "dev":
        # Needed for fastapi to be able to import directly from the repository.
        env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + str(repo_folder)
        fastapi_cmd = ["fastapi", "dev", api_folder / "server.py"]
    else:
        fastapi_cmd = ["fastapi", "run", api_folder / "server.py"]
    try:
        # Ignore S603: Validate input to run method. False positive.
        subprocess.run(fastapi_cmd, env=env, cwd=repo_folder)  # noqa: S603
    except KeyboardInterrupt:
        pass


def main() -> None:
    """Entry point for script."""
    args = _argparse()
    _run_fastapi(args)


if __name__ == "__main__":
    main()
