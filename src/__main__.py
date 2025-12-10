from __future__ import annotations
import os, subprocess, signal, psutil
from pathlib import Path
import typer
import torch
import multiprocessing
from dotenv import load_dotenv

load_dotenv()


def _num_gpus():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1


def _num_cpus():
    return multiprocessing.cpu_count()


def create_procfile(procfile: Path, mode="dev"):
    # Ray manages workers internally, so we only need the API server
    if mode == "dev":
        start = "api: uvicorn src.api.main:app --host 127.0.0.1 --port 8765 --reload\n"
    elif mode == "prod":
        start = "api: gunicorn src.api.main:app --config gunicorn.conf.py\n"
    with open(procfile, "w") as f:
        f.write(start)


def create_envfile(envfile: Path, mode="dev"):
    if mode == "dev":
        with open(envfile, "w") as f:
            f.write(f"NUM_GPUS={_num_gpus()}")
    elif mode == "prod":
        with open(envfile, "w") as f:
            f.write(f"NUM_GPUS={_num_gpus()}")


app = typer.Typer(help="Apex command line")


def _run(cmd: list[str], *, cwd: Path | None = None, daemon: bool = False):
    if daemon:
        # Run in background as daemon
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        print(f"Started daemon process with PID: {proc.pid}")
        return proc.pid
    else:
        # Run in foreground
        proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
        raise SystemExit(proc.returncode)


@app.command()
def start(
    procfile: Path = typer.Option(
        Path("Procfile"), "--procfile", "-f", help="Path to Procfile"
    ),
    envfile: Path | None = typer.Option(None, "--env", "-e", help=".env file to load"),
    cwd: Path = typer.Option(
        Path("."), "--cwd", help="Working directory where processes run"
    ),
    daemon: bool = typer.Option(
        False, "--daemon", "-d", help="Run as daemon in background"
    ),
):
    """
    Start FastAPI + Celery (and anything else in your Procfile) via Honcho.
    Equivalent to: honcho start -f Procfile [-e .env]
    """
    create_procfile(
        cwd / "Procfile", "dev" if procfile.name == "Procfile.dev" else "prod"
    )
    args = ["python3", "-m", "honcho", "start", "-f", str(procfile)]
    if envfile:
        args += ["-e", str(envfile)]
    _run(args, cwd=cwd, daemon=daemon)


def _find_apex_processes():
    """Find running processes related to apex engine (uvicorn, ray, honcho)"""
    processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            # Look for our specific processes
            if any(
                pattern in cmdline
                for pattern in [
                    "uvicorn src.api.main:app",
                    "ray::",
                    "honcho start",
                    "python3 -m honcho start",
                ]
            ):
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return processes


@app.command()
def stop(
    force: bool = typer.Option(
        False, "--force", "-f", help="Force kill processes (SIGKILL instead of SIGTERM)"
    ),
):
    """
    Stop running Apex Engine processes (uvicorn, ray, honcho).
    Finds and terminates any existing server processes.
    """
    processes = _find_apex_processes()

    if not processes:
        print("No running Apex Engine processes found.")
        return

    print(f"Found {len(processes)} running process(es):")
    for proc in processes:
        try:
            cmdline = " ".join(proc.cmdline())
            print(f"  PID {proc.pid}: {cmdline}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    signal_type = signal.SIGKILL if force else signal.SIGTERM
    signal_name = "SIGKILL" if force else "SIGTERM"

    killed_count = 0
    for proc in processes:
        try:
            print(f"Sending {signal_name} to PID {proc.pid}...")
            proc.send_signal(signal_type)
            killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"Could not kill PID {proc.pid}: {e}")

    if killed_count > 0:
        print(f"Successfully sent {signal_name} to {killed_count} process(es).")
        if not force:
            print(
                "Processes should shutdown gracefully. Use --force if they don't stop."
            )
    else:
        print("No processes were terminated.")


# Optional sugar: `apex dev` alias
@app.command()
def dev(cwd: Path = Path(".")):
    """Convenience alias for apex start -f Procfile.dev"""
    create_procfile(cwd / "Procfile.dev", "dev")
    _run(["python", "-m", "honcho", "start", "-f", "Procfile.dev"], cwd=cwd)
