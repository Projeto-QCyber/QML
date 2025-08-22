import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request

# Minimal, no-args script: install Ollama if missing, pull model, cleanup port 11434, then run server
HOST = "127.0.0.1:11434"
MODEL = "qwen2.5:3b"


def port_in_use(host_port: str) -> bool:
    host, port_str = host_port.split(":", 1)
    port = int(port_str)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def kill_listeners_on_port(port: int) -> None:
    # Try multiple strategies, ignore failures
    subprocess.run(["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["bash", "-lc", f"lsof -ti :{port} 2>/dev/null | xargs -r kill -9"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Parse ss output to extract pid= and kill
    ss_cmd = "ss -ltnp 2>/dev/null | awk '/:%d / {print $NF}' | sed -n 's/.*pid=\\([0-9]*\\).*/\\1/p'" % port
    subprocess.run(["bash", "-lc", f"for p in $({ss_cmd}); do kill -9 $p 2>/dev/null || true; done"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_systemd_service_if_any() -> None:
    # Best-effort stop/disable system service; skip if no permissions
    subprocess.run(["systemctl", "is-active", "--quiet", "ollama"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["sudo", "-n", "systemctl", "disable", "--now", "ollama"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ensure_installed() -> None:
    if shutil.which("ollama") is None:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve("https://ollama.com/install.sh", tmp.name)
        os.chmod(tmp.name, 0o755)
        subprocess.run(["/bin/bash", tmp.name], check=True)


def main() -> None:
    ensure_installed()
    os.environ["OLLAMA_HOST"] = HOST

    # Pull model (best-effort if already present)
    subprocess.run(["ollama", "pull", MODEL])

    # Cleanup anything bound to the port to avoid conflicts
    if port_in_use(HOST):
        stop_systemd_service_if_any()
        for _ in range(3):
            kill_listeners_on_port(11434)
            time.sleep(1)
            if not port_in_use(HOST):
                break

    # Start server only if port is free; otherwise assume an existing server is fine
    if not port_in_use(HOST):
        subprocess.run(["ollama", "serve"])  # foreground
    else:
        print(f"Ollama already running on {HOST}. Not starting another instance.")


if __name__ == "__main__":
    main()


