import os
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "smoke_primus_docker.sh"


class SmokePrimusDockerTests(unittest.TestCase):
    def run_smoke(self, env):
        merged_env = {
            **os.environ,
            "AGENTS_AGENT_MODE": "1",
            "AGENTS_ALLOW_INSTALL": "1",
            "DOCKER_GPU_ARGS": "",
            **env,
        }
        return subprocess.run(
            ["/bin/bash", str(SCRIPT)],
            cwd=ROOT,
            env=merged_env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_reports_missing_docker_cli_with_host_fix_hint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_smoke({"PATH": f"{tmpdir}:/usr/bin:/bin"})

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Docker CLI is not installed", result.stderr)
        self.assertIn("Host fix candidates", result.stderr)
        self.assertIn("PR899_DOCKER_HOST_FIX.md", result.stderr)

    def test_reports_docker_daemon_connectivity_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_docker = Path(tmpdir) / "docker"
            fake_docker.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    if [[ "$1" == "info" ]]; then
                      echo "failed to connect to the docker API at unix:///var/run/docker.sock" >&2
                      exit 1
                    fi
                    exit 99
                    """
                ),
                encoding="utf-8",
            )
            fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IEXEC)

            result = self.run_smoke({"PATH": f"{tmpdir}:/usr/bin:/bin", "DOCKER_HOST": ""})

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Docker CLI is available, but it cannot reach a Docker daemon", result.stderr)
        self.assertIn("failed to connect to the docker API", result.stderr)
        self.assertIn("sudo apt-get install -y docker.io", result.stderr)

    def test_reports_devpts_failure_with_diagnostics_and_host_fix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_docker = Path(tmpdir) / "docker"
            fake_docker.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    if [[ "$1" == "info" && "$2" == "--format" ]]; then
                      case "$3" in
                        "{{.Driver}}") echo "vfs" ;;
                        "{{.CgroupDriver}}") echo "cgroupfs" ;;
                        "{{.CgroupVersion}}") echo "2" ;;
                        "{{json .SecurityOptions}}") echo '["name=seccomp"]' ;;
                        *) echo "unknown" ;;
                      esac
                      exit 0
                    fi
                    if [[ "$1" == "info" ]]; then
                      exit 0
                    fi
                    if [[ "$1" == "version" ]]; then
                      echo "29.5.2"
                      exit 0
                    fi
                    if [[ "$1" == "run" ]]; then
                      echo 'docker: Error response from daemon: OCI runtime create failed: error mounting "devpts" to rootfs at "/dev/pts": gid=5: invalid argument' >&2
                      exit 125
                    fi
                    exit 99
                    """
                ),
                encoding="utf-8",
            )
            fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IEXEC)

            result = self.run_smoke({"PATH": f"{tmpdir}:/usr/bin:/bin"})

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Docker cannot execute a trivial container", result.stderr)
        self.assertIn("docker_cli=29.5.2", result.stderr)
        self.assertIn("storage_driver=vfs", result.stderr)
        self.assertIn("uidmap=missing newuidmap/newgidmap", result.stderr)
        self.assertIn("Docker reached OCI runtime startup", result.stderr)
        self.assertIn("sudo apt-get install -y uidmap", result.stderr)


if __name__ == "__main__":
    unittest.main()
