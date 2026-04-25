import os
import shutil
import signal
import subprocess
import time


CURRENT_DIR_PATH = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR_BASE = os.path.abspath(os.path.join(CURRENT_DIR_PATH, '../'))


def _as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value or "")


def _resolve_bugsinpy_tool(tool_name: str) -> str:
    candidates = []
    bin_dir = str(os.environ.get("BUGSINPY_BIN_DIR", "")).strip()
    if bin_dir:
        candidates.append(os.path.join(bin_dir, tool_name))

    candidates.extend(
        [
            os.path.join(PROJECT_DIR_BASE, "BugsInPy", "framework", "bin", tool_name),
            os.path.join("/workspace", "BugsInPy", "framework", "bin", tool_name),
            os.path.join("/BugsInPy", "framework", "bin", tool_name),
        ]
    )

    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return tool_name


def map_git_to_bugsinpy_project_name(git_project_name: str):
    if git_project_name == "cli":
        bugsinpy_project_name = "httpie"
    elif git_project_name == "spaCy":
        bugsinpy_project_name = "spacy"
    else:
        bugsinpy_project_name = git_project_name
    return bugsinpy_project_name


def bugsinpy_checkout(project_name, bug_id, checkout_path) -> bool:
    # if ti success, it always has "Removing bugsinpy_run_test.sh"
    if os.path.isdir(checkout_path):
        print(f"checkout path {checkout_path} exists, delete it first!")
        shutil.rmtree(checkout_path)
    print("start checkout...")
    command = [
        _resolve_bugsinpy_tool("bugsinpy-checkout"),
        "-p", project_name,
        "-i", bug_id,
        "-w", checkout_path
    ]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # p = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out_text = _as_text(out)
    err_text = _as_text(err)
    print(out_text)
    if p.returncode != 0:
        print(f"Checkout failed for {project_name}-{bug_id}\n")
        print(f"out: {out_text}\n")
        print(f"err: {err_text}\n")
        return False
    if not os.path.isdir(checkout_path):
        print(f"Checkout failed for {project_name}-{bug_id}: not os.path.isdir(checkout_path)")
        return False
    print("checkout succeeded")
    print("finish checkout...\n\n")
    if "Removing bugsinpy_run_test.sh" in str(out):
        return True
    return True


def bugsinpy_compile(project_dir) -> bool:
    os.chdir(project_dir)
    print("start compile...")
    print(f"current work dir is: {os.getcwd()}")
    # if wrong, it always has "This is not a checkout project folder"
    p = subprocess.Popen(
        [_resolve_bugsinpy_tool("bugsinpy-compile")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = p.communicate()
    out_text = _as_text(out)
    err_text = _as_text(err)
    print(out_text or err_text)
    print("finish compile...\n\n")
    if p.returncode != 0:
        print(f"Compile failed with return code {p.returncode}")
        return False
    if "This is not a checkout project folder" in out_text:
        return False
    return True


def bugsinpy_test(project_dir) -> str:
    os.chdir(project_dir)
    print("\n------start test------")
    print(f"current work dir is: {os.getcwd()}")
    out, err = command_with_timeout([_resolve_bugsinpy_tool("bugsinpy-test")], timeout=120)
    out_text = _as_text(out)
    err_text = _as_text(err)
    print(out_text or err_text)
    print("------finish test------")
    # if there are 1 passed and 1 failed, it will return False
    # unittest return "FAILED" or "OK"
    # pytest return "failed" or "passed"
    if "FAILED" in out_text or "failed" in out_text:
        return 'Fail'
    if "passed" in out_text or "OK" in out_text:
        return 'Plausible'
    # It is possible return something like "module is not found"
    return 'Fail'


def command_with_timeout(cmd, timeout=300):
    p = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    t_beginning = time.time()
    while True:
        if p.poll() is not None:
            break
        seconds_passed = time.time() - t_beginning
        if timeout and seconds_passed > timeout:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                out, err = p.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                out, err = p.communicate()
            return (out or b"") + b"\nTIMEOUT", (err or b"") + b"\nTIMEOUT"
        time.sleep(1)
    out, err = p.communicate()
    return out, err
