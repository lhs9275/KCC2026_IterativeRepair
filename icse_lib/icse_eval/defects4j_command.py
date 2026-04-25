import os
import re
import signal
import subprocess
import shutil
import time
from typing import Any, Dict

defects4j_project_name_repository_map = {
    'Chart': 'jfreechart',
    'Cli': 'commons-cli',
    'Closure': 'closure-compiler',
    'Codec': 'commons-codec',
    'Collections': 'commons-collections',
    'Compress': 'commons-compress',
    'Csv': 'commons-csv',
    'Gson': 'gson',
    'JacksonCore': 'jackson-core',
    'JacksonDatabind': 'jackson-databind',
    'JacksonXml': 'jackson-dataformat-xml',
    'Jsoup': 'jsoup',
    'JxPath': 'commons-jxpath',
    'Lang': 'commons-lang',
    'Math': 'commons-math',
    'Mockito': 'mockito',
    'Time': 'joda-time'
}


COMPILE_FAIL_SIGNAL_RE = re.compile(r"\bFAIL\b", re.IGNORECASE)


def _has_compile_fail_signal(stdout: str, stderr: str) -> bool:
    return bool(COMPILE_FAIL_SIGNAL_RE.search(f"{stdout}\n{stderr}"))


def _classify_compile_error_family(stdout: str, stderr: str) -> str:
    combined = f"{stdout}\n{stderr}".lower()
    if "cannot find symbol" in combined:
        return "cannot_find_symbol"
    if (
        "no suitable method found" in combined
        or "cannot be applied to given types" in combined
        or "does not override" in combined
    ):
        return "method_signature"
    if "incompatible types" in combined:
        return "type_mismatch"
    if (
        "must be caught or declared to be thrown" in combined
        or "unreported exception" in combined
    ):
        return "checked_exception"
    if "has private access" in combined or "has protected access" in combined:
        return "access_control"
    if (
        re.search(r"package\s+.+\s+does not exist", combined)
        or re.search(r"import\s+.+\s+does not exist", combined)
    ):
        return "package_or_import"
    if (
        "';' expected" in combined
        or "'}' expected" in combined
        or "reached end of file while parsing" in combined
        or "illegal start of" in combined
    ):
        return "syntax_or_parse"
    return "other_compile_fail"


def clean_tmp_folder(tmp_dir):
    if os.path.isdir(tmp_dir):
        for files in os.listdir(tmp_dir):
            file_p = os.path.join(tmp_dir, files)
            try:
                if os.path.isfile(file_p):
                    os.unlink(file_p)
                elif os.path.isdir(file_p):
                    shutil.rmtree(file_p)
            except Exception as e:
                print(e)
    else:
        os.makedirs(tmp_dir)


def defects4j_checkout(project_name, bug_id, checkout_path) -> bool:
    # delete the checkout path
    if os.path.isdir(checkout_path):
        print(f"checkout path {checkout_path} exists, delete it first!")
        shutil.rmtree(checkout_path)
    print("start checkout...")
    command = [
        "defects4j", "checkout",
        "-p", project_name,
        "-v", f"{bug_id}f",
        "-w", checkout_path
    ]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    print(f"{out.decode()}")
    if p.returncode != 0:
        print(f"Checkout failed for {project_name}-{bug_id}\n")
        print(f"out: {out}\n")
        print(f"err: {err}\n")
        return False
    if not os.path.isdir(checkout_path):
        print(f"Checkout failed for {project_name}-{bug_id}: not os.path.isdir(checkout_path)")
        return False
    print("checkout succeeded")
    print("finish checkout...\n\n")
    return True


def defects4j_compile(project_dir) -> bool:
    return bool(defects4j_compile_detailed(project_dir).get("ok", False))


def defects4j_compile_detailed(project_dir) -> Dict[str, Any]:
    os.chdir(project_dir)
    print("start compile...")
    print(f"current work dir is: {os.getcwd()}")
    started_at = time.time()
    p = subprocess.Popen(["defects4j", "compile"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    stdout = out.decode(errors="replace")
    stderr = err.decode(errors="replace")
    elapsed_sec = round(time.time() - started_at, 6)
    print(f"{stdout}")
    if stderr:
        print(f"{stderr}")
    print("finish compile...\n\n")
    ok = p.returncode == 0 and not _has_compile_fail_signal(stdout, stderr)
    return {
        "ok": bool(ok),
        "returncode": int(p.returncode),
        "stdout": stdout,
        "stderr": stderr,
        "elapsed_sec": float(elapsed_sec),
        "error_family": "" if ok else _classify_compile_error_family(stdout, stderr),
    }


def command_with_timeout(cmd, timeout=300):
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE,
                         start_new_session=True)
    t_beginning = time.time()
    while True:
        if p.poll() is not None:
            break
        seconds_passed = time.time() - t_beginning
        if timeout and seconds_passed > timeout:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            return b'TIMEOUT', b'TIMEOUT'
        time.sleep(1)
    out, err = p.communicate()
    return out, err


def defects4j_test(project_dir, timeout=300) -> str:
    print("\n------start test------")
    os.chdir(project_dir)
    print(f"current work dir is: {os.getcwd()}")
    out, err = command_with_timeout(["defects4j", "test", "-r"], timeout)
    print(f"{out.decode() or err.decode()}")
    print("------finish test------")

    if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
        correctness = 'Timeout'
    elif 'FAIL' in str(err) or 'FAIL' in str(out):
        correctness = 'Fail'
    elif "Failing tests: 0" in str(out):
        correctness = 'Plausible'
    else:
        correctness = 'Fail'
    return correctness


def defects4j_trigger(project_dir, timeout=300):
    os.chdir(project_dir)
    out, err = command_with_timeout(["defects4j", "export", "-p", "tests.trigger"], timeout)
    return out, err


def defects4j_relevant(project_dir, timeout=300):
    os.chdir(project_dir)
    out, err = command_with_timeout(["defects4j", "export", "-p", "tests.relevant"], timeout)
    return out, err


def defects4j_test_one(project_dir, test_case, timeout=300):
    os.chdir(project_dir)
    out, err = command_with_timeout(["defects4j", "test", "-t", test_case], timeout)
    return out, err
