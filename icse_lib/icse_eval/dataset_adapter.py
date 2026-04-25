import os
import shutil
from typing import Any, Dict

import bugsinpy_command
import defects4j_command
from bugsinpy_command import map_git_to_bugsinpy_project_name
from defects4j_command import defects4j_project_name_repository_map
from util import bugsinpy_bugs_fail_ground_test

CURRENT_DIR_PATH = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR_BASE = os.path.abspath(os.path.join(CURRENT_DIR_PATH, '../'))


class DatasetAdapter:
    def checkout(self, project_name, bug_id, project_path): raise NotImplementedError

    def compile(self, project_path): raise NotImplementedError

    def compile_detailed(self, project_path) -> Dict[str, Any]:
        ok = self.compile(project_path)
        return {
            "ok": bool(ok),
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "elapsed_sec": None,
            "error_family": "" if ok else "unknown_compile_fail",
        }

    def test(self, project_path): raise NotImplementedError

    def test_one(self, project_path, test_case, timeout=300):
        return b"", b""

    def map_project_name(self, name): raise NotImplementedError

    def should_skip_bug(self, bug_id): return False

    def build_project_path(self, project_name, bug_id): raise NotImplementedError

    def reset_cwd(self): os.chdir(PROJECT_DIR_BASE)

    def clean_checkout_dir(self, path):
        if os.path.isdir(path):
            shutil.rmtree(path)

    @property
    def dataset_name(self): raise NotImplementedError


class BugsInPy(DatasetAdapter):
    def checkout(self, project_name, bug_id, project_path):
        # make sure to reset the working directory
        self.reset_cwd()
        return bugsinpy_command.bugsinpy_checkout(project_name, bug_id, project_path)

    def compile(self, project_path):
        return bugsinpy_command.bugsinpy_compile(project_path)

    def test(self, project_path):
        return bugsinpy_command.bugsinpy_test(project_path)

    def map_project_name(self, name):
        return map_git_to_bugsinpy_project_name(name)

    def should_skip_bug(self, bug_id):
        return str(bug_id) in bugsinpy_bugs_fail_ground_test

    def build_project_path(self, project_name, bug_id):
        return os.path.abspath(
            os.path.join(PROJECT_DIR_BASE, f"../BugsInPy/framework/bin/temp/{project_name}_{bug_id}"))

    def clean_checkout_dir(self, checkout_path_parent):
        if checkout_path_parent is None:
            checkout_path_parent = os.path.abspath(os.path.join(PROJECT_DIR_BASE, f"../BugsInPy/framework/bin/temp"))
        super().clean_checkout_dir(checkout_path_parent)

    @property
    def dataset_name(self):
        return "bugsinpy"


class Defects4J(DatasetAdapter):
    def checkout(self, project_name, bug_id, project_path):
        # make sure to reset the working directory
        self.reset_cwd()
        return defects4j_command.defects4j_checkout(project_name, bug_id, project_path)

    def compile(self, project_path):
        return defects4j_command.defects4j_compile(project_path)

    def compile_detailed(self, project_path):
        return defects4j_command.defects4j_compile_detailed(project_path)

    def test(self, project_path):
        return defects4j_command.defects4j_test(project_path)

    def test_one(self, project_path, test_case, timeout=300):
        return defects4j_command.defects4j_test_one(project_path, test_case, timeout)

    def map_project_name(self, name):
        return self._get_key_by_value(defects4j_project_name_repository_map, name)

    def build_project_path(self, project_name, bug_id):
        return os.path.abspath(
            os.path.join(PROJECT_DIR_BASE, f"../defects4j/framework/bin/temp/{project_name}_{bug_id}"))

    @staticmethod
    def _get_key_by_value(d, target_value):
        return next((k for k, v in d.items() if v == target_value), None)

    def clean_checkout_dir(self, checkout_path_parent=None):
        if checkout_path_parent is None:
            checkout_path_parent = os.path.abspath(os.path.join(PROJECT_DIR_BASE, f"../defects4j/framework/bin/temp"))
        super().clean_checkout_dir(checkout_path_parent)

    @property
    def dataset_name(self):
        return "defects4j"
