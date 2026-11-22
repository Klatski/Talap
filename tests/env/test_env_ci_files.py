"""Файлы CI читаются GitHub: без PyYAML (только stdlib) проверяем структуру, которую ломает правка через heredoc.

22.11 шаг с `--format '{{join .RepoDigests "\\n"}}'` записался с настоящим переводом строки: строка с нулевым
отступом посреди шага сделала YAML неверным, и GitHub не запускал ни одного задания (Failure за 0 с), пока
никто не открыл вкладку Actions. Локальный check.py этого не видел.
"""

import re
import unittest

from env_helpers import ROOT

WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
TOP_LEVEL = re.compile(r"^(name|on|jobs|permissions|env|concurrency|defaults):")


class Workflows(unittest.TestCase):
    def test_workflow_files_exist(self):
        self.assertTrue(WORKFLOWS)

    def test_workflow_structure_is_valid_yaml_shape(self):
        for path in WORKFLOWS:
            raw = path.read_bytes()
            with self.subTest(workflow=path.name):
                self.assertNotIn(b"\r", raw)
                self.assertNotIn(b"\t", raw)  # табуляция в отступах YAML запрещена
                for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                    if not line.strip() or line.lstrip().startswith("#"):
                        continue
                    indent = len(line) - len(line.lstrip(" "))
                    if indent == 0:
                        self.assertRegex(line, TOP_LEVEL, f"line {n}: a value broke onto a new line? {line!r}")
                    else:
                        self.assertEqual(indent % 2, 0, f"line {n}: odd indent {line!r}")
                    m = re.match(r"^\s*(?:- )?run: (?![|>])(.*)$", line)
                    if m:  # однострочная команда: кавычки должны закрываться в этой же строке
                        value = m.group(1)
                        self.assertEqual(value.count("'") % 2, 0, f"line {n}: unbalanced ' in {line!r}")
                        self.assertEqual(value.count('"') % 2, 0, f"line {n}: unbalanced \" in {line!r}")

    def test_ci_runs_the_single_check_command(self):
        text = "\n".join(p.read_text(encoding="utf-8") for p in WORKFLOWS)
        self.assertIn("run: python check.py", text)
        self.assertIn("--network none", text)


if __name__ == "__main__":
    unittest.main()
