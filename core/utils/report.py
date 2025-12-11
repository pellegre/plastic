from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import shutil
import subprocess
import textwrap

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base.trainer import TrainingResult


class RunReportBuilder:
    """
    Build a LaTeX report for a single run directory and optionally compile it
    to a PDF. Keeps formatting minimal so that it works even with a very small
    TeX installation.
    """

    def __init__(
        self,
        run_dir: Path,
        config: Any,
        metrics: Dict[str, Any],
        artifacts: Dict[str, str],
        log_path: Path,
        training_result: Optional["TrainingResult"] = None,
    ) -> None:
        self.run_dir = run_dir
        self.config = config
        self.metrics = dict(metrics)
        self.artifacts = dict(artifacts)
        self.log_path = log_path
        self.training_result = training_result

    # Public API -----------------------------------------------------------
    def write_tex(self, log_tail_lines: int = 200) -> Path:
        """
        Render a LaTeX file containing configuration, metrics, plots, and the
        tail of the run log.
        """
        tex_body = self._render_document(log_tail_lines=log_tail_lines)
        tex_path = self.run_dir / "report.tex"
        tex_path.write_text(tex_body, encoding="utf-8")
        return tex_path

    def compile_pdf(self, tex_path: Path) -> Tuple[Optional[Path], str]:
        """
        Compile the generated TeX file to PDF. Tries lightweight engines in
        order (tectonic, then pdflatex) if available on the system. Returns the
        PDF path and a message describing the outcome.
        """
        engines: List[tuple[str, List[str]]] = [
            ("tectonic", ["tectonic", "--keep-logs", tex_path.name]),
            ("pdflatex", ["pdflatex", "-interaction=nonstopmode", tex_path.name]),
        ]

        def _decode_output(out: Optional[bytes]) -> str:
            if out is None:
                return ""
            try:
                return out.decode("utf-8", errors="replace")
            except Exception:
                return str(out)

        missing_engines: List[str] = []
        errors: List[str] = []

        for name, cmd in engines:
            if shutil.which(name) is None:
                missing_engines.append(name)
                continue
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self.run_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                pdf_path = tex_path.with_suffix(".pdf")
                if pdf_path.exists():
                    out_text = _decode_output(proc.stdout).strip()
                    msg = f"{name} succeeded"
                    if out_text:
                        msg += f"; output: {out_text[:1000]}"
                    return pdf_path, msg
                errors.append(f"{name} ran but no PDF found at {pdf_path}")
            except subprocess.CalledProcessError as exc:
                output = _decode_output(exc.stdout).strip()
                errors.append(
                    f"{name} failed with exit code {exc.returncode}: {output[:1000]}"
                )

        if missing_engines and not errors:
            return None, f"no latex engine found (looked for: {', '.join(missing_engines)})"

        if errors:
            return None, " | ".join(errors)

        return None, "latex engine not available or compilation failed for unknown reasons"

    # Internal helpers ----------------------------------------------------
    def _render_document(self, log_tail_lines: int) -> str:
        cfg_table = self._tabular(
            self._config_rows(), ("parameter", "value"), align="ll"
        )
        metrics_table = self._tabular(
            self._metric_rows(), ("metric", "value"), align="ll"
        )

        figures = [
            self._figure_block(
                "training and validation curves",
                self._artifact_path("training_curves", "training_curves.png"),
            ),
            self._figure_block(
                "rolling accuracy (single graph)",
                self._artifact_path("rolling_accuracy", "rolling_accuracy.png"),
            ),
            self._figure_block(
                "rolling accuracy across graph boundaries",
                self._artifact_path(
                    "rolling_accuracy_boundaries", "rolling_accuracy_boundaries.png"
                ),
            ),
            self._double_figure_block(
                "random graphs",
                [
                    (
                        self._artifact_path("train_graph", "train_graph.png"),
                        "train split example",
                    ),
                    (
                        self._artifact_path("test_graph", "test_graph.png"),
                        "test split example",
                    ),
                ],
            ),
            self._double_figure_block(
                "action distributions",
                [
                    (
                        self._artifact_path(
                            "train_action_hist", "train_action_hist.png"
                        ),
                        "train action-id histogram",
                    ),
                    (
                        self._artifact_path(
                            "test_action_hist", "test_action_hist.png"
                        ),
                        "test action-id histogram",
                    ),
                ],
            ),
        ]
        figure_blocks = "\n\n".join(filter(None, figures))

        document = textwrap.dedent(
            f"""
            \\documentclass[11pt]{{article}}
            \\usepackage{{graphicx}}
            \\usepackage{{booktabs}}
            \\usepackage{{geometry}}
            \\usepackage{{float}}
            \\usepackage{{hyperref}}
            \\usepackage{{array}}
            \\geometry{{margin=1in}}

            \\title{{Graph Run Report}}
            \\author{{workflow {self._escape(self.run_dir.name)}}}
            \\date{{}}

            \\begin{{document}}
            \\maketitle

            \\section*{{Run Configuration}}
            \\begin{{center}}
            {cfg_table}
            \\end{{center}}

            \\section*{{Key Metrics}}
            \\begin{{center}}
            {metrics_table}
            \\end{{center}}

            \\section*{{Visualizations}}
            {figure_blocks}

            \\end{{document}}
            """
        ).strip() + "\n"

        return document

    def _config_rows(self) -> List[tuple[str, str]]:
        if hasattr(self.config, "__dict__"):
            items = list(self.config.__dict__.items())
        elif isinstance(self.config, dict):
            items = list(self.config.items())
        else:
            items = []
        rows: List[tuple[str, str]] = []
        for key, value in items:
            rows.append((str(key), self._format_value(value)))
        return rows

    def _metric_rows(self) -> List[tuple[str, str]]:
        rows: Dict[str, str] = {}
        for key, value in self.metrics.items():
            rows[key] = self._format_metric_value(key, value)
        rows.update(self._training_metrics_from_result())
        return sorted(rows.items(), key=lambda kv: kv[0])

    def _training_metrics_from_result(self) -> Dict[str, str]:
        if self.training_result is None:
            return {}
        res: Dict[str, str] = {}

        def last(items: Iterable[float]) -> Optional[float]:
            items_list = list(items)
            return items_list[-1] if items_list else None

        final_train_loss = last(self.training_result.train_loss)
        final_val_loss = last(self.training_result.val_loss)
        final_train_acc = last(self.training_result.train_accuracy)
        final_val_acc = last(self.training_result.val_accuracy)

        if final_train_loss is not None:
            res.setdefault("final_train_loss", self._format_value(final_train_loss))
        if final_val_loss is not None:
            res.setdefault("final_val_loss", self._format_value(final_val_loss))
        if final_train_acc is not None:
            res.setdefault(
                "final_train_accuracy",
                self._format_metric_value("final_train_accuracy", final_train_acc),
            )
        if final_val_acc is not None:
            res.setdefault(
                "final_val_accuracy",
                self._format_metric_value("final_val_accuracy", final_val_acc),
            )

        res.setdefault(
            "best_epoch", self._format_value(getattr(self.training_result, "best_epoch", ""))
        )
        res.setdefault(
            "best_val_loss",
            self._format_value(getattr(self.training_result, "best_val_loss", "")),
        )
        return res

    def _format_value(self, value: Any) -> str:
        if isinstance(value, float):
            if abs(value) >= 1000:
                return f"{value:,.1f}"
            return f"{value:.4f}"
        return str(value)

    def _format_metric_value(self, key: str, value: Any) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if self._is_percentage_key(key) and isinstance(value, (int, float)):
            # If value looks like a fraction, scale; otherwise assume already percent.
            pct_value = float(value)
            if abs(pct_value) <= 1.5:
                pct_value *= 100.0
            return f"{pct_value:.2f}%"
        return self._format_value(value)

    def _is_percentage_key(self, key: str) -> bool:
        keywords = ("accuracy", "rate", "overlap")
        k = key.lower()
        return any(term in k for term in keywords)

    def _tabular(self, rows: List[tuple[str, str]], headers: tuple[str, str], align: str) -> str:
        if not rows:
            return "\\emph{no data available}"
        header_line = f"{self._escape(headers[0])} & {self._escape(headers[1])} \\\\ \\midrule"
        body = "\n".join(
            f"{self._escape(k)} & {self._escape(v)} \\\\" for k, v in rows
        )
        return textwrap.dedent(
            f"""
            \\begin{{tabular}}{{{align}}}
            \\toprule
            {header_line}
            {body}
            \\bottomrule
            \\end{{tabular}}
            """
        ).strip()

    def _figure_block(self, caption: str, path: Optional[Path]) -> str:
        if path is None or not path.exists():
            return ""
        rel = self._relative_path(path)
        safe_caption = self._escape(caption)
        return textwrap.dedent(
            f"""
            \\begin{{figure}}[H]
            \\centering
            \\includegraphics[width=0.95\\textwidth]{{{rel}}}
            \\caption{{{safe_caption}}}
            \\end{{figure}}
            """
        ).strip()

    def _double_figure_block(self, caption: str, entries: List[tuple[Optional[Path], str]]) -> str:
        existing = [
            (p, c) for p, c in entries if p is not None and p.exists()
        ]
        if not existing:
            return ""
        figures = []
        for path, cap in existing:
            rel = self._relative_path(path)
            figures.append(
                textwrap.dedent(
                    f"""
                    \\begin{{minipage}}{{0.48\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{{rel}}}
                    \\small {self._escape(cap)}
                    \\end{{minipage}}
                    """
                ).strip()
            )
        body = "\n\\hfill\n".join(figures)
        return textwrap.dedent(
            f"""
            \\begin{{figure}}[H]
            \\centering
            {body}
            \\caption{{{self._escape(caption)}}}
            \\end{{figure}}
            """
        ).strip()

    def _read_log_tail(self, lines: int) -> str:
        if not self.log_path.exists():
            return "log file not found"
        content = self.log_path.read_text(encoding="utf-8").splitlines()
        tail = content[-lines:] if lines > 0 else content
        return "\n".join(tail)

    def _escape(self, value: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        text = str(value)
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        return text

    def _artifact_path(self, key: str, default_name: Optional[str] = None) -> Optional[Path]:
        path_str = self.artifacts.get(key)
        if path_str:
            path = Path(path_str)
            if not path.is_absolute():
                path = self.run_dir / path
            return path
        if default_name:
            candidate = self.run_dir / default_name
            if candidate.exists():
                return candidate
        return None

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.run_dir))
        except ValueError:
            return path.name
