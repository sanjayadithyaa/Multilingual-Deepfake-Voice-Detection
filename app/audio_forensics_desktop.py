from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from audio_forensics_core import AudioForensicsAnalyzer, AnalysisResult, FEATURE_DESCRIPTIONS


class AudioForensicsApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Audio Forensics Workbench")
        self.root.geometry("1280x860")
        self.root.minsize(1100, 760)

        self.analyzer = AudioForensicsAnalyzer()
        self.results: list[AnalysisResult] = []
        self.current_path = tk.StringVar()
        self.model_name = tk.StringVar(value="CNN")
        self.status_text = tk.StringVar(value="Ready")

        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        top = ttk.LabelFrame(main, text="Input")
        top.pack(fill="x", pady=(0, 10))

        ttk.Label(top, text="Selected file or folder").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(top, textvariable=self.current_path, width=90).grid(row=1, column=0, columnspan=5, sticky="ew", padx=8)
        ttk.Button(top, text="Choose File", command=self.choose_file).grid(row=0, column=1, padx=6, pady=8)
        ttk.Button(top, text="Choose Folder", command=self.choose_folder).grid(row=0, column=2, padx=6, pady=8)
        ttk.Label(top, text="Model").grid(row=0, column=3, sticky="e", padx=(20, 6))
        ttk.Combobox(top, textvariable=self.model_name, values=["CNN", "Random Forest"], state="readonly", width=18).grid(
            row=0, column=4, padx=8, pady=8
        )
        ttk.Button(top, text="Analyze", command=self.start_analysis).grid(row=0, column=5, padx=8, pady=8)
        top.columnconfigure(0, weight=1)

        button_bar = ttk.Frame(main)
        button_bar.pack(fill="x", pady=(0, 8))
        ttk.Button(button_bar, text="Export PDF Report", command=self.export_pdf).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Export CSV Summary", command=self.export_csv).pack(side="left")
        ttk.Label(button_bar, textvariable=self.status_text).pack(side="right")

        content = ttk.PanedWindow(main, orient="horizontal")
        content.pack(fill="both", expand=True)

        left = ttk.Frame(content)
        right = ttk.Frame(content)
        content.add(left, weight=1)
        content.add(right, weight=2)

        result_frame = ttk.LabelFrame(left, text="Prediction Results")
        result_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(
            result_frame,
            columns=("file", "model", "label", "real", "fake"),
            show="headings",
            height=20,
        )
        self.tree.heading("file", text="File")
        self.tree.heading("model", text="Model")
        self.tree.heading("label", text="Prediction")
        self.tree.heading("real", text="Real %")
        self.tree.heading("fake", text="Fake %")
        self.tree.column("file", width=270)
        self.tree.column("model", width=110, anchor="center")
        self.tree.column("label", width=100, anchor="center")
        self.tree.column("real", width=80, anchor="center")
        self.tree.column("fake", width=80, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=8, pady=8)
        self.tree.bind("<<TreeviewSelect>>", self.on_result_select)

        detail_frame = ttk.LabelFrame(right, text="Forensic Detail")
        detail_frame.pack(fill="both", expand=True)

        self.detail_text = tk.Text(detail_frame, wrap="word", font=("Segoe UI", 10))
        self.detail_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.detail_text.configure(state="disabled")

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose audio file",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac"), ("All files", "*.*")],
        )
        if path:
            self.current_path.set(path)

    def choose_folder(self) -> None:
        path = filedialog.askdirectory(title="Choose folder with audio files")
        if path:
            self.current_path.set(path)

    def start_analysis(self) -> None:
        selected_path = self.current_path.get().strip()
        if not selected_path:
            messagebox.showwarning("Missing input", "Choose an audio file or a folder first.")
            return

        path = Path(selected_path)
        if not path.exists():
            messagebox.showerror("Invalid path", "The selected path does not exist.")
            return

        self.status_text.set("Analyzing...")
        self._clear_results()

        worker = threading.Thread(target=self._run_analysis, args=(path, self.model_name.get()), daemon=True)
        worker.start()

    def _run_analysis(self, path: Path, model_name: str) -> None:
        try:
            if path.is_dir():
                results = self.analyzer.analyze_folder(path, model_name=model_name)
            else:
                results = [self.analyzer.analyze_file(path, model_name=model_name)]
            self.root.after(0, self._finish_analysis, results)
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, self._show_error, str(exc))

    def _finish_analysis(self, results: list[AnalysisResult]) -> None:
        self.results = results
        for index, result in enumerate(results):
            self.tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    Path(result.file_path).name,
                    result.model_name,
                    result.predicted_label,
                    f"{result.real_probability:.2f}",
                    f"{result.fake_probability:.2f}",
                ),
            )

        if results:
            self.tree.selection_set("0")
            self.show_result_detail(results[0])

        self.status_text.set(f"Completed: {len(results)} item(s) analyzed")

    def _show_error(self, message: str) -> None:
        self.status_text.set("Failed")
        messagebox.showerror("Analysis failed", message)

    def _clear_results(self) -> None:
        self.results = []
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._set_detail_text("")

    def on_result_select(self, _event: object) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        index = int(selection[0])
        if 0 <= index < len(self.results):
            self.show_result_detail(self.results[index])

    def show_result_detail(self, result: AnalysisResult) -> None:
        lines = [
            "AUDIO FORENSIC SCREENING RESULT",
            "",
            f"File: {result.file_path}",
            f"Model: {result.model_name}",
            f"Predicted label: {result.predicted_label}",
            f"Real probability: {result.real_probability:.2f}%",
            f"Fake probability: {result.fake_probability:.2f}%",
            "",
            "Extracted audio features",
            "------------------------",
        ]
        for feature_name, value in result.summary_features.items():
            lines.append(f"{feature_name}: {value:.4f}")

        lines.extend(
            [
                "",
                "Reason for prediction",
                "---------------------",
            ]
        )
        for reason in result.reason_lines:
            lines.append(f"- {reason}")

        lines.extend(
            [
                "",
                "User-friendly explanation",
                "-------------------------",
                result.user_explanation,
                "",
                "Feature descriptions",
                "--------------------",
            ]
        )
        for feature_name in result.summary_features:
            description = FEATURE_DESCRIPTIONS.get(feature_name, "")
            lines.append(f"{feature_name}: {description}")

        self._set_detail_text("\n".join(lines))

    def _set_detail_text(self, content: str) -> None:
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", content)
        self.detail_text.configure(state="disabled")

    def export_pdf(self) -> None:
        if not self.results:
            messagebox.showwarning("No results", "Analyze at least one file before exporting a report.")
            return

        path = filedialog.asksaveasfilename(
            title="Save forensic PDF report",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
        )
        if not path:
            return

        try:
            self.analyzer.export_pdf_report(self.results, path)
            self.status_text.set(f"PDF report saved: {path}")
            messagebox.showinfo("Report saved", f"PDF report saved to:\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", str(exc))

    def export_csv(self) -> None:
        if not self.results:
            messagebox.showwarning("No results", "Analyze at least one file before exporting a CSV summary.")
            return

        path = filedialog.asksaveasfilename(
            title="Save CSV summary",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return

        try:
            self.analyzer.save_results_csv(self.results, path)
            self.status_text.set(f"CSV saved: {path}")
            messagebox.showinfo("CSV saved", f"CSV summary saved to:\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", str(exc))


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = AudioForensicsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
