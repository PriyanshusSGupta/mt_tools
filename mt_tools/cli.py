"""Command-line interface for mt_tools

Provides a simple CLI that wraps the library functionality.
"""
import argparse
import sys
from pathlib import Path
import shutil
from typing import Tuple, Optional

from . import __version__
from .core import EDIParser, MTDimensionalityAnalyzer, MTVisualizer, MTReportGenerator


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog='mt_tools', description='Magnetotelluric dimensionality analysis')

    # existing EDI processing args
    p.add_argument('edi_files', nargs='*', help='One or more EDI files to analyze')
    p.add_argument('-o', '--output', default='results', help='Output directory for plots and reports')
    p.add_argument('--no-plots', action='store_true', help='Do not generate plots (only CSV/text reports)')
    p.add_argument('--version', action='store_true', help='Print version and exit')

    # new: sort .ats files non-interactively
    p.add_argument('--sort-ats', nargs=2, metavar=('SOURCE', 'DEST'),
                   help='Sort .ats files from SOURCE into DEST by run and band (copy by default)')
    p.add_argument('--move-ats', action='store_true', help='Move .ats files instead of copying when using --sort-ats')

    return p.parse_args(argv)


def process_file(edi_path: Path, output_dir: Path, make_plots: bool):
    print(f"Processing: {edi_path}")
    parser = EDIParser(str(edi_path))
    edi_data = parser.parse()
    analyzer = MTDimensionalityAnalyzer(edi_data)
    results = analyzer.analyze()

    if make_plots:
        visualizer = MTVisualizer(edi_data, results, output_dir=output_dir)
        visualizer.create_all_plots()

    report_gen = MTReportGenerator(edi_data, results, output_dir=output_dir)
    report_gen.generate_all_reports()


def main(argv=None):
    args = parse_args(argv)

    if args.version:
        print(f"mt_tools {__version__}")
        return 0

    # If user requested .ats sorting, perform that and exit
    if args.sort_ats:
        src_str, dst_str = args.sort_ats
        src = Path(src_str)
        dst = Path(dst_str)
        try:
            sort_ats_cli(src, dst, move=args.move_ats)
        except Exception as e:
            print(f"Error sorting .ats files: {e}", file=sys.stderr)
            return 2
        print(".ats sorting complete.")
        return 0

    # Otherwise proceed with regular EDI processing
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    for edi in args.edi_files:
        edi_path = Path(edi)
        if not edi_path.exists():
            print(f"File not found: {edi}", file=sys.stderr)
            continue
        try:
            process_file(edi_path, outdir, not args.no_plots)
        except Exception as e:
            print(f"Error processing {edi}: {e}", file=sys.stderr)

    print("Analysis complete. Results saved to:", outdir)
    return 0


def sort_ats_cli(src: Path, dst: Path, move: bool = False) -> Tuple[int, Optional[str]]:
    """Sort .ats files from src into dst by run number and band.

    Files are expected to follow the naming convention used in the original script
    (run number at positions 4-5 and band code at position 7). If parsing fails for a
    file, it will be skipped with a warning.

    Returns (count, message) where count is number of files processed.
    """
    band_map = {
        'A': 'HF',
        'B': 'LF1',
        'C': 'LF2',
        'D': 'LF3',
        'E': 'LF4',
        'F': 'Free',
        'G': 'LF5',
    }

    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Source folder not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    processed = 0
    for entry in sorted(src.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() != '.ats':
            continue

        name = entry.name
        # Defensive parsing: ensure string long enough
        run_no = None
        band_code = None
        try:
            if len(name) >= 8:
                run_no = name[4:6]
                band_code = name[7]
            else:
                raise ValueError("filename too short to parse run/band")

            try:
                run_label = f"Run{int(run_no)}"
            except Exception:
                run_label = f"Run{run_no}"

            band_label = band_map.get(band_code, band_code)

            folder_name = f"{run_label}_{band_label}"
            target_path = dst / folder_name
            target_path.mkdir(parents=True, exist_ok=True)

            if move:
                shutil.move(str(entry), str(target_path / name))
            else:
                shutil.copy2(str(entry), str(target_path / name))

            processed += 1
            print(f"{('Moved' if move else 'Copied')}: {name} -> {target_path}")
        except Exception as e:
            print(f"Skipping {name}: {e}", file=sys.stderr)

    return processed, None


if __name__ == '__main__':
    raise SystemExit(main())


# from mt_tools import  MTReportGenerator, EDIParser, plotting , MTVisualizer, MTDimensionalityAnalyzer
#path = "path_to_your_edi_file"

# parser = EDIParser(path)

# edi_data = parser.parse()
# analyzer = MTDimensionalityAnalyzer(edi_data=edi_data)
# results = analyzer.analyze()
# reporter = MTReportGenerator(edi_data, results)
# reporter.generate_all_reports()

# viz = MTVisualizer(edi_data, results)
# viz.create_all_plots()


# from mt_tools.compute_mt import edi_to_dataframe
# from mt_tools.smoothing import smooth_mt_df
# from mt_tools.plotting import plot_mt

# df = edi_to_dataframe(path)

# df_s = smooth_mt_df(df, mode="sharp")

# plot_mt(df_s)