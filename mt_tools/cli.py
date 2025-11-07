"""Command-line interface for mt_tools

Provides a simple CLI that wraps the library functionality.
"""
import argparse
import sys
from pathlib import Path

from . import __version__
from .core import EDIParser, MTDimensionalityAnalyzer, MTVisualizer, MTReportGenerator


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog='mt_tools', description='Magnetotelluric dimensionality analysis')
    p.add_argument('edi_files', nargs='+', help='One or more EDI files to analyze')
    p.add_argument('-o', '--output', default='results', help='Output directory for plots and reports')
    p.add_argument('--no-plots', action='store_true', help='Do not generate plots (only CSV/text reports)')
    p.add_argument('--version', action='store_true', help='Print version and exit')
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