# sample script: process_edi.py
from mt_tools import EDIParser, MTDimensionalityAnalyzer, MTVisualizer, MTReportGenerator

# parse EDI
parser = EDIParser("edi file name")
edi_data = parser.parse()

# run analysis
analyzer = MTDimensionalityAnalyzer(edi_data)
results = analyzer.analyze()   # returns results dict or populates analyzer.results

# save reports
reporter = MTReportGenerator(edi_data, results, output_dir="results")
reporter.generate_all_reports()

# create plots (if implemented)
viz = MTVisualizer(edi_data, results, output_dir="results")
viz.create_all_plots()