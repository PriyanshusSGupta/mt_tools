# mt_tools

Magnetotelluric (MT) dimensionality analysis library and CLI.

Usage (after installing the package):

    mt_tools data1.edi data2.edi -o results/

Or programmatically:

    from mt_tools import EDIParser, MTDimensionalityAnalyzer

    parser = EDIParser('file.edi')
    data = parser.parse()
    analyzer = MTDimensionalityAnalyzer(data)
    results = analyzer.analyze()

See the `pyproject.toml` for metadata and console script configuration.
