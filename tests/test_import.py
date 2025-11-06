def test_import_mt_tools():
    import mt_tools
    from mt_tools import EDIParser, MTDimensionalityAnalyzer

    assert hasattr(mt_tools, '__version__')
    assert callable(EDIParser)
    assert callable(MTDimensionalityAnalyzer)
