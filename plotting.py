from mt_tools.compute_mt import edi_to_dataframe
from mt_tools.smoothing import smooth_mt_df
from mt_tools.plotting import plot_mt

df = edi_to_dataframe("edi file name")

df_s = smooth_mt_df(df, mode="medium")

plot_mt(df_s)
