import os
import shutil
import tkinter as tk
from tkinter import filedialog

band_map = {
    'A':'HF',
    'B':'LF1',
    'C':'LF2',
    'D':'LF3',
    'E':'LF4',
    'F':'Free',
    'G':'LF5',
}

def pick_folders():
    root = tk.Tk()
    root.withdraw()
    root.attributes(-'-topmost', True)
    src = filedialog.askdirectory(title="Select Source Folder")
    if not src:
        print("Source selection cancelled")
        return None, None
    dst = filedialog.askdirectory(title="Select Destination Folder")
    if not dst:
        print("Destination folder not selected")
        return None, None
    return src, dst

if __name__ == "__main__":
    source, dest = pick_folders()
    if source and dest:
        print("Source", source)
        print("Destination", dest)

    os.makedirs(dest, exist_ok=True)

    for filename in os.listdir(source):
        if filename.endswith(".ats"):
            run_no = filename[4:6]
            band_code = filename[7]

            run_label = f"Run{int(run_no)}"
            band_label = band_map.get(band_code, band_code)

            folder_name = f"{run_label}_{band_label}"
            target_path = os.path.join(dest, folder_name)
            os.makedirs(target_path, exist_ok=True)
            
            shutil.copy2(
                os.path.join(source, filename).
                os.path.join(target_path, filename)
            )

    print("Done!")