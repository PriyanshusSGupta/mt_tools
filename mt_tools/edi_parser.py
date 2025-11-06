import numpy as np

class EDIParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {}

    def parse(self):
        with open(self.filepath, "r") as f:
            lines = f.readlines()

        for line in lines:
            if "DATAID" in line:
                self.data["station_id"] = line.split("=")[1].strip().replace('"', '')
                break

        keys = ["FREQ", "ZXXR", "ZXXI", "ZXYR", "ZXYI", "ZYXR", "ZYXI", "ZYYR", "ZYYI"]

        for k in keys:
            self.data[k.lower()] = self._extract(lines, ">" + k)

        self.data["Zxx"] = self.data["zxxr"] + 1j * self.data["zxxi"]
        self.data["Zxy"] = self.data["zxyr"] + 1j * self.data["zxyi"]
        self.data["Zyx"] = self.data["zyxr"] + 1j * self.data["zyxi"]
        self.data["Zyy"] = self.data["zyyr"] + 1j * self.data["zyyi"]

        return self.data

    def _extract(self, lines, marker):
        arr = []
        capture = False
        for line in lines:
            if marker in line:
                capture = True
                continue
            if capture:
                if line.strip().startswith(">"):
                    break
                for v in line.split():
                    try:
                        arr.append(float(v))
                    except:
                        pass
        return np.array(arr)
