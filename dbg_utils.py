from pathlib import Path

import numpy as np


class DbgUtils:
    @staticmethod
    def export_npy_file(data, out_npy_path):
        np.save(out_npy_path, np.array(data))
        print("export:", out_npy_path)


class TextWriter:
    def __init__(self):
        self.fp_dict = {}

    def __del__(self):
        for fp in self.fp_dict.values():
            fp.close()

    def get_file_handle(self, filepath):
        name = Path(filepath).stem
        if name in self.fp_dict:
            return self.fp_dict[name]
        else:
            fp = open(filepath, "w", encoding="utf8")
            self.fp_dict[name] = fp
            return fp

    def write(self, filepath, data, factor=32768):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        data = data.reshape(-1)

        if factor:
            data = data * factor

        fp = self.get_file_handle(filepath)
        fp.write(", ".join(data.astype(int).astype(str)) + ", ")
        fp.write("\n")
