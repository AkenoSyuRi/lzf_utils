import glob
import os
import random
import re
from itertools import chain


class FileUtils:

    @staticmethod
    def filename_sub(filepath, *sub_patterns, base_dir, splitter='/'):
        filename = os.path.basename(filepath)

        for sub_pat in sub_patterns:
            assert splitter in sub_pat, f"invalid sub_pattern: {sub_pat}, splitter is missing: {splitter}"
            from_pat, to_pat = sub_pat.split(splitter)
            filename = re.sub(from_pat, to_pat, filename)

        new_filepath = os.path.join(base_dir, filename)
        return new_filepath

    @staticmethod
    def iglob_files(*patterns):
        it_list = []
        for pat in patterns:
            recursive = "**" in pat
            it_list.append(glob.iglob(pat, recursive=recursive))
        return chain(*it_list)

    @classmethod
    def glob_files(cls, *patterns, shuffle=False):
        it = cls.iglob_files(*patterns)
        files = list(it)
        if shuffle:
            random.shuffle(files)
        return files
