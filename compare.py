#!/usr/bin/python3
import numpy as np
from numpy._typing import NDArray

import ast
import argparse
import re


class Compare():
    def __init__(self, enter_files: list[list[str]]) -> None:
        self.enter_files: list[list[str]] = enter_files
        self.result: list[str] = []

    def antiplagiat(self, insertion_cost=1, deletion_cost=1,
                    substitution_cost=1, transposition_cost=1, advanced=False,
                    pure=False, nice_view=False, round_number=0) -> None:
        for i in range(0, len(self.enter_files)):
            result_of_reading: bool = True
            print(
                f'{self.enter_files[i][0]} & {self.enter_files[i][1]}:',
                end=' ')
            if pure:
                result_of_reading: bool = self.read(
                    self.enter_files[i][0], self.enter_files[i][1])
            else:
                result_of_reading: bool = self.ast_process_files(
                    self.enter_files[i][0], self.enter_files[i][1])
            if not result_of_reading:
                continue
            print("success read", end=', ')

            if advanced:
                answer: np.float64 = self.damerau_levenstain_distance(
                    insertion_cost=insertion_cost,
                    deletion_cost=deletion_cost,
                    substitution_cost=substitution_cost,
                    transposition_cost=transposition_cost)\
                    / max(self.length_f1, self.length_f2)
            else:
                answer: np.float64 = self.levenstain_distance(
                    insertion_cost=insertion_cost,
                    deletion_cost=deletion_cost,
                    substitution_cost=substitution_cost)\
                    / max(self.length_f1, self.length_f2)

            if round_number:
                answer: np.float64 = round(answer, round_number)

            if nice_view:
                self.result.append(str(100 - answer * 100) + '%')
            else:
                self.result.append(str(1.0 - answer))
            print(f"answer: {self.result[-1]} ✅")

    def damerau_levenstain_distance(self, insertion_cost=1, deletion_cost=1,
                                    substitution_cost=1,
                                    transposition_cost=1) -> np.float64:
        n = self.length_f1 + 1
        m = self.length_f2 + 1
        mat: NDArray[np.float64] = np.empty((3, n), dtype=np.float64)

        mat[0, 0] = 0
        for i in range(1, n):
            mat[0, i] = i

        current_line: int = 0  # tip to the last line in imaginary table
        for j in range(1, m):
            current_line = (current_line + 1) % 3
            mat[current_line, 0] = j
            for i in range(1, n):
                if self.f1[i - 1] == self.f2[j - 1]:
                    letter_cost: int = 0
                else:
                    letter_cost: int = substitution_cost

                mat[current_line, i] = min(mat[(current_line - 1) % 3,
                                               i] + deletion_cost,
                                           mat[current_line,
                                               i - 1] + insertion_cost,
                                           mat[(current_line - 1) % 3,
                                               i - 1] + letter_cost)

                # transposition
                if i > 1 and j > 1 and self.f1[i - 1] == self.f2[j - 2]\
                        and self.f1[i - 2] == self.f2[j - 1]:
                    mat[current_line, i] = min(mat[current_line, i], mat[(
                        current_line - 2) % 3, i - 2] + transposition_cost)
        return mat[current_line, -1]

    def levenstain_distance(self, insertion_cost=1, deletion_cost=1,
                            substitution_cost=1) -> np.float64:
        n: int = self.length_f1 + 1
        m: int = self.length_f2 + 1
        mat: NDArray[np.float64] = np.empty((2, n), dtype=np.float64)

        mat[0, 0] = 0
        for i in range(1, n):
            mat[0, i] = i

        current_line: int = 0  # tip to the last line in imaginary table
        for j in range(1, m):
            current_line = (current_line + 1) % 2
            mat[current_line, 0] = j
            for i in range(1, n):
                if self.f1[i - 1] == self.f2[j - 1]:
                    letter_cost = 0
                else:
                    letter_cost = substitution_cost
                mat[current_line, i] = min(mat[(current_line - 1) % 2,
                                               i] + deletion_cost,
                                           mat[current_line,
                                               i - 1] + insertion_cost,
                                           mat[(current_line - 1) % 2,
                                               i - 1] + letter_cost)
        return mat[current_line, -1]

    def ast_process_files(self, python_file1: str, python_file2: str) -> bool:
        try:
            with open(python_file1, 'r') as f1, open(python_file2, 'r') as f2:
                f1 = f1.read().strip()
                f2 = f2.read().strip()
        except FileNotFoundError:
            print('files not found 😕')
            return False
        # replace any strings to empty
        f1 = re.sub(r'"[\s\S]*?"', r"''", f1)
        f1 = re.sub(r"'[\s\S]*?'", r"''", f1)
        f2 = re.sub(r'"[\s\S]*?"', r"''", f2)
        f2 = re.sub(r"'[\s\S]*?'", r"''", f2)

        # remove docstrings and long comments
        f1 = re.sub(r'= """[\s\S]*?"""', r"= ''", f1)
        f1 = re.sub(r'= """[\s\S]*?"""', r"= ''", f1)
        f2 = re.sub(r"= '''[\s\S]*?'''", r"= ''", f2)
        f2 = re.sub(r"= '''[\s\S]*?'''", r"= ''", f2)

        f1 = re.sub(r'"""[\s\S]*?"""', '', f1)
        f1 = re.sub(r"'''[\s\S]*?'''", '', f1)
        f2 = re.sub(r'"""[\s\S]*?"""', '', f2)
        f2 = re.sub(r"'''[\s\S]*?'''", '', f2)

        try:
            self.f1: str = ast.dump(ast.parse(f1))
            self.f2: str = ast.dump(ast.parse(f2))
        except SyntaxError:
            print(
                'bad syntax in files(you can use - p flag'
                ' to read files without special pre-processing) 💔')
            return False

        self.length_f1: int = len(self.f1)
        self.length_f2: int = len(self.f2)
        return True

    def read(self, file1: str, file2: str) -> bool:
        try:
            with open(file1, 'r') as f1, open(file2, 'r') as f2:
                self.f1: str = f1.read().strip()
                self.f2: str = f2.read().strip()
        except FileNotFoundError:
            print('files not found 😕')
            return False
        self.length_f1: int = len(self.f1)
        self.length_f2: int = len(self.f2)
        return True

    def write(self, result_file: str) -> None:
        with open(result_file, 'w') as f:
            f.write('\n'.join(self.result))
        print(f'Success write into {result_file} 🗂️')


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="AntiPlagiat", description="Send me two txt files: first'\
        ' with list of filepaths to check for plagiat an'\
        ' second to save result of my work",
        epilog="You can also combine flags e.g. '-na' equal to '-n -a'")
    parser.add_argument('file1', type=str,
                        help="txt file of filepaths with this format:"
                        "\"file plagiat_file\" etc.")
    parser.add_argument('file2', type=str,
                        help="txt file for results."
                        " Previous data will be removed ❌")

    parser.add_argument('-p', '--pure', action='store_true',
                        help="Don't pre-process files. Without -p flag I will"
                        " use my cool ast-base pre-process technologies")
    parser.add_argument('-a', '--advanced', action='store_true',
                        help="Use Damerau–Levenshtein distance"
                        " instead of Levenshtein distance")

    parser.add_argument('-n', '--nice-view', action='store_true',
                        help="Show result in persents instead of fractions")
    parser.add_argument('-r', '--round-number', action='store', metavar='',
                        type=int, default=0, help="Round result")

    parser.add_argument('-i', '--insertion',
                        action='store', type=float,
                        metavar='', default=1.0,
                        help="Edit insertion cost. Default 1.0")
    parser.add_argument('-d', '--deletion',
                        action='store', type=float,
                        metavar='', default=1.0,
                        help="Edit deletion cost. Default 1.0")
    parser.add_argument('-s', '--substitution',
                        action='store', type=float,
                        metavar='', default=1.0,
                        help="Edit substitution cost. Default 1.0")
    parser.add_argument('-t', '--trans',
                        action='store', type=float,
                        metavar='', default=1.0,
                        help="Edit transposition cost."
                        " Default 1.0. Only for -a flag")

    args: argparse.Namespace = parser.parse_args()

    try:
        with open(args.file1, 'r') as enter:
            enter_files: list[list[str]] = list(
                map(lambda s: s.strip().split(), enter.readlines()))
    except FileNotFoundError:
        print(f'File not found: {args.file1} 🙁')
    else:
        compare_session: Compare = Compare(enter_files)
        compare_session.antiplagiat(insertion_cost=args.insertion,
                                    deletion_cost=args.deletion,
                                    substitution_cost=args.substitution,
                                    transposition_cost=args.trans,
                                    advanced=args.advanced,
                                    pure=args.pure,
                                    nice_view=args.nice_view,
                                    round_number=args.round_number)
        compare_session.write(args.file2)


if __name__ == "__main__":
    main()
