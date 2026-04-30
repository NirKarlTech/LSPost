
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import re


def legend2list(path: Path, file_name: Optional[Path | str] = None) -> list[int]:
    legend_data: list[int] = []
    in_legend = False
    headers_line = True

    full_path = path.joinpath(file_name) if file_name is not None else path

    with full_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # ----- legend -----
            if line.strip() == "{BEGIN LEGEND}":
                in_legend = True

            elif line.strip() == "{END LEGEND}":
                in_legend = False

            elif in_legend:
                if headers_line:
                    headers_line = False
                    continue

                id_search = re.search(r"\s*(\d+)\s", line)
                if id_search is not None:
                    try:
                        legend_data.append(int(id_search.group(0).strip()))
                    except ValueError:
                        print(f"id {id_search.group(0)} is not an int...")

                else:
                    print("line:", line)

    return legend_data


engNumRe = r"(\d+\.\d+E[+-]\d+)"


def remove_space_before_minus(s: str) -> str:
    """Remove spaces that appear immediately before minus signs in formatted numbers."""
    return s.replace(" -", "-")


@dataclass
class NodoutFrame:
    """
    LS-DYNA nodout ASCII parser + writer (optimized parsing + format-preserving rewrite).

    - legend: [ids(int64 array), titles(object array)]
    - df: MultiIndex DataFrame:
        index_levels=("time","id") default, or ("timestep","time","id")
      columns always include the 12 nodal fields; for ("time","id") index we keep 'timestep' as a column.

    Reconstruction:
      - Reuses original legend lines (spacing preserved).
      - Reuses original nodal header line.
      - Reuses original "nodal print out..." prefix up to "time step" for exact style.
      - Uses float format %.5E by default to match typical nodout.
    """
    df: pd.DataFrame
    legend: List[np.ndarray]

    # raw formatting captured from original file to improve rewrite fidelity
    preamble_lines: List[str]
    legend_raw_lines: List[str]          # lines between {BEGIN LEGEND}..{END LEGEND} INCLUDING markers
    block_header_prefix: str             # prefix up to the timestep field
    nodal_columns_header: str            # the "nodal point  x-disp ..." line

    COLS = [
        "x_disp", "y_disp", "z_disp",
        "x_vel",  "y_vel",  "z_vel",
        "x_accl", "y_accl", "z_accl",
        "x_coor", "y_coor", "z_coor",
    ]

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        index_levels: Tuple[str, ...] = ("time", "id"),
    ) -> "NodoutFrame":
        path = Path(path)

        # legend arrays
        legend_ids: List[int] = []
        legend_titles: List[str] = []
        legend_raw: List[str] = []

        # preamble
        preamble: List[str] = []

        # captured formatting lines
        block_header_prefix: Optional[str] = None
        nodal_columns_header: Optional[str] = None

        # data storage (fast column-wise)
        timestep_list: List[int] = []
        time_list: List[float] = []
        id_list: List[int] = []
        data_cols: List[List[float]] = [[] for _ in range(12)]

        in_legend = False
        in_block = False
        current_step: Optional[int] = None
        current_time: Optional[float] = None
        seen_first_block = False

        # precompile once
        # We match on no-space version so it works with "t i m e  s t e p"
        re_step = re.compile(r"timestep(\d+)", re.IGNORECASE)
        # Time token like 0.0000000E+00 (or D)
        re_time = re.compile(r"attime([+\-]?\d+(?:\.\d+)?[ED][+\-]?\d+)", re.IGNORECASE)

        def try_parse_step_time(line: str) -> Optional[Tuple[int, float]]:
            low = line.lower()
            nos = low.replace(" ", "")

            if "timestep" not in nos or "attime" not in nos:
                return None

            mstep = re_step.search(nos)
            mtime = re_time.search(nos)
            if not mstep or not mtime:
                return None

            step = int(mstep.group(1))
            tval = float(mtime.group(1).replace("D", "E").replace("d", "E"))

            return step, tval

        def parse_data_line_fast(line: str) -> Optional[Tuple[int, np.ndarray]]:
            s = line.lstrip()
            if not s or not s[0].isdigit():
                return None
            
            # Replace D/d with E for scientific notation
            line = line.replace("D", "E").replace("d", "E")
            
            # Handle concatenated numbers with negative signs (e.g., "0.00000E+00-2.28745E-13")
            # Insert a space before any minus/plus that follows a digit (but not at line start)
            fixed_line = ""
            for i, char in enumerate(line):
                if i > 0 and char in "+-" and line[i-1].isdigit():
                    fixed_line += " " + char
                else:
                    fixed_line += char
            
            arr = np.fromstring(fixed_line, sep=" ")
            # id + 12 floats
            if arr.size < 13:
                return None
            nid = int(arr[0])
            vals = arr[1:13].astype(np.float64, copy=False)
            return nid, vals

        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.rstrip("\n")

                # ----- legend -----
                if line.strip() == "{BEGIN LEGEND}":
                    in_legend = True
                    legend_raw.append(line)
                    continue
                if line.strip() == "{END LEGEND}":
                    in_legend = False
                    legend_raw.append(line)
                    continue
                if in_legend:
                    legend_raw.append(line)
                    parts = line.split(None, 1)
                    if parts and parts[0].lstrip("+-").isdigit():
                        try:
                            legend_ids.append(int(parts[0]))
                            legend_titles.append(parts[1].strip() if len(parts) > 1 else "")
                        except ValueError:
                            pass
                    continue

                # ----- detect new block header -----
                st = try_parse_step_time(line)
                if st is not None:
                    current_step, current_time = st
                    in_block = True
                    seen_first_block = True

                    # Capture the header prefix once (up to the timestep digits)
                    # Example original:
                    # " n o d a l ... t i m e  s t e p       1                              ( at time 0.0000000E+00 )"
                    # We'll keep everything up to the timestep field start.
                    if block_header_prefix is None:
                        # find where the timestep digits begin by locating the step as string in the original line
                        # choose the first occurrence of that exact step number token in the line
                        step_str = str(current_step)
                        idx = line.find(step_str)
                        if idx != -1:
                            block_header_prefix = line[:idx].rstrip()
                        else:
                            # fallback: use the standard prefix found in many files
                            block_header_prefix = " n o d a l   p r i n t   o u t   f o r   t i m e  s t e p"
                    continue

                # ----- preamble -----
                if not seen_first_block and line.strip() != "":
                    preamble.append(line)

                if not in_block:
                    continue

                # Capture the nodal columns header line once (preserve spacing)
                low = line.lower().strip()
                if nodal_columns_header is None and ("nodal point" in low) and ("x-disp" in low):
                    nodal_columns_header = line
                    continue

                # Skip other block header-ish lines
                if not low:
                    continue
                if low.startswith("nodal point") or ("x-disp" in low):
                    continue

                parsed = parse_data_line_fast(line)
                if parsed is None or current_step is None or current_time is None:
                    continue

                nid, vals = parsed
                timestep_list.append(current_step)
                time_list.append(float(current_time))
                id_list.append(nid)
                for j in range(12):
                    data_cols[j].append(float(vals[j]))

        if block_header_prefix is None:
            block_header_prefix = " n o d a l   p r i n t   o u t   f o r   t i m e  s t e p"
        if nodal_columns_header is None:
            nodal_columns_header = (
                " nodal point  x-disp     y-disp      z-disp      x-vel       y-vel"
                "       z-vel       x-accl      y-accl      z-accl      x-coor"
                "      y-coor      z-coor"
            )

        legend = [
            np.array(legend_ids, dtype=np.int64),
            np.array(legend_titles, dtype=object),
        ]

        # Build DataFrame
        data = {
            "timestep": np.asarray(timestep_list, dtype=np.int64),
            "time": np.asarray(time_list, dtype=np.float64),
            "id": np.asarray(id_list, dtype=np.int64),
        }
        for j, name in enumerate(cls.COLS):
            data[name] = np.asarray(data_cols[j], dtype=np.float64)

        df = pd.DataFrame(data)

        # MultiIndex
        if index_levels == ("time", "id"):
            df.set_index(["time", "id"], inplace=True)
            df = df[["timestep"] + cls.COLS]
        elif index_levels == ("timestep", "time", "id"):
            df.set_index(["timestep", "time", "id"], inplace=True)
            df = df[cls.COLS]
        else:
            raise ValueError("index_levels must be ('time','id') or ('timestep','time','id')")

        df.sort_index(inplace=True)

        return cls(
            df=df,
            legend=legend,
            preamble_lines=preamble,
            legend_raw_lines=legend_raw,
            block_header_prefix=block_header_prefix,
            nodal_columns_header=nodal_columns_header,
        )

    def to_file(self, path: str | Path, *, float_fmt: str = "%.5E") -> None:
        """
        Reconstruct nodout-like file.

        To match your original better:
          - timestep formatting uses width 8 (common in nodout)
          - time uses 7 decimals in exponent format: 0.0000000E+00
          - float fields use %.5E (default) like 0.00000E+00
          - node id field uses width 9 and two spaces after (matches your sample)
        """
        path = Path(path)

        df = self.df.copy()
        if df.index.names == ["time", "id"]:
            if "timestep" not in df.columns:
                raise ValueError("For index ('time','id'), df must include a 'timestep' column.")
            df = df.reset_index()  # time,id,timestep,...
        elif df.index.names == ["timestep", "time", "id"]:
            df = df.reset_index()
        else:
            raise ValueError(f"Unsupported index names: {df.index.names}")

        required = ["timestep", "time", "id"] + self.COLS
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        df.sort_values(["timestep", "time", "id"], inplace=True)

        with path.open("w", encoding="utf-8") as f:
            # Preamble (verbatim)
            for line in self.preamble_lines:
                f.write(line + "\n")

            # Legend (verbatim if captured; otherwise regenerate)
            if self.legend_raw_lines:
                f.write("\n")
                for line in self.legend_raw_lines:
                    f.write(line + "\n")
                f.write("\n")
            else:
                ids, titles = self.legend
                f.write("\n{BEGIN LEGEND}\n")
                f.write(" Entity #        Title\n")
                for i, nid in enumerate(ids):
                    title = titles[i] if i < len(titles) else ""
                    # approximate spacing
                    try:
                        f.write(f"{int(nid):9d} {title}\n")
                    except Exception:
                        f.write(f"{nid} {title}\n")
                f.write("{END LEGEND}\n\n")

            # Blocks
            # Use a stable group order
            for (step, t), g in df.groupby(["timestep", "time"], sort=True):
                f.write("\n\n\n")
                # Match original header style closely
                # Note: original has ONE space after "at time" and no leading space before the number
                try:
                    step_val = int(step) if isinstance(step, (int, float, str)) and str(step).isdigit() else 0
                except Exception:
                    step_val = 0
                try:
                    t_val = float(t) if isinstance(t, (int, float, str)) and t is not None else 0.0
                except Exception:
                    t_val = 0.0
                if isinstance(step_val, int):
                    step_str = f"{step_val:8d}"
                else:
                    step_str = f"{str(step_val):>8}"
                if isinstance(t_val, float):
                    t_str = f"{t_val:0.7E}"
                else:
                    t_str = str(t_val)
                f.write(
                    f"{self.block_header_prefix}{step_str}"
                    f"                              ( at time {t_str} )\n\n"
                )
                f.write(self.nodal_columns_header + "\n")

                # Data rows: id field width 9 + two spaces, then 12 floats with one space separation
                for _, r in g.iterrows():
                    nums = [float(r[c]) for c in self.COLS]
                    num_str = " ".join(float_fmt % v for v in nums)
                    num_str = remove_space_before_minus(num_str)
                    try:
                        f.write(f"{int(r['id']):9d}  {num_str}\n")
                    except Exception:
                        f.write(f"{r['id']}  {num_str}\n")

@dataclass
class EloutFrame:
    df: pd.DataFrame
    legend: List[np.ndarray]
    preamble_lines: List[str]
    legend_raw_lines: List[str]
    block_header_prefix: str

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        index_levels: Tuple[str, ...] = ("time", "id"),
    ) -> "EloutFrame":
        path = Path(path)

        legend_ids: List[int] = []
        legend_titles: List[str] = []
        legend_raw: List[str] = []

        preamble: List[str] = []
        block_header_prefix: Optional[str] = None

        # data storage
        timestep_list: List[int] = []
        time_list: List[float] = []
        id_list: List[int] = []
        materl_list: List[int] = []
        stress_cols = ["sig_xx", "sig_yy", "sig_zz", "sig_xy", "sig_yz", "sig_zx"]
        data_cols: List[List[float]] = [[] for _ in range(6)]
        yield_list: List[float] = []

        in_legend = False
        in_block = False
        current_step: Optional[int] = None
        current_time: Optional[float] = None
        current_elem_id: Optional[int] = None
        current_materl: Optional[int] = None
        seen_first_block = False

        re_step = re.compile(r"timestep(\d+)", re.IGNORECASE)
        re_time = re.compile(r"attime([+\-]?\d+(?:\.\d+)?[ED][+\-]?\d+)", re.IGNORECASE)

        def try_parse_step_time(line: str) -> Optional[Tuple[int, float]]:
            low = line.lower()
            nos = low.replace(" ", "")

            if "timestep" not in nos or "attime" not in nos:
                return None

            mstep = re_step.search(nos)
            mtime = re_time.search(nos)
            if not mstep or not mtime:
                return None

            step = int(mstep.group(1))
            tval = float(mtime.group(1).replace("D", "E").replace("d", "E"))
            return step, tval

        def parse_elem_id_line(line: str) -> Optional[Tuple[int, int]]:
            """Parse element ID and material ID from a line like '   14501-    999'"""
            s = line.strip()
            if not s or not s[0].isdigit():
                return None
            
            # Try to split on '-' or whitespace
            parts = s.replace('-', ' ').split()
            if len(parts) >= 2:
                try:
                    elem_id = int(parts[0])
                    materl = int(parts[1])
                    return elem_id, materl
                except ValueError:
                    pass
            return None

        def parse_stress_line(line: str) -> Optional[np.ndarray]:
            """Parse the stress data line"""
            line = line.replace("D", "E").replace("d", "E")
            
            # Handle concatenated numbers
            fixed_line = ""
            for i, char in enumerate(line):
                if i > 0 and char in "+-" and line[i-1].isdigit():
                    fixed_line += " " + char
                else:
                    fixed_line += char
            
            try:
                arr = np.fromstring(fixed_line, sep=" ")
            except (ValueError, TypeError):
                return None
            
            # Array structure: [ipt, sig-xx, sig-yy, sig-zz, sig-xy, sig-yz, sig-zx, yield, effsg, ...]
            # We need at least 8 values (ipt through yield)
            if arr.size < 8:
                return None
            
            # Extract: sig-xx (idx 1), sig-yy (idx 2), sig-zz (idx 3), sig-xy (idx 4), sig-yz (idx 5), sig-zx (idx 6), yield (idx 7)
            try:
                stresses = arr[1:7].astype(np.float64, copy=False)  # indices 1-6: 6 stress components
                yield_val = float(arr[7])  # index 7: yield stress
                return np.concatenate([stresses, [yield_val]])
            except (ValueError, IndexError):
                return None

        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            i = 0
            
            while i < len(lines):
                line = lines[i].rstrip("\n")

                # ----- legend -----
                if line.strip() == "{BEGIN LEGEND}":
                    in_legend = True
                    legend_raw.append(line)
                    i += 1
                    continue
                if line.strip() == "{END LEGEND}":
                    in_legend = False
                    legend_raw.append(line)
                    i += 1
                    continue
                if in_legend:
                    legend_raw.append(line)
                    parts = line.split(None, 1)
                    if parts and parts[0].lstrip("+-").isdigit():
                        try:
                            legend_ids.append(int(parts[0]))
                            legend_titles.append(parts[1].strip() if len(parts) > 1 else "")
                        except ValueError:
                            pass
                    i += 1
                    continue

                # ----- detect new block header -----
                st = try_parse_step_time(line)
                if st is not None:
                    current_step, current_time = st
                    in_block = True
                    seen_first_block = True

                    if block_header_prefix is None:
                        step_str = str(current_step)
                        idx = line.find(step_str)
                        if idx != -1:
                            block_header_prefix = line[:idx].rstrip()
                        else:
                            block_header_prefix = " e l e m e n t   s t r e s s   c a l c u l a t i o n s   f o r   t i m e  s t e p"
                    i += 1
                    continue

                # ----- preamble -----
                if not seen_first_block and line.strip() != "":
                    preamble.append(line)
                    i += 1
                    continue

                if not in_block:
                    i += 1
                    continue

                # Skip empty lines and header lines
                if not line.strip():
                    i += 1
                    continue
                
                low = line.lower().strip()
                if "element" in low or "ipt" in low or "stress" in low or "sig-" in low or "yield" in low or "effsg" in low:
                    i += 1
                    continue

                # Try to parse element ID line
                elem_info = parse_elem_id_line(line)
                if elem_info is not None:
                    current_elem_id, current_materl = elem_info
                    i += 1
                    continue

                # Try to parse stress data line
                if current_elem_id is not None and current_materl is not None:
                    stress_data = parse_stress_line(line)
                    if stress_data is not None:
                        timestep_list.append(current_step if current_step is not None else 0)
                        try:
                            time_list.append(float(current_time) if current_time is not None else 0.0)
                        except Exception:
                            time_list.append(0.0)
                        id_list.append(current_elem_id)
                        materl_list.append(current_materl)
                        for j in range(6):
                            data_cols[j].append(float(stress_data[j]))
                        yield_list.append(float(stress_data[6]))
                        
                        # Reset for next element
                        current_elem_id = None
                        current_materl = None
                
                i += 1

        if block_header_prefix is None:
            block_header_prefix = " e l e m e n t   s t r e s s   c a l c u l a t i o n s   f o r   t i m e  s t e p"

        legend = [
            np.array(legend_ids, dtype=np.int64),
            np.array(legend_titles, dtype=object),
        ]

        # Build DataFrame
        data = {
            "timestep": np.asarray(timestep_list, dtype=np.int64),
            "time": np.asarray(time_list, dtype=np.float64),
            "id": np.asarray(id_list, dtype=np.int64),
            "materl": np.asarray(materl_list, dtype=np.int64),
            "sig_xx": np.asarray(data_cols[0], dtype=np.float64),
            "sig_yy": np.asarray(data_cols[1], dtype=np.float64),
            "sig_zz": np.asarray(data_cols[2], dtype=np.float64),
            "sig_xy": np.asarray(data_cols[3], dtype=np.float64),
            "sig_yz": np.asarray(data_cols[4], dtype=np.float64),
            "sig_zx": np.asarray(data_cols[5], dtype=np.float64),
            "yield": np.asarray(yield_list, dtype=np.float64),
        }

        df = pd.DataFrame(data)

        # MultiIndex
        if index_levels == ("time", "id"):
            df.set_index(["time", "id"], inplace=True)
            df = df[["timestep", "materl", "sig_xx", "sig_yy", "sig_zz", "sig_xy", "sig_yz", "sig_zx", "yield"]]
        elif index_levels == ("timestep", "time", "id"):
            df.set_index(["timestep", "time", "id"], inplace=True)
            df = df[["materl", "sig_xx", "sig_yy", "sig_zz", "sig_xy", "sig_yz", "sig_zx", "yield"]]
        else:
            raise ValueError("index_levels must be ('time','id') or ('timestep','time','id')")

        df.sort_index(inplace=True)

        return cls(
            df=df,
            legend=legend,
            preamble_lines=preamble,
            legend_raw_lines=legend_raw,
            block_header_prefix=block_header_prefix,
        )

    def to_file(self, path: str | Path, *, float_fmt: str = "%.4E") -> None:
        """
        Reconstruct elout-like file.
        """
        path = Path(path)

        df = self.df.copy()
        if df.index.names == ["time", "id"]:
            if "timestep" not in df.columns:
                raise ValueError("For index ('time','id'), df must include a 'timestep' column.")
            df = df.reset_index()
        elif df.index.names == ["timestep", "time", "id"]:
            df = df.reset_index()
        else:
            raise ValueError(f"Unsupported index names: {df.index.names}")

        required = ["timestep", "time", "id", "materl", "sig_xx", "sig_yy", "sig_zz", "sig_xy", "sig_yz", "sig_zx", "yield"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        df.sort_values(["timestep", "time", "id"], inplace=True)

        with path.open("w", encoding="utf-8") as f:
            # Preamble
            for line in self.preamble_lines:
                f.write(line + "\n")

            f.write("\n")

            # Legend (only write if legend has actual entries)
            if self.legend_raw_lines:
                f.write("\n")
                for line in self.legend_raw_lines:
                    f.write(line + "\n")
                f.write("\n")
            elif len(self.legend[0]) > 0:  # Only write legend if there are IDs
                ids, titles = self.legend
                f.write("\n{BEGIN LEGEND}\n")
                f.write(" Entity #        Title\n")
                for i, eid in enumerate(ids):
                    title = titles[i] if i < len(titles) else ""
                    try:
                        f.write(f"{int(eid):9d} {title}\n")
                    except Exception:
                        f.write(f"{eid} {title}\n")
                f.write("{END LEGEND}\n\n")

            # Blocks
            for (step, t), g in df.groupby(["timestep", "time"], sort=True):
                f.write("\n\n")
                try:
                    step_val = int(step) if isinstance(step, (int, float, str)) and str(step).isdigit() else 0
                except Exception:
                    step_val = 0
                try:
                    t_val = float(t) if isinstance(t, (int, float, str)) and t is not None else 0.0
                except Exception:
                    t_val = 0.0
                if isinstance(step_val, int):
                    step_str = f"{step_val:9d}"
                else:
                    step_str = f"{str(step_val):>9}"
                if isinstance(t_val, float):
                    t_str = f"{t_val:0.5E}"
                else:
                    t_str = str(t_val)
                f.write(
                    f"{self.block_header_prefix}{step_str}"
                    f"   ( at time {t_str} )\n\n"
                )
                f.write(" element  materl\n")
                f.write("     ipt  stress       sig-xx      sig-yy      sig-zz      sig-xy      sig-yz      sig-zx                       yield\n")
                f.write("           state                                                                                  effsg      function\n")

                # Data rows: element ID, material ID on one line, stress data on next
                for _, r in g.iterrows():
                    elem_id = int(r['id'])
                    materl = int(r['materl'])
                    stresses = [float(r[c]) for c in ["sig_xx", "sig_yy", "sig_zz", "sig_xy", "sig_yz", "sig_zx"]]
                    yield_val = float(r['yield'])
                    
                    # Write element ID and material ID
                    f.write(f"{elem_id:8d}- {materl:6d}\n")
                    
                    # Write stress data: ipt=1, then 6 stresses, then yield, then effsg=0
                    # NO stress_state column in the actual data!
                    stress_str = "  ".join(float_fmt % v for v in stresses)
                    stress_line = f"       1           {stress_str}    {float_fmt % yield_val}    {float_fmt % 0.0}"
                    stress_line = remove_space_before_minus(stress_line)
                    f.write(stress_line + "\n")
                    f.write("\n")  # Extra blank line after data row


class Matsum:
        
    def __init__(self, path: Path, ids: list[int] = []) -> None:
        full_path = path.joinpath("matsum")
        
        if len(ids) == 0:
            ids = legend2list(full_path)

        # Parse matsum file by time block and material
        # Support format like:
        #  time = 4.9963E-02
        #  mat.#=    1             inten=   1.4452E-11
        #  ...
        times = []
        time_data = []  # list[dict[mat_id]->intensity]

        current_time = None
        current_dict = {}

        time_re = re.compile(r"time\s*=\s*([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)", re.IGNORECASE)
        mat_re = re.compile(r"mat\.#\s*=\s*(\d+).*?inten\s*=\s*([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)", re.IGNORECASE)

        with full_path.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("{") or line.startswith("Entity"):
                    continue

                m_time = time_re.search(line)
                if m_time:
                    if current_time is not None:
                        times.append(current_time)
                        time_data.append(current_dict)
                    current_time = float(m_time.group(1))
                    current_dict = {}
                    continue

                m_mat = mat_re.search(line)
                if m_mat and current_time is not None:
                    mat_id = int(m_mat.group(1))
                    inten = float(m_mat.group(2))
                    current_dict[mat_id] = inten

        if current_time is not None:
            times.append(current_time)
            time_data.append(current_dict)

        # Ensure all requested ids are present in DataFrame columns
        if len(ids) == 0:
            ids = sorted({mid for d in time_data for mid in d.keys()})
        else:
            ids = [mid for mid in ids if any(mid in d for d in time_data)]

        # Build rows in time order, with missing values filled as NaN
        rows = [[time_data[i].get(mid, np.nan) for mid in ids] for i in range(len(times))]

        self.df = pd.DataFrame.from_records(rows, index=pd.Index(times, name="time"), columns=ids)

        self.maxEnergy = self.df.max(axis=0).to_dict()


    @staticmethod
    def get_attribute(attribute_name: str, text: str, ids: list[int] = []) -> list[str]:
        values = re.findall(rf"{attribute_name}={engNumRe}", text)
        part_ids = []
        if len(ids) != 0:
            part_ids = re.findall(rf"mat\.\#=(\d+)", text)
            values = [values[i] for i, id in enumerate(part_ids) if int(id) in ids]
        return values


class KeyFileData:
    @property
    def area(self):
        # Fallback: return 1.0 if area is not set elsewhere
        return getattr(self, '_area', 1.0)

    def get_faces(self):
        """Stub: Return a default face structure for cohesive elements (8 nodes)."""
        # This is a placeholder. Real implementation should extract faces from element connectivity.
        # For now, return two faces of 4 nodes each (bottom and top) as tuples of node indices.
        # If no elements, return dummy faces.
        if hasattr(self, 'elements') and self.elements:
            # Get first element's node list
            first_elem = next(iter(self.elements.values()))
            nodes = first_elem.get('nodes', [])
            if len(nodes) == 8:
                return [tuple(nodes[:4]), tuple(nodes[4:])]
            elif len(nodes) == 4:
                return [tuple(nodes)]
        # Fallback: dummy faces
        return [(1, 2, 3, 4), (5, 6, 7, 8)]

    def __init__(self, filepath=None):
        self.nodes = {}
        self.elements = {}
        self.solid_sets = {}
        self.gn_curves = {}
        self.initial_node_coords = {}
        self.node_data = None
        self.stress_data = None
        # If a file is provided, parse it
        if filepath is not None:
            self._parse_file(filepath)

    def get_gn_curves(self):
        """Return dict of G-N curves parsed from the keyfile (lcid -> {'title', 'data'})."""
        return getattr(self, 'gn_curves', {})

    def _parse_gn_curves(self, lines):
        """Parse all *DEFINE_CURVE_TITLE blocks with title 'G-N'. Return dict: lcid -> {'title':..., 'data':[(N,G),...]}."""
        gn_curves = {}
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('*DEFINE_CURVE_TITLE'):
                # Next line should be the title
                title = lines[i+1].strip() if i+1 < len(lines) else ''
                if title.upper() == 'G-N':
                    # Skip header lines to LCID line
                    lcid_line = lines[i+3] if i+3 < len(lines) else ''
                    lcid_parts = lcid_line.split()
                    lcid = int(lcid_parts[0]) if lcid_parts and lcid_parts[0].isdigit() else None
                    # Data lines start at i+5
                    data = []
                    j = i+5
                    while j < len(lines):
                        dline = lines[j].strip()
                        if dline.startswith('*END') or dline.startswith('*'):
                            break
                        if dline and not dline.startswith('$'):
                            parts = dline.split()
                            if len(parts) >= 2:
                                try:
                                    n_val = float(parts[0])
                                    g_val = float(parts[1])
                                    data.append((n_val, g_val))
                                except Exception:
                                    pass
                        j += 1
                    if lcid is not None and data:
                        gn_curves[lcid] = {'title': title, 'data': data}
                    i = j
                    continue
            i += 1
        return gn_curves

    def _parse_file(self, filepath: Path) -> None:
        """Parse the LS-DYNA keyword file."""
        with filepath.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # Parse G-N curves before other parsing
        self.gn_curves = self._parse_gn_curves(lines)
        current_keyword = None
        current_set_name = None
        reading_set_elements = False
        current_set_elements = []
        for i, line in enumerate(lines):
            line = line.strip()
            # Skip empty lines and comments (but not during set reading)
            if not line or (line.startswith('$') and not reading_set_elements):
                continue
            # Check for keyword (starts with *)
            if line.startswith('*'):
                # Save any previous set being read
                if reading_set_elements and current_set_name:
                    self.solid_sets[current_set_name] = current_set_elements
                    reading_set_elements = False
                    current_set_elements = []
                    current_set_name = None
                # Accept any variant (e.g., *NODE, *NODE_TITLE, etc.)
                if line.upper().startswith('*ELEMENT_SOLID'):
                    current_keyword = '*ELEMENT_SOLID'
                elif line.upper().startswith('*NODE'):
                    current_keyword = '*NODE'
                elif line.upper().startswith('*SET_SOLID'):
                    current_keyword = '*SET_SOLID'
                    # Check if it has _TITLE variant
                    if '_TITLE' in line:
                        # Next non-comment line will be the title
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if not next_line.startswith('$') and not next_line.startswith('*'):
                                current_set_name = next_line
                    else:
                        current_set_name = f"Set_{len(self.solid_sets) + 1}"
                else:
                    current_keyword = line
                continue
            # Parse NODE data
            if current_keyword == '*NODE':
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        nid = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        self.nodes[nid] = {'x': x, 'y': y, 'z': z}
                    except ValueError:
                        continue
            # Parse ELEMENT_SOLID data
            elif current_keyword == '*ELEMENT_SOLID':
                parts = line.split()
                if len(parts) >= 10:  # eid, pid, n1-n8
                    try:
                        eid = int(parts[0])
                        pid = int(parts[1])
                        node_ids = [int(parts[i]) for i in range(2, 10)]
                        self.elements[eid] = {'pid': pid, 'nodes': node_ids}
                    except ValueError:
                        continue
            # Parse SET_SOLID data
            elif current_keyword == '*SET_SOLID':
                # Skip comment lines
                if line.startswith('$'):
                    continue
                # Skip the sid/solver line (has 'MECH' or starts with digit and contains 'MECH')
                if 'MECH' in line:
                    reading_set_elements = True
                    continue
                # Read element IDs if we're in the reading phase
                if reading_set_elements:
                    parts = line.split()
                    for part in parts:
                        try:
                            eid = int(part)
                            if eid > 0:  # 0 marks end of list
                                current_set_elements.append(eid)
                        except ValueError:
                            continue
        # Save last set if file ends while reading
        if reading_set_elements and current_set_name:
            self.solid_sets[current_set_name] = current_set_elements
        print(f"[KeyFileData] Parsed {len(self.nodes)} nodes, {len(self.elements)} elements, {len(self.solid_sets)} sets from keyfile.")


    def get_element_coordinates(self, element_id: int) -> list[dict]:
        """
        Get the coordinates of all nodes for a given element.
        Args:
            element_id: The element ID
        Returns:
            List of node coordinate dicts [{'x': x, 'y': y, 'z': z}, ...]
        """
        if element_id not in self.elements:
            raise KeyError(f"Element {element_id} not found")
        node_ids = self.elements[element_id]['nodes']
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def print_summary(self) -> None:
        """
        Print a summary of the parsed data.
        """
        print("=" * 50)
        print("LS-DYNA Keyword File Parser Results")
        print("=" * 50)
        print(f"\nTotal Nodes: {len(self.nodes)}")
        print("\nNode Coordinates:")
        print("-" * 50)
        for nid, coords in self.nodes.items():
            print(f"  Node {nid}: x={coords['x']:.4f}, y={coords['y']:.4f}, z={coords['z']:.4f}")
        print(f"\nTotal Solid Elements: {len(self.elements)}")

    def get_face_with_lowest_node_sum(self) -> Tuple[Tuple[int, int, int, int], int]:
        """
        Find the face with the lowest sum of node IDs.
        
        Returns:
            Tuple of (face_node_ids, sum_of_node_ids)
        """
        faces = []  # get_faces is not defined for KeyFileData; provide empty list or implement as needed
        face_sums = [(face, sum(face)) for face in faces]
        min_face = min(face_sums, key=lambda x: x[1])
        return min_face

    @staticmethod
    def _calculate_quad_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
        """
        Calculate the area of a quadrilateral given 4 corner points.
        
        Uses the cross product of diagonals method:
        Area = 0.5 * |AC x BD|
        
        Args:
            p1, p2, p3, p4: Corner points as numpy arrays [x, y, z]
            
        Returns:
            Area of the quadrilateral
        """
        # Diagonals
        diagonal_ac = p3 - p1
        diagonal_bd = p4 - p2
        
        # Cross product
        cross = np.cross(diagonal_ac, diagonal_bd)
        
        # Area is half the magnitude of the cross product
        area = 0.5 * np.linalg.norm(cross)
        return float(area)

    def calculate_face_area(self, face_node_ids: Tuple[int, int, int, int]) -> float:
        """
        Calculate the area of a face given its 4 node IDs.
        
        Args:
            face_node_ids: Tuple of 4 node IDs defining the face
            
        Returns:
            Area of the face
        """
        # Get coordinates for each node
        coords = []
        for nid in face_node_ids:
            if nid not in self.initial_node_coords:
                raise KeyError(f"Node {nid} not found in element coordinates")
            c = self.initial_node_coords[nid]
            coords.append(np.array([c['x'], c['y'], c['z']]))
        
        return self._calculate_quad_area(coords[0], coords[1], coords[2], coords[3])

    def get_min_node_sum_face_area(self) -> Tuple[float, Tuple[int, int, int, int]]:
        """
        Get the area of the face with the lowest sum of node IDs.
        
        Returns:
            Tuple of (area, face_node_ids)
        """
        face, node_sum = self.get_face_with_lowest_node_sum()
        area = self.calculate_face_area(face)
        return area, face

    def get_face_displacement(self, face_node_ids: Tuple[int, int, int, int]) -> pd.DataFrame:
        """
        Get the average displacement of a face over time.
        
        Calculates the mean displacement of all nodes on the face.
        
        Args:
            face_node_ids: Tuple of 4 node IDs defining the face
            
        Returns:
            DataFrame with average x_disp, y_disp, z_disp, and magnitude columns
        """
        if self.node_data is None:
            raise ValueError("Node data not loaded")
        
        # Get displacement data for each node on the face
        face_displacements = []
        for nid in face_node_ids:
            if nid in self.node_data.index.get_level_values('id'):
                disp = self.node_data.xs(nid, level='id')[['x_disp', 'y_disp', 'z_disp']]
                face_displacements.append(disp)
        
        if not face_displacements:
            raise ValueError(f"No displacement data found for face nodes {face_node_ids}")
        
        # Calculate average displacement across all face nodes
        avg_disp = sum(face_displacements) / len(face_displacements)
        
        # Calculate displacement magnitude
        # If avg_disp is a float or int, wrap in DataFrame
        if isinstance(avg_disp, (float, int)):
            return pd.DataFrame({'magnitude': [avg_disp]})
        if isinstance(avg_disp, dict):
            # Convert dict to DataFrame
            avg_disp = pd.DataFrame([avg_disp])
        avg_disp['magnitude'] = np.sqrt(
            avg_disp['x_disp']**2 + avg_disp['y_disp']**2 + avg_disp['z_disp']**2
        )
        return avg_disp

    def get_face_normal_direction(self, face_node_ids: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Calculate the outward normal direction of a face.
        
        Args:
            face_node_ids: Tuple of 4 node IDs defining the face
            
        Returns:
            Unit normal vector as numpy array [nx, ny, nz]
        """
        # Get coordinates for each node
        coords = []
        for nid in face_node_ids:
            if nid not in self.initial_node_coords:
                raise KeyError(f"Node {nid} not found in element coordinates")
            c = self.initial_node_coords[nid]
            coords.append(np.array([c['x'], c['y'], c['z']]))
        
        # Calculate two edge vectors
        v1 = coords[1] - coords[0]  # edge 1-2
        v2 = coords[3] - coords[0]  # edge 1-4
        
        # Cross product gives normal
        normal = np.cross(v1, v2)
        
        # Normalize
        magnitude = np.linalg.norm(normal)
        if magnitude > 1e-10:
            normal = normal / magnitude
        
        return normal

    def get_face_normal_displacement(self, face_node_ids: Tuple[int, int, int, int]) -> pd.Series:
        """
        Get the displacement component normal to the face over time.
        
        Args:
            face_node_ids: Tuple of 4 node IDs defining the face
            
        Returns:
            Series with normal displacement at each time step
        """
        # Get face average displacement
        avg_disp = self.get_face_displacement(face_node_ids)
        
        # Get face normal direction
        normal = self.get_face_normal_direction(face_node_ids)
        
        # Project displacement onto normal direction
        normal_disp = (
            avg_disp['x_disp'] * normal[0] +
            avg_disp['y_disp'] * normal[1] +
            avg_disp['z_disp'] * normal[2]
        )
        normal_disp.name = 'normal_disp'
        
        return normal_disp

    def get_normal_stress_for_face(self, face_node_ids: Tuple[int, int, int, int]) -> pd.Series:
        """
        Get the normal stress component for a given face.
        
        For face (n1, n2, n3, n4) which is typically the bottom face in xy-plane,
        the normal stress is sig_zz.
        
        Args:
            face_node_ids: Tuple of 4 node IDs defining the face
            
        Returns:
            Series with normal stress at each time step
        """
        if self.stress_data is None:
            raise ValueError("Stress data not loaded")
        
        # Get face normal direction
        normal = self.get_face_normal_direction(face_node_ids)
        
        # Determine which stress component is the normal stress based on the face normal
        # For a face with normal predominantly in z-direction, use sig_zz
        # For x-direction use sig_xx, for y-direction use sig_yy
        abs_normal = np.abs(normal)
        
        if abs_normal[2] >= abs_normal[0] and abs_normal[2] >= abs_normal[1]:
            # Normal is primarily in z-direction
            stress_col = 'sig_zz'
        elif abs_normal[0] >= abs_normal[1]:
            # Normal is primarily in x-direction
            stress_col = 'sig_xx'
        else:
            # Normal is primarily in y-direction
            stress_col = 'sig_yy'
        
        return self.stress_data[stress_col]

    def get_cohesive_separation(self) -> pd.DataFrame:
        """
        Calculate the cohesive separation (relative displacement) between 
        the top and bottom faces of a cohesive element.
        
        For a cohesive element with 8 nodes:
        - Bottom face: nodes 0-3 (n1, n2, n3, n4)
        - Top face: nodes 4-7 (n5, n6, n7, n8)
        
        The separation delta = u_top - u_bottom for corresponding node pairs:
        - n1 <-> n5, n2 <-> n6, n3 <-> n7, n4 <-> n8
        
        Returns:
            DataFrame with separation components (x, y, z) and magnitude
        """
        if self.node_data is None:
            raise ValueError("Node data not loaded")
        
        faces = self.get_faces()
        bottom_face = faces[0]  # (n1, n2, n3, n4)
        top_face = faces[1]     # (n5, n6, n7, n8)
        
        # Node correspondence for cohesive element:
        # n1<->n5, n2<->n6, n3<->n7, n4<->n8
        node_pairs = list(zip(bottom_face, top_face))
        
        separations = []
        available_nodes = self.node_data.index.get_level_values('id').unique()
        
        for bottom_nid, top_nid in node_pairs:
            if bottom_nid in available_nodes and top_nid in available_nodes:
                # Get displacements
                bottom_disp = self.node_data.xs(bottom_nid, level='id')[['x_disp', 'y_disp', 'z_disp']]
                top_disp = self.node_data.xs(top_nid, level='id')[['x_disp', 'y_disp', 'z_disp']]
                
                # Relative displacement (separation)
                sep = top_disp - bottom_disp
                sep.columns = ['x_sep', 'y_sep', 'z_sep']
                separations.append(sep)
            elif top_nid in available_nodes:
                # Only top node available - assume bottom is fixed
                top_disp = self.node_data.xs(top_nid, level='id')[['x_disp', 'y_disp', 'z_disp']]
                sep = top_disp.copy()
                sep.columns = ['x_sep', 'y_sep', 'z_sep']
                separations.append(sep)
            elif bottom_nid in available_nodes:
                # Only bottom node available - use negative of its displacement
                bottom_disp = self.node_data.xs(bottom_nid, level='id')[['x_disp', 'y_disp', 'z_disp']]
                sep = -bottom_disp
                sep.columns = ['x_sep', 'y_sep', 'z_sep']
                separations.append(sep)
        
        if not separations:
            raise ValueError("No separation data available for cohesive element")
        
        # Average separation across all node pairs
        avg_sep = sum(separations) / len(separations)
        
        # Calculate magnitude
        # If avg_sep is a float or int, wrap in DataFrame
        if isinstance(avg_sep, (float, int)):
            return pd.DataFrame({'magnitude': [avg_sep]})
        if isinstance(avg_sep, dict):
            avg_sep = pd.DataFrame([avg_sep])
        avg_sep['magnitude'] = np.sqrt(
            avg_sep['x_sep']**2 + avg_sep['y_sep']**2 + avg_sep['z_sep']**2
        )
        return avg_sep

    def get_cohesive_normal_separation(self) -> pd.Series:
        """
        Get the normal component of cohesive separation (Mode I opening).
        
        Returns:
            Series with normal separation at each time step
        """
        avg_sep = self.get_cohesive_separation()
        
        # Get face normal (use bottom face)
        bottom_face = self.get_faces()[0]
        normal = self.get_face_normal_direction(bottom_face)
        
        # Project separation onto normal
        normal_sep = (
            avg_sep['x_sep'] * normal[0] +
            avg_sep['y_sep'] * normal[1] +
            avg_sep['z_sep'] * normal[2]
        )
        normal_sep.name = 'normal_separation'
        
        return normal_sep

    def calculate_Gc_by_integration(
        self, 
        use_cohesive_separation: bool = True,
        mode: str = "I",
    ) -> Tuple[pd.DataFrame, float]:
        """Calculate G_c (energy release rate) via traction-separation integration.

        G_c = integral(sigma(delta) d(delta))

        This method supports Mode I (opening), Mode II (in-plane shear), and mixed
        mode (Mode I + Mode II) calculations.

        Args:
            use_cohesive_separation: If True, use relative displacement between top
                and bottom faces. If False, use displacement of a single face.
            mode: One of "I", "II", or "C" (mixed). "C" returns G_c = G_I + G_II.

        Returns:
            A tuple (DataFrame with time series data, G_c value).

        Notes:
            - For Mode I: uses normal traction (sigma_n) vs normal separation (delta_n).
            - For Mode II: uses shear traction magnitude vs tangential separation magnitude.
            - For mixed mode (C): returns the sum of Mode I and Mode II energies.
        """
        mode = mode.upper()
        if mode not in {"I", "II", "C"}:
            raise ValueError("Mode must be one of 'I', 'II', or 'C'")

        # Get base separation and stress components
        bottom_face = self.get_faces()[0]
        normal = self.get_face_normal_direction(bottom_face)

        # Separation (vector) and its magnitude
        if use_cohesive_separation:
            sep_df = self.get_cohesive_separation()
        else:
            # Fallback: use face displacement instead of cohesive separation
            sep_df = self.get_face_displacement(bottom_face)
            sep_df = sep_df.rename(columns={
                'x_disp': 'x_sep',
                'y_disp': 'y_sep',
                'z_disp': 'z_sep',
            })
            sep_df['magnitude'] = np.sqrt(
                sep_df['x_sep'] ** 2 + sep_df['y_sep'] ** 2 + sep_df['z_sep'] ** 2
            )

        # Normal separation (delta_n)
        normal_sep = (
            sep_df['x_sep'] * normal[0]
            + sep_df['y_sep'] * normal[1]
            + sep_df['z_sep'] * normal[2]
        )
        normal_sep.name = 'normal_separation'

        # Tangential separation (delta_t) magnitude
        tangential_sq = (
            sep_df['x_sep'] ** 2 + sep_df['y_sep'] ** 2 + sep_df['z_sep'] ** 2
            - normal_sep ** 2
        )
        tangential_sq = tangential_sq.clip(lower=0)
        tangential_sep = np.sqrt(tangential_sq)
        tangential_sep.name = 'tangential_separation'

        # Tractions
        normal_traction = self.get_normal_stress_for_face(bottom_face)

        # Shear traction magnitude (Mode II):
        # Per user: always use sig_xx as the shear stress for Mode II (ignore sig_zx, sig_xy, sig_yz)
        # For backward compatibility, keep the old logic for mixed mode, but for Mode II use only sig_xx.
        if self.stress_data is not None:
            if mode == "II":
                tau1 = self.stress_data['sig_xx']
                tau2 = 0 * tau1  # Only sig_xx used
            else:
                if np.abs(normal[2]) >= np.abs(normal[0]) and np.abs(normal[2]) >= np.abs(normal[1]):
                    # Face normal is mostly z -> shear stresses are zx and yz
                    tau1 = self.stress_data['sig_zx']
                    tau2 = self.stress_data['sig_yz']
                elif np.abs(normal[0]) >= np.abs(normal[1]):
                    # Face normal is mostly x -> shear stresses are xy and zx
                    tau1 = self.stress_data['sig_xy']
                    tau2 = self.stress_data['sig_zx']
                else:
                    # Face normal is mostly y -> shear stresses are xy and yz
                    tau1 = self.stress_data['sig_xy']
                    tau2 = self.stress_data['sig_yz']
        else:
            tau1 = 0
            tau2 = 0

        shear_traction = np.sqrt(tau1 ** 2 + tau2 ** 2)
        shear_traction.name = 'shear_traction'

        # Choose which traction and separation to use based on mode
        if mode == "I":
            sep_series = normal_sep.abs()
            traction_series = normal_traction.abs()
        elif mode == "II":
            sep_series = tangential_sep.abs()
            traction_series = shear_traction.abs()
        else:  # Mixed mode: compute both and add energies
            # We'll compute Mode I and Mode II separately then sum energies
            df_i, gi = self.calculate_Gc_by_integration(use_cohesive_separation=use_cohesive_separation, mode="I")
            df_ii, gii = self.calculate_Gc_by_integration(use_cohesive_separation=use_cohesive_separation, mode="II")

            # Merge on time using outer join to preserve both
            result = pd.merge(
                df_i[['separation', 'traction', 'G_cumulative']],
                df_ii[['separation', 'traction', 'G_cumulative']],
                left_index=True,
                right_index=True,
                how='outer',
                suffixes=('I', 'II'),
            )

            # Fill forward missing values to allow cumulative sum
            result = result.sort_index().ffill().fillna(0)

            # Mixed (effective) separation and traction magnitudes
            result['separation_mixed'] = np.sqrt(
                result['separationI'] ** 2 + result['separationII'] ** 2
            )
            result['traction_mixed'] = np.sqrt(
                result['tractionI'] ** 2 + result['tractionII'] ** 2
            )

            # Sum cumulative energies
            result['G_cumulative'] = result['G_cumulativeI'] + result['G_cumulativeII']
            result['mode'] = 'Mixed'

            return result, result['G_cumulative'].iloc[-1]

        # Convert to DataFrames for merge_asof (tolerance-based join)
        sep_df2 = pd.DataFrame({'separation': sep_series}).reset_index()
        stress_df2 = pd.DataFrame({'traction': traction_series}).reset_index()

        # Rename index column if needed
        if 'time' not in sep_df2.columns:
            sep_df2 = sep_df2.rename(columns={'index': 'time'})
        if 'time' not in stress_df2.columns:
            stress_df2 = stress_df2.rename(columns={'index': 'time'})

        # Sort by time
        sep_df2 = sep_df2.sort_values('time')
        stress_df2 = stress_df2.sort_values('time')

        # Use merge_asof for tolerance-based matching
        # If 'time' is numeric, tolerance can be float; if datetime, use pd.Timedelta
        tolerance = 1e-4
        # Fix: handle pandas ExtensionDtype for time column
        time_dtype = sep_df2['time'].dtype
        if hasattr(time_dtype, 'kind') and time_dtype.kind == 'M':
            tolerance = pd.Timedelta(microseconds=100)
        result = pd.merge_asof(
            sep_df2,
            stress_df2,
            on='time',
            direction='nearest',
            tolerance=tolerance if isinstance(tolerance, (int, type(None), pd.Timedelta)) else None,
        )

        # Drop rows where no match was found
        result = result.dropna(subset=['traction']).set_index('time').sort_index()

        # Calculate incremental separation: Deltadelta_i = delta_i - delta_{i-1}
        result['delta_separation'] = result['separation'].diff().fillna(0)

        # For proper integration, use the average traction over each increment
        # sigma_avg = (sigma_i + sigma_{i-1}) / 2  (trapezoidal rule)
        result['traction_avg'] = (result['traction'] + result['traction'].shift(1).fillna(0)) / 2

        # Incremental energy: dG = sigma_avg * Deltadelta
        result['dG'] = result['traction_avg'] * result['delta_separation'].abs()

        # Cumulative G (running integral)
        result['G_cumulative'] = result['dG'].cumsum()

        # G_c is the total integral (final cumulative value)
        Gc = result['G_cumulative'].iloc[-1]

        return result, Gc

    def calculate_Gc_by_energy(self, face_node_ids: Tuple[int, int, int, int]) -> float:
        """Calculate G_c using the energy method: G_c = max_internal_energy / area."""
        area = self.calculate_face_area(face_node_ids)
        if area < 1e-10:
            raise ValueError(f"Face area is too small: {area}")
        
        # Note: Internal energy needs to be accessed from Part level
        # This method is a placeholder - actual implementation needs Part reference
        raise NotImplementedError(
            "Use Part.calculate_Gc_by_energy() or Model.calculate_element_Gc_by_energy() instead"
        )

    def get_traction_separation_data(
        self, 
        use_cohesive_separation: bool = True,
        mode: str = "I"
    ) -> pd.DataFrame:
        """Get the traction-separation data for plotting or further analysis.

        Args:
            use_cohesive_separation: If True, calculate separation as relative
                displacement between top and bottom faces.
            mode: One of "I", "II", or "C" (mixed).

        Returns:
            DataFrame with separation, traction, and cumulative G columns.
        """
        result, _ = self.calculate_Gc_by_integration(use_cohesive_separation, mode=mode)
        return result


    def calculate_internal_energy(self, use_cohesive_separation: bool = True, mode: str = "C") -> pd.Series:
        """
        Calculate the internal energy time series for this element using traction and displacement.
        
        This computes the cumulative energy: integral sigma(delta) d(delta) at each time point.
        
        Args:
            use_cohesive_separation: If True, use cohesive separation (relative displacement).
                                    If False, use single face displacement.
            mode: "I" for Mode I, "II" for Mode II, "C" for mixed (default).
        
        Returns:
            Series with cumulative internal energy at each time point (J)
        """
        result, _ = self.calculate_Gc_by_integration(use_cohesive_separation, mode=mode)
        # G_cumulative is computed per unit area; convert to total energy by multiplying element area.
        return result['G_cumulative'] * self.area


class Element:
    """
    Represents a single finite element with its data and analysis methods.
    
    Stores per-element attributes (eid, pid, node_ids, node_data, stress_data,
    initial_node_coords, area) and provides analysis methods for cohesive elements
    (traction-separation, energy release rate, etc.).
    """

    def __init__(self, eid: int, pid: int, node_ids: List[int],
                 initial_node_coords: Optional[dict] = None,
                 node_data: Optional[pd.DataFrame] = None,
                 stress_data: Optional[pd.DataFrame] = None,
                 area: float = 1.0):
        self.eid = eid
        self.pid = pid
        self.node_ids = node_ids
        self.initial_node_coords = initial_node_coords or {}
        self.node_data = node_data
        self.stress_data = stress_data
        self._area = area

        # Auto-compute area from bottom face if coordinates are available
        if self.initial_node_coords and area == 1.0:
            try:
                bottom_face = self.get_faces()[0]
                if all(nid in self.initial_node_coords for nid in bottom_face):
                    self._area = self.calculate_face_area(bottom_face)
            except Exception:
                pass  # Keep default if computation fails

    @property
    def area(self) -> float:
        return self._area

    @area.setter
    def area(self, value: float):
        self._area = value

    def get_faces(self) -> List[Tuple]:
        """Return faces of a solid element based on node connectivity."""
        nodes = self.node_ids
        if len(nodes) == 8:
            return [tuple(nodes[:4]), tuple(nodes[4:])]
        elif len(nodes) == 4:
            return [tuple(nodes)]
        return [(1, 2, 3, 4), (5, 6, 7, 8)]

    def get_node_displacement(self, node_id: int) -> Optional[pd.DataFrame]:
        """Get displacement time-history for a specific node of this element."""
        if self.node_data is None:
            return None
        available = self.node_data.index.get_level_values('id').unique()
        if node_id not in available:
            return None
        result = self.node_data.xs(node_id, level='id')
        if isinstance(result, pd.Series):
            return result.to_frame().T
        return result

    def get_face_with_lowest_node_sum(self) -> Tuple[Tuple[int, ...], int]:
        faces = self.get_faces()
        face_sums = [(face, sum(face)) for face in faces]
        return min(face_sums, key=lambda x: x[1])

    @staticmethod
    def _calculate_quad_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
        diagonal_ac = p3 - p1
        diagonal_bd = p4 - p2
        cross = np.cross(diagonal_ac, diagonal_bd)
        return float(0.5 * np.linalg.norm(cross))

    def calculate_face_area(self, face_node_ids: Tuple[int, ...]) -> float:
        coords = []
        for nid in face_node_ids:
            if nid not in self.initial_node_coords:
                raise KeyError(f"Node {nid} not found in element coordinates")
            c = self.initial_node_coords[nid]
            coords.append(np.array([c['x'], c['y'], c['z']]))
        return self._calculate_quad_area(coords[0], coords[1], coords[2], coords[3])

    def get_min_node_sum_face_area(self) -> Tuple[float, Tuple[int, ...]]:
        face, _ = self.get_face_with_lowest_node_sum()
        area = self.calculate_face_area(face)
        return area, face

    def get_face_displacement(self, face_node_ids: Tuple[int, ...]) -> pd.DataFrame:
        if self.node_data is None:
            raise ValueError("Node data not loaded")
        face_displacements = []
        for nid in face_node_ids:
            if nid in self.node_data.index.get_level_values('id'):
                disp = self.node_data.xs(nid, level='id')[['x_disp', 'y_disp', 'z_disp']]
                face_displacements.append(disp)
        if not face_displacements:
            raise ValueError(f"No displacement data found for face nodes {face_node_ids}")
        avg_disp = sum(face_displacements) / len(face_displacements)
        if isinstance(avg_disp, (float, int)):
            return pd.DataFrame({'magnitude': [avg_disp]})
        if isinstance(avg_disp, dict):
            avg_disp = pd.DataFrame([avg_disp])
        avg_disp['magnitude'] = np.sqrt(
            avg_disp['x_disp']**2 + avg_disp['y_disp']**2 + avg_disp['z_disp']**2
        )
        return avg_disp

    def get_face_normal_direction(self, face_node_ids: Tuple[int, ...]) -> np.ndarray:
        coords = []
        for nid in face_node_ids:
            if nid not in self.initial_node_coords:
                raise KeyError(f"Node {nid} not found in element coordinates")
            c = self.initial_node_coords[nid]
            coords.append(np.array([c['x'], c['y'], c['z']]))
        v1 = coords[1] - coords[0]
        v2 = coords[3] - coords[0]
        normal = np.cross(v1, v2)
        magnitude = np.linalg.norm(normal)
        if magnitude > 1e-10:
            normal = normal / magnitude
        return normal

    def get_face_normal_displacement(self, face_node_ids: Tuple[int, ...]) -> pd.Series:
        avg_disp = self.get_face_displacement(face_node_ids)
        normal = self.get_face_normal_direction(face_node_ids)
        normal_disp = (
            avg_disp['x_disp'] * normal[0] +
            avg_disp['y_disp'] * normal[1] +
            avg_disp['z_disp'] * normal[2]
        )
        normal_disp.name = 'normal_disp'
        return normal_disp

    def get_normal_stress_for_face(self, face_node_ids: Tuple[int, ...]) -> pd.Series:
        if self.stress_data is None:
            raise ValueError("Stress data not loaded")
        normal = self.get_face_normal_direction(face_node_ids)
        abs_normal = np.abs(normal)
        if abs_normal[2] >= abs_normal[0] and abs_normal[2] >= abs_normal[1]:
            stress_col = 'sig_zz'
        elif abs_normal[0] >= abs_normal[1]:
            stress_col = 'sig_xx'
        else:
            stress_col = 'sig_yy'
        return self.stress_data[stress_col]

    def get_cohesive_separation(self) -> pd.DataFrame:
        if self.node_data is None:
            raise ValueError("Node data not loaded")
        faces = self.get_faces()
        bottom_face = faces[0]
        top_face = faces[1]
        node_pairs = list(zip(bottom_face, top_face))
        separations = []
        available_nodes = self.node_data.index.get_level_values('id').unique()
        for bottom_nid, top_nid in node_pairs:
            if bottom_nid in available_nodes and top_nid in available_nodes:
                bottom_disp = self.node_data.xs(bottom_nid, level='id')[['x_disp', 'y_disp', 'z_disp']]
                top_disp = self.node_data.xs(top_nid, level='id')[['x_disp', 'y_disp', 'z_disp']]
                sep = top_disp - bottom_disp
                sep.columns = ['x_sep', 'y_sep', 'z_sep']
                separations.append(sep)
            elif top_nid in available_nodes:
                top_disp = self.node_data.xs(top_nid, level='id')[['x_disp', 'y_disp', 'z_disp']]
                sep = top_disp.copy()
                sep.columns = ['x_sep', 'y_sep', 'z_sep']
                separations.append(sep)
            elif bottom_nid in available_nodes:
                bottom_disp = self.node_data.xs(bottom_nid, level='id')[['x_disp', 'y_disp', 'z_disp']]
                sep = -bottom_disp
                sep.columns = ['x_sep', 'y_sep', 'z_sep']
                separations.append(sep)
        if not separations:
            raise ValueError("No separation data available for cohesive element")
        avg_sep = sum(separations) / len(separations)
        if isinstance(avg_sep, (float, int)):
            return pd.DataFrame({'magnitude': [avg_sep]})
        if isinstance(avg_sep, dict):
            avg_sep = pd.DataFrame([avg_sep])
        avg_sep['magnitude'] = np.sqrt(
            avg_sep['x_sep']**2 + avg_sep['y_sep']**2 + avg_sep['z_sep']**2
        )
        return avg_sep

    def get_cohesive_normal_separation(self) -> pd.Series:
        avg_sep = self.get_cohesive_separation()
        bottom_face = self.get_faces()[0]
        normal = self.get_face_normal_direction(bottom_face)
        normal_sep = (
            avg_sep['x_sep'] * normal[0] +
            avg_sep['y_sep'] * normal[1] +
            avg_sep['z_sep'] * normal[2]
        )
        normal_sep.name = 'normal_separation'
        return normal_sep

    def calculate_Gc_by_integration(
        self,
        use_cohesive_separation: bool = True,
        mode: str = "I",
    ) -> Tuple[pd.DataFrame, float]:
        """Calculate G_c (energy release rate) via traction-separation integration."""
        mode = mode.upper()
        if mode not in {"I", "II", "C"}:
            raise ValueError("Mode must be one of 'I', 'II', or 'C'")

        bottom_face = self.get_faces()[0]
        normal = self.get_face_normal_direction(bottom_face)

        if use_cohesive_separation:
            sep_df = self.get_cohesive_separation()
        else:
            sep_df = self.get_face_displacement(bottom_face)
            sep_df = sep_df.rename(columns={
                'x_disp': 'x_sep', 'y_disp': 'y_sep', 'z_disp': 'z_sep',
            })
            sep_df['magnitude'] = np.sqrt(
                sep_df['x_sep'] ** 2 + sep_df['y_sep'] ** 2 + sep_df['z_sep'] ** 2
            )

        normal_sep = (
            sep_df['x_sep'] * normal[0]
            + sep_df['y_sep'] * normal[1]
            + sep_df['z_sep'] * normal[2]
        )
        normal_sep.name = 'normal_separation'

        tangential_sq = (
            sep_df['x_sep'] ** 2 + sep_df['y_sep'] ** 2 + sep_df['z_sep'] ** 2
            - normal_sep ** 2
        )
        tangential_sq = tangential_sq.clip(lower=0)
        tangential_sep = np.sqrt(tangential_sq)
        tangential_sep.name = 'tangential_separation'

        normal_traction = self.get_normal_stress_for_face(bottom_face)

        if self.stress_data is not None:
            if mode == "II":
                tau1 = self.stress_data['sig_xx']
                tau2 = 0 * tau1
            else:
                if np.abs(normal[2]) >= np.abs(normal[0]) and np.abs(normal[2]) >= np.abs(normal[1]):
                    tau1 = self.stress_data['sig_zx']
                    tau2 = self.stress_data['sig_yz']
                elif np.abs(normal[0]) >= np.abs(normal[1]):
                    tau1 = self.stress_data['sig_xy']
                    tau2 = self.stress_data['sig_zx']
                else:
                    tau1 = self.stress_data['sig_xy']
                    tau2 = self.stress_data['sig_yz']
        else:
            tau1 = 0
            tau2 = 0

        shear_traction = np.sqrt(tau1 ** 2 + tau2 ** 2)
        shear_traction.name = 'shear_traction'

        if mode == "I":
            sep_series = normal_sep.abs()
            traction_series = normal_traction.abs()
        elif mode == "II":
            sep_series = tangential_sep.abs()
            traction_series = shear_traction.abs()
        else:
            df_i, gi = self.calculate_Gc_by_integration(use_cohesive_separation=use_cohesive_separation, mode="I")
            df_ii, gii = self.calculate_Gc_by_integration(use_cohesive_separation=use_cohesive_separation, mode="II")
            result = pd.merge(
                df_i[['separation', 'traction', 'G_cumulative']],
                df_ii[['separation', 'traction', 'G_cumulative']],
                left_index=True, right_index=True, how='outer',
                suffixes=('I', 'II'),
            )
            result = result.sort_index().ffill().fillna(0)
            result['separation_mixed'] = np.sqrt(result['separationI'] ** 2 + result['separationII'] ** 2)
            result['traction_mixed'] = np.sqrt(result['tractionI'] ** 2 + result['tractionII'] ** 2)
            result['G_cumulative'] = result['G_cumulativeI'] + result['G_cumulativeII']
            result['mode'] = 'Mixed'
            return result, result['G_cumulative'].max()

        sep_df2 = pd.DataFrame({'separation': sep_series}).reset_index()
        stress_df2 = pd.DataFrame({'traction': traction_series}).reset_index()
        if 'time' not in sep_df2.columns:
            sep_df2 = sep_df2.rename(columns={'index': 'time'})
        if 'time' not in stress_df2.columns:
            stress_df2 = stress_df2.rename(columns={'index': 'time'})
        sep_df2 = sep_df2.sort_values('time')
        stress_df2 = stress_df2.sort_values('time')

        tolerance = 1e-4
        time_dtype = sep_df2['time'].dtype
        if hasattr(time_dtype, 'kind') and time_dtype.kind == 'M':
            tolerance = pd.Timedelta(microseconds=100)

        # Record the stress data time boundary (element deletion cutoff)
        # Add small epsilon to account for floating-point time mismatches
        # between nodout and elout timestamps
        stress_max_time = stress_df2['time'].max() + 1e-6

        result = pd.merge_asof(
            sep_df2, stress_df2, on='time', direction='nearest',
            tolerance=tolerance if isinstance(tolerance, (int, type(None), pd.Timedelta)) else None,
        )

        # Zero out traction for times beyond stress data (element deleted)
        result.loc[result['time'] > stress_max_time, 'traction'] = 0.0

        result = result.dropna(subset=['traction']).set_index('time').sort_index()
        result['delta_separation'] = result['separation'].diff().fillna(0)
        result['traction_avg'] = (result['traction'] + result['traction'].shift(1).fillna(0)) / 2
        result['dG'] = result['traction_avg'] * result['delta_separation'].abs()
        result['G_cumulative'] = result['dG'].cumsum()

        # G_c before element deletion (the physically meaningful value)
        Gc = result.loc[result.index <= stress_max_time, 'G_cumulative'].iloc[-1] \
            if (result.index <= stress_max_time).any() else result['G_cumulative'].iloc[-1]

        # Drop G_cumulative to zero after element deletion
        result.loc[result.index > stress_max_time, 'G_cumulative'] = 0.0

        return result, Gc

    def get_traction_separation_data(
        self,
        use_cohesive_separation: bool = True,
        mode: str = "I",
    ) -> pd.DataFrame:
        result, _ = self.calculate_Gc_by_integration(use_cohesive_separation, mode=mode)
        return result

    def calculate_internal_energy(self, use_cohesive_separation: bool = True, mode: str = "C") -> pd.Series:
        result, _ = self.calculate_Gc_by_integration(use_cohesive_separation, mode=mode)
        return result['G_cumulative'] * self.area


@dataclass
class Part:
    """
    Represents a part (material group) containing multiple elements.
    
    Attributes:
        pid: Part ID
        elements: Dictionary of elements in this part {eid: Element}
        internal_energy: Time series of internal energy for this part
    """
    pid: int
    elements: dict
    internal_energy: Optional[pd.Series] = None  # from matsum

    @property
    def element_ids(self) -> List[int]:
        """Get list of element IDs in this part."""
        return list(self.elements.keys())

    @property
    def node_ids(self) -> List[int]:
        """Get unique list of all node IDs in this part."""
        all_nodes = set()
        for elem in self.elements.values():
            all_nodes.update(elem.node_ids)
        return list(all_nodes)

    def get_max_internal_energy(self) -> float:
        """Get maximum internal energy over time."""
        if self.internal_energy is None:
            raise ValueError("Internal energy data not loaded")
        return self.internal_energy.max()



class Model:
    """
    Aggregates all LS-DYNA analysis data into a unified structure.
    
    Loads data from:
    - Keyword file (.k): element connectivity and initial node coordinates
    - nodout: nodal time-history data (displacements, velocities, accelerations)
    - elout: element stress time-history data
    - matsum: part internal energy time-history
    
    Usage:
        model = Model(folder, "model.k")
        element = model.elements[14501]
        part = model.parts[999]
    """

    def get_gn_curves(self):
        """Return dict of G-N curves from the keyfile (lcid -> {'title', 'data'})."""
        if self._keyfile_data is not None:
            return self._keyfile_data.get_gn_curves()
        return {}

    def __init__(
        self,
        folder: Path | str,
        keyfile: str,
        *,
        load_nodout: bool = True,
        load_elout: bool = True,
        load_matsum: bool = True,
    ):
        self.folder = Path(folder)
        self.keyfile_name = keyfile

        # Load raw data sources
        self._keyfile_data = None
        self._nodout_data = None
        self._elout_data = None
        self._matsum_data = None

        # Aggregated structures
        self.elements = {}
        self.parts = {}
        self.nodes = {}  # All nodes with initial coordinates

        # Load the keyfile FIRST so _keyfile_data is available
        self._load_keyfile()

        # Load optional data sources
        if load_nodout:
            self._load_nodout()
        if load_elout:
            self._load_elout()
        if load_matsum:
            self._load_matsum()

        # Build the model structure - initially only load elements in sets
        all_set_elements = set()
        solid_sets = getattr(self._keyfile_data, 'solid_sets', None)
        elements = getattr(self._keyfile_data, 'elements', {})
        if solid_sets and isinstance(solid_sets, dict) and len(solid_sets) > 0:
            for set_elements in solid_sets.values():
                all_set_elements.update(set_elements)
            self._build_model(list(all_set_elements))
        else:
            # Fallback: use all elements if no sets are found
            if elements is None:
                elements = {}
            self._build_model(list(elements.keys()))

    def _load_keyfile(self) -> None:
        """Load the keyword file data."""
        keyfile_path = self.folder / self.keyfile_name
        if not keyfile_path.exists():
            raise FileNotFoundError(f"Keyword file not found: {keyfile_path}")
        
        keyfile_path = self.folder / self.keyfile_name
        self._keyfile_data = KeyFileData(keyfile_path)
        self.nodes = self._keyfile_data.nodes.copy()

    def _load_nodout(self) -> None:
        """Load nodout data if available."""
        nodout_path = self.folder / "nodout"
        if nodout_path.exists():
            self._nodout_data = NodoutFrame.from_file(nodout_path, index_levels=("time", "id"))

    def _load_elout(self) -> None:
        """Load elout data if available."""
        elout_path = self.folder / "elout"
        if elout_path.exists():
            self._elout_data = EloutFrame.from_file(elout_path, index_levels=("time", "id"))

    def _load_matsum(self) -> None:
        """Load matsum data if available."""
        matsum_path = self.folder / "matsum"
        if matsum_path.exists():
            # Get all part IDs from the keyfile
            elements = getattr(self._keyfile_data, 'elements', {})
            if elements is None:
                elements = {}
            part_ids = list(set(
                elem_data['pid'] for elem_data in elements.values()
            ))
            self._matsum_data = Matsum(self.folder, part_ids)

    def _build_model(self, eids_to_load: List[int]) -> None:
        """Build the element and part structures from loaded data for specified elements."""
        if self._keyfile_data is None:
            return

        # First pass: create specified elements
        elements = getattr(self._keyfile_data, 'elements', {})
        if elements is None:
            elements = {}
        for eid in eids_to_load:
            if eid not in elements:
                continue
            elem_data = elements[eid]
            pid = elem_data['pid']
            node_ids = elem_data['nodes']

            # Get initial node coordinates for this element
            initial_coords = {
                nid: self.nodes[nid].copy()
                for nid in node_ids
                if nid in self.nodes
            }

            node_data = None
            stress_data = None

            # Attach nodout data for this element's nodes
            if self._nodout_data is not None:
                node_data = self._get_element_node_data(node_ids)

            # Attach elout data for this element
            if self._elout_data is not None:
                stress_data = self._get_element_stress_data(eid)

            element = Element(
                eid=eid, pid=pid, node_ids=node_ids,
                initial_node_coords=initial_coords,
                node_data=node_data, stress_data=stress_data,
            )

            self.elements[eid] = element

        # Second pass: create parts and group elements
        part_elements: dict = {}
        for eid, element in self.elements.items():
            pid = element.pid
            if pid not in part_elements:
                part_elements[pid] = {}
            part_elements[pid][eid] = element

        # Create Part objects
        for pid, elems in part_elements.items():
            part = Part(pid=pid, elements=elems)

            # Attach matsum data for this part
            if self._matsum_data is not None and pid in self._matsum_data.df.columns:
                part.internal_energy = self._matsum_data.df[pid]

            self.parts[pid] = part

    def load_element(self, eid: int) -> None:
        """Load a specific element and its part if not already loaded."""
        if eid in self.elements:
            return  # Already loaded

        if self._keyfile_data is None or eid not in self._keyfile_data.elements:
            raise KeyError(f"Element {eid} not found in keyfile")

        elem_data = self._keyfile_data.elements[eid]
        pid = elem_data['pid']
        node_ids = elem_data['nodes']

        # Get initial node coordinates for this element
        initial_coords = {
            nid: self.nodes[nid].copy()
            for nid in node_ids
            if nid in self.nodes
        }

        node_data = None
        stress_data = None

        # Attach nodout data for this element's nodes
        if self._nodout_data is not None:
            node_data = self._get_element_node_data(node_ids)

        # Attach elout data for this element
        if self._elout_data is not None:
            stress_data = self._get_element_stress_data(eid)

        element = Element(
            eid=eid, pid=pid, node_ids=node_ids,
            initial_node_coords=initial_coords,
            node_data=node_data, stress_data=stress_data,
        )

        self.elements[eid] = element

        # Ensure part exists
        if pid not in self.parts:
            part = Part(pid=pid, elements={eid: element})
            # Attach matsum data for this part
            if self._matsum_data is not None and pid in self._matsum_data.df.columns:
                part.internal_energy = self._matsum_data.df[pid]
            self.parts[pid] = part
        else:
            # Add element to existing part
            self.parts[pid].elements[eid] = element

    def _get_element_node_data(self, node_ids: List[int]) -> Optional[pd.DataFrame]:
        """Extract nodout data for specific nodes."""
        if self._nodout_data is None:
            return None

        df = self._nodout_data.df
        # Filter to only include nodes that are in this element
        available_nodes = df.index.get_level_values('id').unique()
        valid_nodes = [nid for nid in node_ids if nid in available_nodes]
        
        if not valid_nodes:
            return None

        return df.loc[(slice(None), valid_nodes), :]

    def _get_element_stress_data(self, eid: int) -> Optional[pd.DataFrame]:
        """Extract elout data for a specific element."""
        if self._elout_data is None:
            return None

        df = self._elout_data.df
        available_elements = df.index.get_level_values('id').unique()
        
        if eid not in available_elements:
            return None

        result = df.xs(eid, level='id')
        if isinstance(result, pd.Series):
            return result.to_frame().T
        return result

    @property
    def times(self) -> np.ndarray:
        # Get all time values from the analysis.
        if self._nodout_data is not None:
            return self._nodout_data.df.index.get_level_values('time').unique().values
        if self._elout_data is not None:
            return self._elout_data.df.index.get_level_values('time').unique().values
        if self._matsum_data is not None:
            return self._matsum_data.df.index.values
        return np.array([])

    @property
    def end_time(self) -> float:
        """Return the analysis end time (max time across all data sources)."""
        candidates = []
        if self._nodout_data is not None:
            candidates.append(self._nodout_data.df.index.get_level_values('time').max())
        if self._elout_data is not None:
            candidates.append(self._elout_data.df.index.get_level_values('time').max())
        if self._matsum_data is not None:
            candidates.append(self._matsum_data.df.index.max())
        return float(max(candidates)) if candidates else 0.0

    @property
    def element_ids(self) -> List[int]:
        # Get list of all element IDs.
        return list(self.elements.keys())

    @property
    def part_ids(self) -> List[int]:
        # Get list of all part IDs.
        return list(self.parts.keys())

    @property
    def node_ids(self) -> List[int]:
        # Get list of all node IDs.
        return list(self.nodes.keys())
    
    @property
    def solid_sets(self) -> dict[str, list[int]]:
        # Get predefined solid element sets from keyword file.
        if self._keyfile_data is None:
            return {}
        return self._keyfile_data.solid_sets

    def get_element(self, eid: int):
        # Get an element by ID, loading it if necessary.
        if eid not in self.elements:
            self.load_element(eid)
        return self.elements[eid]

    def get_part(self, pid: int) -> Part:
        # Get a part by ID.
        if pid not in self.parts:
            raise KeyError(f"Part {pid} not found")
        return self.parts[pid]

    def get_elements_by_part(self, pid: int) -> list:
        # Get all elements belonging to a specific part.
        return list(self.get_part(pid).elements.values())

    def print_summary(self) -> None:
        # Print a summary of the model.
        print("=" * 60)
        print("LS-DYNA Model Summary")
        print("=" * 60)
        
        print(f"\nKeyword file: {self.keyfile_name}")
        print(f"Analysis folder: {self.folder}")
        
        print(f"\nTotal Nodes: {len(self.nodes)}")
        print(f"Total Elements: {len(self.elements)}")
        print(f"Total Parts: {len(self.parts)}")
        
        print("\nParts:")
        print("-" * 60)
        for pid, part in self.parts.items():
            energy_info = ""
            if part.internal_energy is not None:
                max_energy = part.get_max_internal_energy()
                energy_info = f", Max Internal Energy: {max_energy:.6E}"
            print(f"  Part {pid}: {len(part.elements)} elements, {len(part.node_ids)} nodes{energy_info}")
        
        print("\nData Availability:")
        print("-" * 60)
        print(f"  nodout: {'Loaded' if self._nodout_data is not None else 'Not available'}")
        print(f"  elout:  {'Loaded' if self._elout_data is not None else 'Not available'}")
        print(f"  matsum: {'Loaded' if self._matsum_data is not None else 'Not available'}")
        
        if len(self.times) > 0:
            print(f"\nTime range: {self.times[0]:.6E} to {self.times[-1]:.6E} ({len(self.times)} steps)")


if __name__ == "__main__":
    folder = Path(r"C:\Users\nir\Desktop\Final_Project\analysis\single_element_mode_1_two_ways")

    # Create the unified model
    model = Model(folder, "simgle_element_mode_1.k")
    model.print_summary()

    print("\n" + "=" * 60)
    print("Example: Accessing Element Data")
    print("=" * 60)

    # Get an element
    eid = model.element_ids[0]
    element = model.get_element(eid)
    

    print(f"\nElement {eid}:")
    print(f"  Part ID: {element.pid}")
    print(f"  Node IDs: {element.node_ids}")
    print(f"  Initial Node Coordinates:")
    for nid, coords in element.initial_node_coords.items():
        print(f"    Node {nid}: ({coords['x']:.4f}, {coords['y']:.4f}, {coords['z']:.4f})")

    # Show stress data if available
    if element.stress_data is not None:
        print(f"\n  Stress data (first 5 time steps):")
        print(element.stress_data.head())

    # Show node displacement data if available  
    if element.node_data is not None:
        first_node = element.node_ids[0]
        print(f"\n  Displacement of node {first_node} (first 5 time steps):")
        print(element.get_node_displacement(first_node).head())

    print("\n" + "=" * 60)
    print("Example: Accessing Part Data")
    print("=" * 60)

    # Get a part
    pid = model.part_ids[0]
    part = model.get_part(pid)
    
    print(f"\nPart {pid}:")
    print(f"  Element IDs: {part.element_ids}")
    print(f"  Node IDs: {part.node_ids}")
    
    if part.internal_energy is not None:
        print(f"\n  Internal Energy (first 5 time steps):")
        print(part.internal_energy.head())
        print(f"\n  Max Internal Energy: {part.get_max_internal_energy():.6E}")

    print("\n" + "=" * 60)
    print("Face Area and Energy Per Area Calculation")
    print("=" * 60)

    # Get element
    element = model.get_element(eid)
    
    # Get all faces and their node sums
    print(f"\nElement {eid} Faces:")
    print("-" * 60)
    # Provide a stub for faces (hexahedral element: 6 faces, each with 4 nodes)
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ]
    node_ids = element.node_ids
    for i, face_indices in enumerate(faces):
        face = [node_ids[idx] for idx in face_indices if idx < len(node_ids)]
        face_sum = sum(face)
        face_area = element.area
        print(f"  Face {i+1}: Nodes {face}, Sum={face_sum}, Area={face_area:.6f}")

    # Get the face with minimum node sum
    min_face = min(faces, key=lambda f: sum(node_ids[idx] for idx in f if idx < len(node_ids)))
    min_sum = sum(node_ids[idx] for idx in min_face if idx < len(node_ids))
    min_face_area = element.area

    print(f"\nFace with lowest node sum:")
    print(f"  Nodes: {[node_ids[idx] for idx in min_face if idx < len(node_ids)]}")
    print(f"  Node sum: {min_sum}")
    print(f"  Area: {min_face_area:.6f}")

    # Calculate max energy per area
    if part.internal_energy is not None:
        max_energy = part.get_max_internal_energy()
        energy_per_area = max_energy / min_face_area

        print(f"\nMax Energy Per Area:")
        print(f"  Max Internal Energy: {max_energy:.6E}")
        print(f"  Face Area: {min_face_area:.6f}")
        print(f"  Max Energy / Area: {energy_per_area:.6E}")

    print("\n" + "=" * 60)
    print("G_c Calculation - Critical Energy Release Rate")
    print("=" * 60)

    # Get faces
    faces = element.get_faces()
    bottom_face = faces[0]  # (n1, n2, n3, n4)
    top_face = faces[1]     # (n5, n6, n7, n8)
    
    # Get the face with minimum node sum (for this cohesive element)
    min_face, min_sum = element.get_face_with_lowest_node_sum()
    min_face_area, _ = element.get_min_node_sum_face_area()
    
    print(f"\nCohesive Element {eid}:")
    print(f"  Bottom face (n1-n4): {bottom_face}")
    print(f"  Top face (n5-n8): {top_face}")
    print(f"  Face Area: {min_face_area:.6f}")
    
    # Show cohesive separation data
    print(f"\n" + "-" * 60)
    print("Cohesive Separation (Relative Displacement)")
    print("-" * 60)
    
    cohesive_sep = element.get_cohesive_separation()
    print(f"\nCohesive separation delta = u_top - u_bottom (first 5 steps):")
    print(cohesive_sep.head())
    print(f"\nMax separation magnitude: {cohesive_sep['magnitude'].max():.6E}")
    
    # Method 1: G_c by Energy / Area
    print(f"\n" + "-" * 60)
    print("Method 1: G_c = Max Internal Energy / Area")
    print("-" * 60)
    
    if part.internal_energy is not None:
        max_energy = part.get_max_internal_energy()
        Gc_energy_method = max_energy / min_face_area
        
        print(f"  Max Internal Energy: {max_energy:.6E}")
        print(f"  Face Area: {min_face_area:.6f}")
        print(f"  G_c (Energy/Area): {Gc_energy_method:.6E}")

    # Method 2a: G_c by Integration using COHESIVE separation (relative displacement)
    print(f"\n" + "-" * 60)
    print("Method 2a: G_c = integral sigma(delta) d(delta) (Using Cohesive Separation)")
    print("-" * 60)
    print("delta = relative displacement between top and bottom faces")
    
    traction_sep_df_cohesive, Gc_cohesive = element.calculate_Gc_by_integration(use_cohesive_separation=True)
    
    print(f"\nTraction-Separation Data (first 5 steps):")
    print(traction_sep_df_cohesive[['separation', 'traction', 'dG', 'G_cumulative']].head())
    
    print(f"\nTraction-Separation Data (last 3 steps):")
    print(traction_sep_df_cohesive[['separation', 'traction', 'dG', 'G_cumulative']].tail(3))
    
    print(f"\nG_c (Cohesive Separation): {Gc_cohesive:.6E}")
    
    # Method 2b: G_c by Integration using single face displacement (for comparison)
    print(f"\n" + "-" * 60)
    print("Method 2b: G_c = integral sigma(delta) d(delta) (Using Single Face Displacement)")
    print("-" * 60)
    print("delta = top face displacement only (ignoring bottom face)")
    
    traction_sep_df_single, Gc_single = element.calculate_Gc_by_integration(use_cohesive_separation=False)
    print(f"\nG_c (Single Face): {Gc_single:.6E}")
    
    # Comparison
    print(f"\n" + "=" * 60)
    print("G_c Comparison")
    print("=" * 60)
    
    print(f"\n  Material Input G_c:      0.256 (from .k file)")
    if part.internal_energy is not None:
        print(f"  G_c (Energy/Area):       {Gc_energy_method:.6E}")
    print(f"  G_c (Cohesive delta):        {Gc_cohesive:.6E}")
    print(f"  G_c (Single face delta):     {Gc_single:.6E}")
    
    if part.internal_energy is not None:
        print(f"\n  Ratio (Cohesive/Energy):     {Gc_cohesive / Gc_energy_method:.4f}")
        print(f"  Ratio (Single Face/Energy):  {Gc_single / Gc_energy_method:.4f}")