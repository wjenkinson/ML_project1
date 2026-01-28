from pathlib import Path


def main() -> None:
    """Create an 80/20 train/validation split over available LAMMPS dump files.

    The script:
    - Looks for files matching ``dump.*.LAMMPS`` in the ``data`` directory.
    - Sorts them lexicographically (which matches timestep order for the current naming).
    - Uses the first 80% of frames for training and the remaining 20% for validation.
    - Writes lists of filenames (basenames) to ``data/splits/train_files.txt`` and
      ``data/splits/val_files.txt``.
    """
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    def timestep_from_name(path: Path) -> int:
        """Extract numeric timestep from a filename like 'dump.12345.LAMMPS'."""
        name = path.name
        try:
            parts = name.split(".")
            return int(parts[1])
        except (IndexError, ValueError):
            return 0

    dump_files = sorted(data_dir.glob("dump.*.LAMMPS"), key=timestep_from_name)

    if not dump_files:
        print(f"No LAMMPS dump files found in {data_dir}")
        return

    num_files = len(dump_files)
    num_train = int(0.8 * num_files)

    train_files = dump_files[:num_train]
    val_files = dump_files[num_train:]

    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    def write_file_list(path: Path, files) -> None:
        with path.open("w", encoding="utf-8") as f:
            for p in files:
                # Store just the basename; paths are always resolved relative to data_dir
                f.write(p.name + "\n")

    train_list_path = splits_dir / "train_files.txt"
    val_list_path = splits_dir / "val_files.txt"

    write_file_list(train_list_path, train_files)
    write_file_list(val_list_path, val_files)

    print(f"Found {num_files} dump files in {data_dir}")
    print(f"Training files: {len(train_files)} -> {train_list_path}")
    print(f"Validation files: {len(val_files)} -> {val_list_path}")

    if train_files:
        print("First train file:", train_files[0].name)
        print("Last train file:", train_files[-1].name)
    if val_files:
        print("First val file:", val_files[0].name)
        print("Last val file:", val_files[-1].name)


if __name__ == "__main__":
    main()
