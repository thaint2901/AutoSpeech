from data_objects.preprocess import preprocess_voxceleb1
from data_objects.compute_mean_std import compute_mean_std
from data_objects.partition_voxceleb import partition_voxceleb
from pathlib import Path
import argparse
import subprocess
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(description="Preprocesses audio files from datasets.",
        formatter_class=MyFormatter
    )
    parser.add_argument("dataset_root", type=Path, help= \
        "Path to the directory containing VoxCeleb datasets. It should be arranged as:")
    parser.add_argument("-s", "--skip_existing", action="store_true", help= \
        "Whether to skip existing output files with the same name. Useful if this script was "
        "interrupted.")
    args = parser.parse_args()

    # Process the arguments
    dev_out_dir = args.dataset_root.joinpath("feature", "dev")
    test_out_dir = args.dataset_root.joinpath("feature", "test")
    merged_out_dir = args.dataset_root.joinpath("feature", "merged")
    # assert args.dataset_root.exists()
    # assert args.dataset_root.joinpath('iden_split.txt').exists()
    # assert args.dataset_root.joinpath('veri_test.txt').exists()
    # assert args.dataset_root.joinpath('vox1_meta.csv').exists()
    dev_out_dir.mkdir(exist_ok=True, parents=True)
    test_out_dir.mkdir(exist_ok=True, parents=True)
    merged_out_dir.mkdir(exist_ok=True, parents=True)

    speaker_dirs = args.dataset_root.joinpath("dataset").glob("*")
    speaker_dirs = [speaker_dir for speaker_dir in speaker_dirs]
    speaker_dirs_train, speaker_dirs_test = train_test_split(speaker_dirs, test_size=0.25, random_state=42)

    # Preprocess the datasets
    preprocess_voxceleb1(speaker_dirs_train, dev_out_dir, args.skip_existing)
    preprocess_voxceleb1(speaker_dirs_test, test_out_dir, args.skip_existing)
    for path in dev_out_dir.iterdir():
        subprocess.call(['cp', '-r', path.as_posix(), merged_out_dir.as_posix()])
    for path in test_out_dir.iterdir():
        subprocess.call(['cp', '-r', path.as_posix(), merged_out_dir.as_posix()])
    compute_mean_std(merged_out_dir, args.dataset_root.joinpath('mean.npy'), args.dataset_root.joinpath('std.npy'))
    # partition_voxceleb(merged_out_dir, args.dataset_root.joinpath('iden_split.txt'))
    print("Done")
