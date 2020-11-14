from data_objects.preprocess import preprocess_voxceleb1
from pathlib import Path
import argparse
import subprocess

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
    public_out_dir = args.dataset_root.joinpath("feature", "public-test")
    # assert args.dataset_root.exists()
    # assert args.dataset_root.joinpath('iden_split.txt').exists()
    # assert args.dataset_root.joinpath('veri_test.txt').exists()
    # assert args.dataset_root.joinpath('vox1_meta.csv').exists()
    public_out_dir.mkdir(exist_ok=True, parents=True)

    public_speaker_dirs = args.dataset_root.joinpath("public-test").glob("*")
    public_speaker_dirs = [speaker_dir for speaker_dir in public_speaker_dirs]

    # Preprocess the datasets
    preprocess_voxceleb1(public_speaker_dirs, public_out_dir, args.skip_existing)
    print("Done")
