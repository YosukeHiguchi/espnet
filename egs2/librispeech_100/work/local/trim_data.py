import os
import sys
import shutil
import glob

dataset = sys.argv[1]
dataset_dir = "data/{}".format(dataset)

with open("{}/wav.scp".format(dataset_dir), "r") as f:
    wavs = f.read().split("\n")[:-1]

end_time = {}
for alignment_file in glob.glob(
        "local/LibriSpeech-Alignments/{}/*/*/*".format(dataset.replace("_", "-"))
    ):
    with open(alignment_file, "r") as f:
        for line in f:
            x = line.rstrip().split()
            end_time[x[0]] = float(x[2].split(",")[-2])

new_wavs = []
for wav in wavs:
    utt_id = wav.split()[0]
    utt_file = wav.split()[1]

    if utt_id not in end_time:
        print("Warning: {} does not have alignment".format(utt_id))
        continue

    new_wavs.append(
        "{} sox {} -t wav - trim 0.0 ={} |".format(
            utt_id, utt_file, end_time[utt_id]
        )
    )

new_dataset_dir = dataset_dir + "_trim"
assert not os.path.exists(new_dataset_dir)
shutil.copytree(dataset_dir, new_dataset_dir)

with open("{}/wav.scp".format(new_dataset_dir), "w") as f:
    for wav in new_wavs:
        f.write(wav + "\n")
