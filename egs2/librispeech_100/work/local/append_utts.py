# python local/append_utts.py --file_path dump_multi/raw/{train_clean_100,dev}/wav.scp

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file_path",
    type=str,
    required=True,
)
args = parser.parse_args()


data = {}
with open(args.file_path) as f:
    for s in f.readlines():
        x = s.rstrip().split()
        if x[0][0:2] == "sp":
            rec_name = '-'.join(x[0].split('-')[0:3])
        else:
            rec_name = '-'.join(x[0].split('-')[0:2])
        if rec_name not in data.keys():
            data[rec_name] = {}
        data[rec_name][x[0]] = [x[1]]

for rec_name in data.keys():
    for utt_name in data[rec_name].keys():
        for utt_name_ in data[rec_name].keys():
            if utt_name == utt_name_:
                continue

            data[rec_name][utt_name].append(
                data[rec_name][utt_name_][0]
            )

output_file = '/'.join(
    args.file_path.split('/')[0:-1] + ['wav_multi.scp']
)
with open(output_file, "w") as f:
    for rec_name in data.keys():
        for utt_name in data[rec_name].keys():
            f.write("{} {}\n".format(
                utt_name,
                ' '.join(data[rec_name][utt_name]))
            )