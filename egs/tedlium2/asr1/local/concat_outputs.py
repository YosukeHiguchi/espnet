# python python/concat_targets.py unigram256,unigram2048,unigram16384
# python python/concat_targets.py unigram16384,unigram16384,unigram16384
import os
import sys
import json
import glob

units = sys.argv[1].split(',')

for path in glob.glob('dump/*'):
    dname = os.path.join(path, 'deltafalse')

    data_jsons = []
    for unit in units:
        fname = os.path.join(dname, 'data_{}.json'.format(unit))

        with open(fname, "rb") as f:
            data_json = json.load(f)['utts']
        data_jsons.append(data_json)

    keys = data_jsons[0].keys()
    for data_json in data_jsons:
        assert keys == data_json.keys()

    combined_json = {}
    for key in keys:
        combined_json[key] = {
            'input': data_jsons[0][key]['input'],
            'output': [],
            'utt2spk': data_jsons[0][key]['utt2spk']
        }

        for i, data_json in enumerate(data_jsons):
            combined_json[key]['output'].append(
                {
                    'name': 'target{}_{}'.format(i, units[i]),
                    'shape': data_json[key]['output'][0]['shape'],
                    'text': data_json[key]['output'][0]['text'],
                    'token': data_json[key]['output'][0]['token'],
                    'tokenid': data_json[key]['output'][0]['tokenid'],
                }
            )
    savefile = os.path.join(dname, 'data_{}.json'.format(','.join(units)))
    print('Saving {}'.format(savefile))
    with open(savefile, 'w') as f:
        json.dump({'utts': combined_json}, f, ensure_ascii=False, indent=4)
