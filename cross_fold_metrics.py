from tensorboard.backend.event_processing import event_accumulator
import glob
import os
import pprint
import numpy as np
sets = {
    "RC-Only-Train": "runs/*RC-Only-Train-fineTune-Fold*",
    "RC-Only-Train-Base-Bert": "runs/*RC-Only-BaseBert-Train-fineTune-Fold*",
    "RC+Triage-Train": "runs/*RC-triage-Train-fineTune-Fold*",
    "RC+NER-Train": "runs/*RC-NER-Train-fineTune-Fold*",
    }

components = ['NER', 'RC', 'Triage', 'NER-Norm', "RC-BC6"]
metrics = ['Precision', 'Recall', 'F1']
print("Set", 'Metric', 'Component', 'Fold_num', sep='\t', end='\t')
print('\t'.join([str(i+1) for i in range(10)]))
for set_, path in sets.items():
    for component in components:
        for metric in metrics:
            globStr = f"{path}/{metric}_{component}_*"
            folder_list = glob.glob(globStr)
            folder_list = sorted(folder_list, key=lambda x: x.split('/')[-1])

            if len(folder_list) > 0:
                assert len(folder_list) == 10, f"{globStr}, {len(folder_list)}"
                dataNums = 10 if 'shot' not in set_ else 500
                epoch2values = {i:[] for i in range(dataNums+1)}
                for folder in folder_list:
                    event = glob.glob(folder+'/events*')[0]
                    ea=event_accumulator.EventAccumulator(event)
                    ea.Reload()
                    values=ea.scalars.Items(ea.scalars.Keys()[0])

                    for i in values:
                        if i.step > dataNums:
                            break 
                        epoch2values[i.step].append(i.value)
                    
                    # print('\t'.join([str(i.value) for i in values if i.step < dataNums + 1]))


                print(set_, metric, component, 'Mean', sep = '\t', end='\t')
                print('\t'.join([str(np.mean(values)) for epoch, values in epoch2values.items() if len(values) > 0]))

