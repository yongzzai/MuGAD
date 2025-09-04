# %%
'''
ref: https://github.com/guanwei49/BPAD
modified to mark anomalous events in the case
'''

import os
import itertools
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from generator.generation.anomaly import *
from generator.generation.attribute_generator import CategoricalAttributeGenerator
from processmining.log import EventLog
from utils.fs import EVENTLOG_DIR

ps = [0.1]

def get_log_files(path=None):

    ROOT_DIR = Path(__file__).parent

    if path is None:
        path = os.path.join(ROOT_DIR/'real-life_Logs')
    return [os.path.join(path,f) for f in os.listdir(path)]


def mark_anomalous_events(case, anomaly_type=None):

    for event in case.events:
        event.attributes['is_anomalous'] = 0
        event.attributes['anomaly_type'] = 'none'
    
    if case.attributes.get('label') == 'normal' or anomaly_type is None:
        return
    
    label = case.attributes.get('label')
    
    anomaly_name = anomaly_type.__class__.__name__
    if anomaly_name.endswith('Anomaly'):
        anomaly_name = anomaly_name[:-7]
    
    if isinstance(anomaly_type, SkipSequenceAnomaly):
        start = label['attr']['start']
        if start < len(case):
            case[start].attributes['is_anomalous'] = 1
            case[start].attributes['anomaly_type'] = 'SkipSequence'
    
    elif isinstance(anomaly_type, ReworkAnomaly):
        start = label['attr']['start']
        size = label['attr']['size']
        for i in range(start, min(start + size, len(case))):
            case[i].attributes['is_anomalous'] = 1
            case[i].attributes['anomaly_type'] = 'Rework'
    elif isinstance(anomaly_type, EarlyAnomaly):
        shift_from = label['attr']['shift_from']
        shift_to = label['attr']['shift_to']
        size = label['attr']['size']
        
        for i in range(shift_to, min(shift_to + size, len(case))):
            case[i].attributes['is_anomalous'] = 1
            case[i].attributes['anomaly_type'] = 'Early'
        
        if shift_from < len(case):
            case[shift_from].attributes['is_anomalous'] = 1
            case[shift_from].attributes['anomaly_type'] = 'Early'
    
    elif isinstance(anomaly_type, LateAnomaly):
        shift_from = label['attr']['shift_from']
        shift_to = label['attr']['shift_to']
        size = label['attr']['size']
        
        for i in range(shift_to, min(shift_to + size, len(case))):
            case[i].attributes['is_anomalous'] = 1
            case[i].attributes['anomaly_type'] = 'Late'
        
        if shift_from < len(case):
            case[shift_from].attributes['is_anomalous'] = 1
            case[shift_from].attributes['anomaly_type'] = 'Late'
    
    elif isinstance(anomaly_type, InsertAnomaly):
        for idx in label['attr']['indices']:
            if idx < len(case):
                case[idx].attributes['is_anomalous'] = 1
                case[idx].attributes['anomaly_type'] = 'Insert'
    
    elif isinstance(anomaly_type, AttributeAnomaly):
        for idx in label['attr']['index']:
            if idx < len(case):
                case[idx].attributes['is_anomalous'] = 1
                case[idx].attributes['anomaly_type'] = 'Attribute'


logs = [m for m in get_log_files()]
combinations = list(itertools.product(logs, ps))
for event_log_path, p in tqdm(combinations, desc='Add anomalies'):
    print(event_log_path)
    event_log = EventLog.from_xes(event_log_path)
    anomalies = [
        SkipSequenceAnomaly(max_sequence_size=3),
        ReworkAnomaly(max_distance=5, max_sequence_size=2),
        EarlyAnomaly(max_distance=5, max_sequence_size=2),
        LateAnomaly(max_distance=5, max_sequence_size=2),
        InsertAnomaly(max_inserts=2)]

    if event_log.num_event_attributes > 0:
        anomalies.append(AttributeAnomaly(max_events=3, max_attributes=min(2, event_log.num_activities)))

    for anomaly in anomalies:
        anomaly.activities = event_log.unique_activities
        anomaly.attributes = [CategoricalAttributeGenerator(name=name, values=values) for name, values in
                              event_log.unique_attribute_values.items() if name != 'name']

    num_cases = len(event_log)
    eligible_cases = [i for i, case in enumerate(event_log) if len(case) > 4]
    
    num_anomalous = int(num_cases * p)
    
    if len(eligible_cases) < num_anomalous:
        num_anomalous = len(eligible_cases)
        actual_p = num_anomalous / num_cases

    initial_attempt = min(len(eligible_cases), int(num_anomalous * 1.5))
    anomalous_indices = np.random.choice(eligible_cases, size=initial_attempt, replace=False)
    anomalous_set = set(anomalous_indices)
    
    print(f"Target: {num_anomalous} anomalies in {num_cases} cases ({p*100:.1f}%)")
    successful_anomalies = 0
    
    for i, case in tqdm(enumerate(event_log), total=num_cases):
        if successful_anomalies >= num_anomalous:
            if i in anomalous_set:
                NoneAnomaly().apply_to_case(case)
                mark_anomalous_events(case, None)
            else:
                NoneAnomaly().apply_to_case(case)
                mark_anomalous_events(case, None)
            continue
            
        if i in anomalous_set:
            np.random.shuffle(anomalies)
            
            success = False
            original_case = Case.clone(case)
            
            for anomaly in anomalies:
                result = anomaly.apply_to_case(case)
                
                if isinstance(result.attributes.get('label'), dict) and result.attributes.get('label').get('anomaly') == str(anomaly):
                    mark_anomalous_events(case, anomaly)
                    success = True
                    successful_anomalies += 1
                    break
                else:
                    case = Case.clone(original_case)
            
            if not success:
                NoneAnomaly().apply_to_case(case)
                mark_anomalous_events(case, None)
        else:
            NoneAnomaly().apply_to_case(case)
            mark_anomalous_events(case, None)
    
    print(f"Successfully injected anomalies: {successful_anomalies}/{num_anomalous} ({successful_anomalies/num_anomalous*100:.1f}%)")
    num_anomalous_cases = sum(1 for case in event_log if any(event.attributes.get('is_anomalous') == 1 for event in case.events))
    print(f"Number of anomalous cases: {num_anomalous_cases}")

    base_name=os.path.split(event_log_path)[1].split('.')[0]
    # event_log.save_json(os.path.join(EVENTLOG_DIR, f'{base_name}-{p:.2f}.json.gz'))    
    event_log.save_csv(os.path.join(EVENTLOG_DIR, f'{base_name}-({int(100*p)})temp.csv'))


# %%
