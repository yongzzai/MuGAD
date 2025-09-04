
import os.path
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
EVENTLOG_DIR = os.path.join(ROOT_DIR,'eventlogs')


ATTR_KEYS = {
    'BPIC12':{'AttributeKeys':['name']},
    'BPIC17_offer':{'AttributeKeys':['name', 'org_resource']},
    'BPIC20_DomesticDeclarations':{'AttributeKeys':['name', 'org_resource', 'org_role']},
    'BPIC20_RequestForPayment':{'AttributeKeys':['name', 'org_resource', 'org_role']},
    'Receipt':{'AttributeKeys':['name', 'org_group', 'org_resource']},
    'Road_Traffic_Fine_Management_Process':{'AttributeKeys':['name']},
    'Billing':{'AttributeKeys':['name']}}