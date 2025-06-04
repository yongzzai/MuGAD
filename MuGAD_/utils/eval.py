import numpy as np
from sklearn.metrics import precision_recall_curve

def evaluation(trace_label, trace_preds, event_labels, event_preds):
     precisions_trace, recalls_trace, thresholds_trace = precision_recall_curve(trace_label, trace_preds)
     f1s_trace = (2 * precisions_trace * recalls_trace) / (precisions_trace + recalls_trace)
     f1s_trace = np.nan_to_num(f1s_trace)
  
     best_index_t = np.argmax(f1s_trace)

     binary_trace_preds = (trace_preds >= thresholds_trace[best_index_t]).astype(int)
     detected_anomaly_idx = np.where(binary_trace_preds == 1)[0]
    
     filt_event_pred = []
     filt_event_label = []
     for idx in detected_anomaly_idx:
         filt_event_pred.extend(event_preds[idx])
         filt_event_label.extend(event_labels[idx])
    
     precisions_event, recalls_event, thresholds_event = precision_recall_curve(filt_event_label, filt_event_pred)
     f1s_event = (2 * precisions_event * recalls_event) / (precisions_event + recalls_event)
     f1s_event = np.nan_to_num(f1s_event)
     best_index_e = np.argmax(f1s_event)
     return f1s_trace[best_index_t], f1s_event[best_index_e]
