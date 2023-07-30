import numpy as np

def calculate_acc(pred, label):
    return (pred.round() * label).sum() + ((1-pred).round() * (1 - label)).sum()
