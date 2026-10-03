"""V5 data adapter: keep original sources/sampler/training and use both M5 classes."""
from prepare_m5_combined_v5 import records
import training_v4


def run(args,cfg,base):
    return training_v4.run(args,cfg,base,m5_loader=records,protocol=5)
