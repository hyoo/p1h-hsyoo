#!/usr/bin/env python3
import numpy as np
from datasets import NCI60

data = NCI60.load_by_cell_data('BR:MCF7', subsample=None)
mat = data.as_matrix()
X, Y = mat[:, 1:], mat[:, 0]

feature_counts = X.shape[1]
#feature_counts = 300

for i in range(feature_counts):
    print('checking column ind: ', i, ' => [', np.min(X[:,i]), '...', np.max(X[:,i]), '], var:', np.var(X[:,i]), ' nulls:', np.count_nonzero(np.isnan(X[:,i])) )
