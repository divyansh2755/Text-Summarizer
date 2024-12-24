# Baseline summarizer : picks up the first 5 sentences from a single document

import sys
import os
files = sys.argv

# pick 5 sentences for summarization
k = 5

for index, file in enumerate(sorted(os.listdir(files[1]))):
    f = open('/users/singh/desktop/test_data/baseline/news'+str(index)+'_system'+str(index)+'.txt', 'w+')
    if file != ".DS_Store":
        f1 = open('/users/singh/desktop/test_data/duc2002/docs/'+file, 'r')
        c = f1.read()
        c = c.split('\n')
        for i in range(k):
            f.write(c[i]+"\n")
        f.close()
