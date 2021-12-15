import json
import pandas as pd
import sys

df = pd.read_json(sys.argv[1])
print(df.to_csv(sep='\t'))

