import pandas as pd

filepath = "data/LabelledData (1).txt"
df = pd.read_table(filepath, delimiter=" ,,, ", names=['sentence', 'category'])
sents = df.sentence
cat = df.category

affirmation_df = df.loc[df['category'] == 'affirmation']
first_words = set()

def get_first(x):
    first_words.add(x.split()[0])
    return x

check = affirmation_df.sentence.apply(get_first)

with open('affirmations.txt', 'w') as f:
    f.write('\n'.join(list(first_words)))