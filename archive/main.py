import pandas as pd

df = pd.read_csv("./data/grb2_binding.tsv", sep="\t")

single = df[df["num_mutations"] == 1]
double = df[df["num_mutations"] == 2]

if len(single) < 100 or len (double) < 100:
    raise ValueError("Not enough data for single or double mutations.")

single_sample = single.sample(n=1000, random_state=42)
double_sample = double.sample(n=1000, random_state=42)

sampled = pd.concat(
    [single_sample.assign(mutation_type="single"), 
     double_sample.assign(mutation_type="double")],

     ignore_index=True
)

sampled.groupby("mutation_type")["score"].describe()

