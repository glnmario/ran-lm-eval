### Analysis
Here we search for agreement and government phenomena in the Penn Treebank, and we analyse such phenomena with our language models.

`data.py`, `model.py`, and `ran.py` are the same files you can find in `language-modeling-nlp1/recurrent`. 
They are in this module only for (not unsolvable) compatibility reasons.

#### The `sentences` folder
- `regex_ed` refers to sentences of the type _have|has|had TOKEN(s) VERB-ed_
- `regex_ing` refers to sentences of the type _are|am|is TOKEN(s) VERB-ed|ing_ 
- `regex_irreg` refers to sentences of the type _have|has|had TOKEN(s) IRREGULAR_PAST_PART_
- `subj_verb` refers to sentences that include at least a 3rd-singular or non-3rd-singular person verb
- `verb_form` refers to sentences that include at least a verb form government phenomenon

Numbers from 0 to 5 in the filename, indicate how many _TOKENs_ intervene between the head and the dependent verbs.


The `analysis.py` script accepts the following arguments:

```
optional arguments:
  -h, --help         show this help message and exit
  --file PATH        name of the input file inside the 'sentences/' folder
  --mode MODE        weight visualisation mode: l1, l2, max_w, l1_c
  --print            whether to print the activations to file (perplexities are always printed)
  --plot             whether to show weight activations plots
```
