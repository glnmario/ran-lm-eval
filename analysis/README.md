### Analysis
Here we search for agreement and government phenomena in the Penn Treebank, and we analyse such phenomena with our language models.

`data.py`, `model.py`, and `ran.py` are the same files you can find in `language-modeling-nlp1/recurrent`. 
They are in this module only for (not unsolvable) compatibility reasons.

#### The `sentences` folder
- `regex_ed` refers to sentences of the type _have|has|had TOKEN(s) VERB-ed_
- `regex_ing` refers to sentences of the type _are|am|is TOKEN(s) VERB-ed|ing_ 
- `regex_irreg` refers to sentences of the type _have|has|had TOKEN(s) IRREGULAR_PAST_PART_
- `verb_subj` refers to sentences that include at least a 3rd-singular or non-3rd-singular person verb

Numbers from 1 to 5 in the filename, indicate how many _TOKENs_ intervene between the head and the dependent verbs.
