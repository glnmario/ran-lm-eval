## Evaluating the syntactic competence of RAN language models

Recurrent additive network (RAN) is a new gated RNN architecture whose states are weighted sums
of the linearly transformed input vectors, which allows to trace the importance of each of the
previous elements of the sequence when predicting the next element if the sequence. This paper
inspects a RAN language model’s ability to capture syntactic information by examining its
performance on the linguistic phenomena of subject-verb agreement and verb form government.
We observe that the RAN tends to remember content words most strongly, and does not seem to
specifically remember words that are relevant for dependency constructions. As a result, the
model performs poorly on grammaticality judgements.
