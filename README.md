# Music Embedding Clustering

Music Embedding Clustering Using a Pretrained Speaker Verification Model. Can a model trained for speaker verification separate songs from different bands?

## Method

Given a dataset with songs from some artists, I have extracted 15s excerpts from these songs and generated an embedding with `ECAPA-TDNN` pretrained for speaker-verification task on the VoxCeleb2 dataset.

Once we have the embeddings, we can visualize them on a TSNE plot:

![]("./images/tsne_bands.svg")

The artists where the vocal components are the most predominant, like `pop` and `rap`, are the ones that the model is capable to separate the best.
Interestingly, the `techno` genre represented by Boris Brejcha is also nicely separated and is closer to the `metal` and `rock` bands than to `rap` and `pop`

## Future work

I intend to come back at this task to finetune the model for genre/artist/album identification.

