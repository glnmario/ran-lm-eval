# Word-level language modeling with recurrent neural networks

This example trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
By default, the training script uses the PennTreeBank dataset.

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`)
which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --embeds EMBEDS    location of the pretrained embeddings
  --model MODEL      type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, RAN)
  --embdims EMBDIMS  dimensionality of word embeddings
  --nunits NUNITS    number of hidden units per layer
  --nlayers NLAYERS  number of layers
  --bidir            use bidirectional network
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch-size N     batch size
  --bptt BPTT        sequence (paragraph) length
  --dropout DROPOUT  dropout applied to layers (0 to 1)
  --tied             tie the word embedding and softmax weights
  --seed SEED        random seed
  --cuda             use CUDA
  --log-interval N   report interval
  --save SAVE        path to save the final model
  --log LOG          path to save the epoch-by-epoch log (csv format)
```

With these arguments, a variety of models can be tested:

```bash
python main.py --cuda --model RNN_TANH --embeds '../embeddings/glove.6b/glove.6b.50d.txt' --embdims 50 --epochs 40
python main.py --model LSTM --bidir --embdims 50 --nunits 650 --lr 0.01 --dropout 0.5 --epochs 40 --log 'lstm-log.csv'
python main.py --model RAN --embdims 256 --nunits 1024 --dropout 0.5 --epochs 100 --nlayers 1 --batch-size 64
python main.py --cuda --model GRU --embdims 50 --nunits 650 --lr 0.01 --dropout 0.5 --epochs 40 --save 'path/model.pt'
python main.py --model RNN_RELU --bidir --clip 0.2 --embdims 50 --nunits 650 --lr 0.01 --dropout 0.5 --epochs 40
python main.py --data '../data_check' --model LSTM --bidir --embdims 50 --nunits 300 --lr 0.01 --dropout 0.5 --epochs 20

...and so on. Be creative!
```
