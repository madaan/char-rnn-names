## Training Character Level RNNs on Names



| name              | seed         | african_american           | caucasian                  | hispanic                   | indian                     | all_races                |
|--------------------|---------------|----------------------------|----------------------------|----------------------------|----------------------------|--------------------------|
| undertaker         | underta       | undertall nix#             | undertan starlir#          | underta romero#            | undertala#                 | undertayshawn king#  
| aman madaan        | aman mad      | aman madadenis#            | aman madich#               | aman madro l gonzalez#     | aman madhkaran#            | aman madha#              |
| jose luis          | jose l        | jose l graham#             | jose l ramirez#            | jose l morales#            | jose lal sharma#           | jose l rodriguez#        |
| hideyoshi          | hideyo        | hideyon u bennett#         | hideyo g morio#            | hideyordo rodriguez#       | hideyohar sharma#          | hideyon d brown#         |
| dan fineman        | dan f         | dan f briggs#              | dan f witharr#             | dan flekrez#               | dan farjat saini#          | dan francersiii#         |
 
 

### Configuration
The following configuration file that wraps all the information that's needed for training and generation.
```javascript
{
  "datasets": [{
            "name": "",
            "data": "data/hispanic_names.txt"
        }],
  "n_epochs": 2,
  "log_dir": "logs",
  "model_dir": "models",
  "seeds_file": ,
  "result_file": 
}
```

Before you start:
```bash
 $ export PYTHONPATH=$PYTHONPATH:`pwd`
 ```
 
### Training

```bash
$ python src/training/train.py names_rnn_conf.json
```

### Generation
```bash
$ python src/scoring/batch_generate.py names_rnn_conf.json "data/test.txt" "data/test_output.txt"
```

Where data/test.txt is a text file with the input seed names ([example](https://github.com/madaan/char-rnn-names/blob/master/data/test.txt)).
Each of the models will be fed the prefixes derived from these names, and the output will be stored in "data/test_output.txt".

### Starting Tensorboard
[train.py](https://github.com/madaan/char-rnn-names/blob/master/src/training/train.py) stores logs that can be used to 
track the loss per epoch and visualize embeddings. To start Tensorboard:

```bash
$ tensorboard --logdir=logs/
```

and then go to http://localhost:6006.

### Training Time
- Training time depends on the size of the dataset. It takes about an hour to train on a file with 40,000 names using a 
GTX 1070.


![lossvtime](https://docs.google.com/spreadsheets/d/e/2PACX-1vSc-jXFrvvYt0w7lt_vJsMZgI2azvWJH8qcjguekP0_6E40NME0cEjrPnyN-xWDk0iT6qBBfFfyhcNb/pubchart?oid=1011156586&format=image)