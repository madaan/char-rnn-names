# On Transferring Names

I found a bunch of names over at https://mbejda.github.io/. Thanks, @mbejda. The dataset consists of Hispanic, Indian,
Caucasian and African American names. Now perhaps you have read this [fantastic blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
on RNNs and what they do.

Here is one of quotes from the blog:

> We’ll train RNNs to generate text character by character and ponder the question “how is that even possible?”

Yes, it's pretty mind blowing. The blog goes into the details on *how* it's done. Here's a short summary of *what* RNNs
do: RNNs can take a bunch of text, say T, and learn to generate new text that would *seem* to be taken from T.
For example, we can train an RNN on all the works of William Shakespeare, and the RNN can in turn generate new text that
would seem to be taken from Shakespeare. The blog I've linked to contains this and many other interesting examples.

We have:

1. A way to generate new examples of a piece of text by learning from the existing examples (RNNs).
2. A corpus of names.

So it would not be very inspiring if I say that we can use RNNs to generate new names. Yes we can, and yes, I did.
We had an "Indian Name Generator" generating *Deepaks* and *Nehas*, a "Caucasian Name Generator" generating
*Michaels* and *Jennifers* and so on. It was something, but as I said, nothing very surprising.


#### Cross Seeding

Now, I didn't feel like throwing all these RNNs away, so wondered what would happen if I feed, say, the "Indian Name Generator"
with a first few characters from a Caucasian name, and let it generate the rest? Will it try to **create** a name that
sounds Indian? So I ran a bunch of these experiments, and present the somewhat more interesting results in this post.
I also used seeds from names of Pokemons, wrestlers and such. It was fun to see all these different RNNs take a stab at
creating a name that sounds to be from their domain. Before we jump on to the results, we'll have a brief look at the 
dataset. We'll then discuss how the dataset was transformed to be fed to RNN. This will be followed by a discussion of 
the model and the prediction process and finally, we'll look at the results.


## II. Looking at the Data
Although there is no limit to the number of different analysis we can run, we'll present only two here in the interest of
space: the most popular names and the name length distributions. Looking at the most popular names will give us a feel for 
the dataset, and the name length distribution was added to add plots and look fancy (and it's used somewhere down the line, too).
 

##### Top 5 Most Popular Names

The following tables list the top 5 most popular names for each of the corpus. 

| African American    |                   |                  |                 |
|---------------------|-------------------|------------------|-----------------|
| Female  First Names | Female Last names | Male First Names | Male Last Names |
| latoya              | williams          | michael          | johnson         |
| ashley              | johnson           | james            | brown           |
| patricia            | brown             | anthony          | jones           |
| angela              | smith             | willie           | jackson         |
| mary                | jackson           | robert           | davis           |



| Caucasian          |                   |                   |                 |
|--------------------|-------------------|-------------------|-----------------|
| Female First Names | Female Last Names | Male  First Names | Male Last Names |
| jennifer           | smith             | michael           | johnson         |
| amanda             | brown             | james             | rodriguez       |
| kimberly           | williams          | robert            | davis           |
| jessica            | miller            | david             | jones           |
| ashley             | johnson           | john              | brown           |



| Hispanic            |                   |                  |                  |
|---------------------|-------------------|------------------|------------------|
| Female  First Names | Female Last Names | Male First Names | Male  Last Names |
| maria               | rodriguez         | jose             | rodriguez        |
| melissa             | gonzalez          | juan             | garcia           |
| jennifer            | rivera            | luis             | martinez         |
| gloria              | perez             | carlos           | rivera           |
| elizabeth           | garcia            | jorge            | hernandez        |


| Indian             |                   |                  |                 |
|--------------------|-------------------|------------------|-----------------|
| Female First Names | Female Last Names | Male First Names | Male Last Names |
| smt                | devi              | deepak           | kumar           |
| pooja              | pooja             | rahul            | singh           |
| smt.               | kumari            | amit             | sharma          |
| jyoti              | jyoti             | ram              | lal             |
| kumari             | bai               | sanjay           | ram             |



##### Name Length Distributions

The name length distributions are next plotted for each of the races. Indian names seem to be the only outlier, with short
names giving rise to the minor mode. All the other races seem to be evenly distributed with majority of names being 15
characters long and thereabouts.

![African American Names](https://raw.githubusercontent.com/madaan/char-rnn-names/master/docs/african_american_name_len_dist.png) ![Caucasian](https://raw.githubusercontent.com/madaan/char-rnn-names/master/docs/caucasian_name_len_dist.png)


![Hispanic](https://raw.githubusercontent.com/madaan/char-rnn-names/master/docs/hispanic_name_len_dist.png) ![Indian](https://raw.githubusercontent.com/madaan/char-rnn-names/master/docs/indian_name_len_dist.png)


## III. Input Representation

In this section, we will spend some time looking at how do we take these bunch of names and convert them into something
that can be fed to something like an RNN. We will get to that representation in two steps: encoding, standardization and
embeddings. 

Let's quickly recap what we are set to do. The algorithm we are trying to train must learn to predict the next character
of a sequence given the previous characters. Consider a sample name, “han[]yao#”. Note that the name is in small case, []
is a space, and “#” is a special character that denotes end of the name. So we have (prev character →  next character):    
h → a    
ha → n    
han → []    
han[] → han[]y    
han[]y → han[]ya    
han[]ya → han[]yao    
han[]yao → han[]yao#

#### a) Encoding

We convert each character in our name to a number. This is achieved by defining a simple mapping as follows:

| Character             | Encoding |
|-----------------------|----------|
| a-z                   | 0-25     |
| " " (Space)           | 26       |
| "#" (End of Name)     | 27       |
| . (Invalid Character) | 28       |

- We assume that all the names are made up of lowercase english alphabets, with a space separating different components
of a name. Every name ends with a special name end character (“#”). Every other character is mapped to an invalid
character, mapped to a “.”. 

- Given the above table, it just becomes a dictionary lookup. Nothing special. The map is defined in ```char-rnn-names/src/features/char_codec.py```. 

#### b) Standardization 

Length of the names varies a lot. However, for training purposes, the RNNs are usually approximated by an “unrolled”
version. This is basically the RNN but unrolled over some n timesteps, where each timestep is an element in the input
sequence. In short, we will need to fix on a good “maximum name length”. Names longer than the maximum name length will
be truncated, and names smaller than the maximum name length will be padded with Invalid characters. As discussed, we
assume that every name ends with a “Name End” character(#).  All of this happens in the following piece of code:
```python
@staticmethod
def encode_and_standardize(name):
   name = name + CharCodec.NAME_END #add the end of the name symbol for everyname
   name = CharCodec.encode(name) #encode
   if name_len >= CharCodec.max_name_length:
       truncated_name = name[:(CharCodec.max_name_length - 1)]
       truncated_name.append(CharCodec.char_to_class[CharCodec.NAME_END]) #must attach the name end
       return np.array(truncated_name, dtype=np.int32)
   else:
       padded_name = np.empty(CharCodec.max_name_length, dtype=np.int32)
       padded_name.fill(CharCodec.INVALID_CHAR_CLASS)
       padded_name[:name_len] = name
       return padded_name
```
Note that we retain the name end character (#) even after the truncation.

So how do we arrive at the ```max_name_length```? A simple way to do that is fix something safe like 100. However, that
would mean that our network will be wider than we perhaps want (most of the names will be smaller than 100 characters). This
will lead to lots of wasted computation and slower training times (give it a try!). Or, we could just plot a distribution
of the name lengths and pick something simple. We've already done that, and as you can see, it seems like 25 will cover 
most of the cases. That’s what I use in the experiments.

#### c) Embeddings

So far, we have standardized each name to a fixed length, added a character to mark the end of the name, and encoded the
name from an array of chars to an array of integers. 

If you don’t know what embeddings are at all, I recommend checking out [this](https://www.tensorflow.org/programmers_guide/embedding) or [this](https://deeplearning4j.org/word2vec.html) link. 


The tl;dr of the technique is that each of the characters is mapped to an array of numbers. The array is called an
embedding, and the length of the array is the dimensionality of the embedding. For example, if we choose to use 5 
dimensional embeddings, it just means that every character is mapped to a 5 dimensional real vector. We can start with
pre-trained embeddings, or learn them as part of the training process (which is what our model does). In our setting,
that means that we will hopefully learn embeddings that makes it easier for us to predict the next character in the name.
The following 3 lines of code is all we need to add embeddings to your codebase. ```embeddings``` is a matrix, which has
one row per element in our vocabulary. ``tf.nn.embedding_lookup`` takes in the input, which is ```batch_size x max_name_length```,
and maps (*looks up*) each character in the name to an embedding, to yield an input with dimensions ```batch_size x max_name_length x n_embeddings```.
```py 
names = tf.placeholder(tf.int32, shape=(None, max_name_length), name="input")
embeddings = tf.Variable(tf.random_uniform([vocab_size, n_embeddings], -1.0, 1.0))
embedded_names = tf.nn.embedding_lookup(embeddings, names) #(?, max_name_length, n_embeddings)
``` 
Tensorboard can help in [visualizing embeddings](https://www.tensorflow.org/versions/r1.1/get_started/embedding_viz) using
PCA and t-SNE. The t-SNE visualization of the embedding matrix from the indian name generator are as follows. As you can
see, the vowels are all close to each other, which hints at the fact that the learned embeddings wrap some linguistic 
properties of the names pertaining to the race, and are perhaps useful in generating new ones. 

|![Embeddings](https://raw.githubusercontent.com/madaan/char-rnn-names/master/docs/embedding.png)|
|:--:|
|*Embeddings learned by the Indian Name Generator visualized using Tensorboard*|

## IV. Model

The model used is a character level recurrent neural network (LSTM in this particular case). [This book](http://shop.oreilly.com/product/0636920052289.do),
has a pretty good explanation of the LSTMs, and [this neat blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
is another standard reference. A high level overview over the model follows. Each character in the normalized name is
converted to the corresponding embedding vector, which is fed to the first LSTM in the stack. The output from this first
LSTM is fed to a second LSTM. The second LSTM is then fed to a dense layer, which emits a logits vector of length 29. 
An argmax over the logits vector is used to calculate the loss and make predictions. 



|![Model Architecture](https://raw.githubusercontent.com/madaan/char-rnn-names/master/docs/model.png)|
|:--:| 
| *The overall model setup generated by tensorboard. The input Embeddings are fed to stacked LSTMs, which in turn feed to a dense layer* |

A cooked up instance of the model is shown below. It's a replica of [this diagram](http://karpathy.github.io/assets/rnn/charseq.jpeg) from [this blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) I've already linked to.

|![Model Example](https://raw.githubusercontent.com/madaan/char-rnn-names/master/docs/model_example.png)|
|:--:|
|*A sample instance of the model. The input name is "Amy#" (# being the end of the name character). At each step, the network is expected to predict the next character. Thus, At step 1, the expected output is "M". As explained in the encoding section, the characters after the "#", "." signal end of the input.*|

---
## V. Transferring Name Styles

Since we have discussed a lot, let's quickly recap before moving ahead. Some of the following may seem to be a repeat from
the introduction, because it sort of is.

- Dataset: We have a dataset of names from different races.
- Generator: We have discussed a model that can be trained to predict the next character in a sentence.
 Let's call this model the **generator**.  

We train one generator per dataset. Thus, we have a model that has learned to predict the next few characters in an
Indian name given the first few, and so on. We then seed each of these generators with a few characters from a name,
say "Der" from "Derek", and compare the results.

The prediction process is illustrated in the figure below, followed by the code.

![Prediction Process](https://raw.githubusercontent.com/madaan/char-rnn-names/master/docs/prediction_process.png)

Code:

```py
res = seed
initial_seed_offset = len(seed)
for i in range(CharCodec.max_name_length - len(seed)):
    feats = CharCodec.encode_and_standardize(res).reshape(1, CharCodec.max_name_length)
    prediction = sess.run(model.prediction, feed_dict={names: feats})
    res += " ".join(CharCodec.decode(prediction[0])[i + initial_seed_offset - 1])
    if res[-1] == CharCodec.NAME_END:
        break
```

## VI. Results
The results are compiled in the following table. The whole table is appended at the end, we discuss a few examples in
this section. The first column is the name, the second the seed used as an initial input to the model. The subsequent 
columns list the names generated by different generator using the given seed. 

| name              | seed         | african_american           | caucasian                  | hispanic                   | indian                     | all_races                |
|--------------------|---------------|----------------------------|----------------------------|----------------------------|----------------------------|--------------------------|
zhang wei          | zhan          | zhankhea l stencor#        | zhane a nelson#            | zhanole s estrada#         | zhanna sankar#             | zhanson e martin#

In the above example, the seed used is "zhan", from the chinese name "zhang wei". All the networks take this tricky seed
and generate a name that finally *looks* like a name from the given model. For example, the african american generator 
yields "zhankhea l stencor" (the "#" is the end of the name marker), the caucasian generator yields "zhane a nelson" and
so on. 

The following table contains some cherry-picked examples. Results from [this test file](https://github.com/madaan/char-rnn-names/blob/master/data/test.txt)
are appended at the end.



| name              | seed         | african_american           | caucasian                  | hispanic                   | indian                     | all_races                |
|--------------------|---------------|----------------------------|----------------------------|----------------------------|----------------------------|--------------------------|
| undertaker         | underta       | undertall nix#             | undertan starlir#          | underta romero#            | undertala#                 | undertayshawn king#  
| aman madaan        | aman mad      | aman madadenis#            | aman madich#               | aman madro l gonzalez#     | aman madhkaran#            | aman madha#              |
| jose luis          | jose l        | jose l graham#             | jose l ramirez#            | jose l morales#            | jose lal sharma#           | jose l rodriguez#        |
| hideyoshi          | hideyo        | hideyon u bennett#         | hideyo g morio#            | hideyordo rodriguez#       | hideyohar sharma#          | hideyon d brown#         |
| dan fineman        | dan f         | dan f briggs#              | dan f witharr#             | dan flekrez#               | dan farjat saini#          | dan francersiii#         |
 
 

| name              | seed         | african_american           | caucasian                  | hispanic                   | indian                     | all_races                |
|--------------------|---------------|----------------------------|----------------------------|----------------------------|----------------------------|--------------------------|
| derek vroom        | de            | derrick l curtis#          | dennis r coleman#          | dennis r suro#             | deepak . pardeep#          | derrick l brown#         |
| derek vroom        | derek         | derek a clark#             | derek a hardellin#         | derek m sanchez#           | derekha  bhai#             | derek a carter#          |
| derek vroom        | derek vr      | derek vright#              | derek vraworit#            | derek vrkilla.carcial#     | derek vrasha#              | derek vrow#              |
| dan fineman        | da            | david l morgan#            | david l mccoy#             | daniel carachure#          | dalip shah#                | david l brown#           |
| dan fineman        | dan f         | dan f briggs#              | dan f witharr#             | dan flekrez#               | dan farjat saini#          | dan francersiii#         |
| dan fineman        | dan fine      | dan finette tyrel#         | dan fines#                 | dan finera.garcia#         | dan fine#                  | dan fine#                |
| carolyn kennedy    | car           | carlos e jr robinson#      | carlos r reyes#            | carlos a guerra#           | carta#                     | carlos a cardona#        |
| carolyn kennedy    | carolyn       | carolyn r miller#          | carolyn a roe#             | carolyno  jr rodriguez#    | carolynir#                 | carolyn r martinez#      |
| carolyn kennedy    | carolyn ken   | carolyn kendry#            | carolyn kendrick#          | carolyn kendeles.rodriguez | carolyn kent#              | carolyn kent#            |
| michael zuckerberg | mich          | michael a chapman#         | michael a mckinney#        | michael a martinez#        | michael#                   | michael a mccormick#     |
| michael zuckerberg | michael z     | michael z chatman#         | michael z baker#           | michael z martinez#        | michael zah#               | michael z russell#       |
| michael zuckerberg | michael zucke | michael zuckenne#          | michael zuckellan#         | michael zucke#             | michael zuckena#           | michael zuckelli#        |
| gary yao           | ga            | gary l hayes#              | gary w jr davis#           | gabriel a romero#          | gaurav sharam#             | gary l baker#            |
| gary yao           | gary          | gary l hayes#              | gary w jr davis#           | gary avila#                | gary#                      | gary l baker#            |
| gary yao           | gary y        | gary y jr brown#           | gary yancey#               | gary y avares#             | gary yadav#                | gary yadav#              |
| pranay chopra      | pra           | pradel lucas#              | praie r saliska#           | pramie rivera#             | pramod kumar tiwari#       | pramod kumar tiwari#     |
| pranay chopra      | pranay        | pranay s mccarter#         | pranayoos n santana#       | pranay a gonzalez#         | pranayavara bijgut#        | pranayan devi#           |
| pranay chopra      | pranay ch     | pranay charlon#            | pranay chess#              | pranay chaveda#            | pranay chawla#             | pranay chand#            |
| mayank bhardwaj    | may           | maydreona l miles#         | maykel c calderon#         | maynor padilla.estrada#    | maya kumari#               | maya devi#               |
| mayank bhardwaj    | mayank        | mayank redison#            | mayank j romero#           | mayank h rodriguez#        | mayank .aakash#            | mayank pandey#           |
| mayank bhardwaj    | mayank bhar   | mayank bharder#            | mayank bharresse#          | mayank bharque#            | mayank bhardwaj#           | mayank bhardwaj#         |
| aman madaan        | am            | amos e robinson#           | amanda l hatcher#          | amalio rivera#             | amit kumar singh#          | amanda l hutchinson#     |
| aman madaan        | aman          | aman j joseph#             | aman p jr marshall#        | aman f rivera#             | aman saroj#                | aman sharma#             |
| aman madaan        | aman mad      | aman madadenis#            | aman madich#               | aman madro l gonzalez#     | aman madhkaran#            | aman madha#              |
| priyanka lingawal  | priy          | priy loveranc#             | priye j mergtone#          | priy g martinez#           | priyanka . pinki#          | priyanka . pinki#        |
| priyanka lingawal  | priyanka      | priyankauguson#            | priyanka l monagan#        | priyankaune#               | priyanka . pinki#          | priyanka . pinki#        |
| priyanka lingawal  | priyanka lin  | priyanka linkley#          | priyanka linto#            | priyanka linorriz#         | priyanka linditay#         | priyanka linda#          |
| uma sawant         | um            | umar a hassan#             | umid r mann#               | umeliar m sanchez#         | uma . barkha#              | uma devi#                |
| uma sawant         | uma s         | uma stimer l jenkins#      | uma s vancey#              | uma s muniz#               | uma shankar soni#          | uma shankar#             |
| uma sawant         | uma saw       | uma sawrm crow#            | uma sawyornig#             | uma sawillas#              | uma sawan#                 | uma saweer#              |
| ankur dewan        | an            | anthony l mcclain#         | anthony r horn#            | antonio r hernandez#       | anil kumar sain#           | anthony j perry#         |
| ankur dewan        | ankur         | ankurick reed#             | ankur m davis#             | ankur mondales#            | ankur kumar#               | ankur sharma#            |
| ankur dewan        | ankur de      | ankur dewin#               | ankur dehalf#              | ankur delapamill#          | ankur devi#                | ankur devi#              |
| rahul gupta        | ra            | raymond l hopps#           | randall l rice#            | ramon a feliciano#         | ram sagar#                 | raymond l holland#       |
| rahul gupta        | rahul         | rahuleshor c robbin#       | rahul a jr gonzalez#       | rahul g martinez#          | rahul sharma#              | rahul sharma#            |
| rahul gupta        | rahul gu      | rahul gudert#              | rahul guenoa#              | rahul gutineza#            | rahul gupta#               | rahul gupta#             |
| zhang wei          | zh            | zhivago rodgers#           | zhenya v deckarri#         | zhecory colinza#           | zhini devi#                | zhenya santana#          |
| zhang wei          | zhan          | zhankhea l stencor#        | zhane a nelson#            | zhanole s estrada#         | zhanna sankar#             | zhanson e martin#        |
| zhang wei          | zhang         | zhang m forbeson#          | zhang j henderson#         | zhang g diaz.hernandez#    | zhang depo#                | zhang r johnson#         |
| wang xiu ying      | wan           | wanda j malone#            | wanda j dentoscionle#      | wanda rodriguez#           | wani bhavi#                | wanda j davis#           |
| wang xiu ying      | wang x        | wang x jamison#            | wang x bailey#             | wang x rodriguez#          | wang xai d.o lageh ri par# | wang xiluseiia mond#     |
| wang xiu ying      | wang xiu      | wang xiu douglas#          | wang xiu r mccall#         | wang xiu castillo#         | wang xiu shaker#           | wang xiu santing#        |
| hideyoshi          | hi            | hillard j london#          | hilario m gomez#           | hiram a maldonado#         | himanshu . annu#           | himanshu . annu#         |
| hideyoshi          | hide          | hideajrake redmon#         | hidel f martinez#          | hidel garcia#              | hidee chaudhry#            | hidena#                  |
| hideyoshi          | hideyo        | hideyon u bennett#         | hideyo g morio#            | hideyordo rodriguez#       | hideyohar sharma#          | hideyon d brown#         |
| zhang xiu ying     | zha           | zharran e austin#          | zhays j notz#              | zhanole s estrada#         | zhanna sankar#             | zhayni essidion#         |
| zhang xiu ying     | zhang x       | zhang x hall#              | zhang x herty#             | zhang x corrova#           | zhang xeta#                | zhang x i barraman#      |
| zhang xiu ying     | zhang xiu     | zhang xiu donner#          | zhang xiu r brack#         | zhang xiu cardenas#        | zhang xiu w.o#             | zhang xiu partadill#     |
| hashimoto          | ha            | harry l jr caine#          | harold d minor#            | harold a cardenas.valenci# | harish chand dua#          | harold l harris#         |
| hashimoto          | hash          | hashantae walleggter#      | hashia l sannero.ferzill.c | hashus rodriguez.gutierre# | hashim#                    | hashim#                  |
| hashimoto          | hashim        | hashimmela j slove#        | hashim naviguenteras#      | hashimaer m gonzalez.matha | hashim#                    | hashim#                  |
| shinzo abe         | sh            | shantae d dennis#          | shannon l bennett#         | shawn d rodriguez#         | shankar lal gopalani#      | shane a brown#           |
| shinzo abe         | shinz         | shinzella b dowers#        | shinzeyne a scharden#      | shinz a medina#            | shinz yadav#               | shinza delvacy#          |
| shinzo abe         | shinzo        | shinzo  iii hardy#         | shinzo a dejesunnell.igos# | shinzo d rodriguez#        | shinzo . raju#             | shinzo . ammo#           |
| maria guadalupe    | mar           | marcus d cook#             | mark a beasley#            | mario a martinez#          | mariyam . sonam#           | mark a barnes#           |
| maria guadalupe    | maria g       | maria green#               | maria g pierce#            | maria g serrano#           | maria garg#                | maria g castillo#        |
| maria guadalupe    | maria guada   | maria guada#               | maria guadaguez#           | maria guadal.pont#         | maria guadal#              | maria guada#             |
| jose luis          | jo            | john l graham#             | joseph a bernston#         | jose a correa#             | joyti sharma#              | joseph a brown#          |
| jose luis          | jose          | joseph l henry#            | joseph a bernston#         | jose a correa#             | josender  sharma#          | joseph a brown#          |
| jose luis          | jose l        | jose l graham#             | jose l ramirez#            | jose l morales#            | jose lal sharma#           | jose l rodriguez#        |
| veronica           | ve            | vernon l hankerson#        | vernon r locke#            | veronica hernandez#        | veer bhan#                 | vernon l brown#          |
| veronica           | vero          | veronica m butler#         | veronica b jamison#        | veronica hernandez#        | veronika bhinder#          | veronica l brown#        |
| veronica           | veroni        | veronica m butler#         | veronica b jamison#        | veronica hernandez#        | veronika bhinder#          | veronica l brown#        |
| juan carlos        | ju            | julius mcdaniels#          | justin a moore#            | juan c montalba#           | julfi . sonu#              | justin l barnette#       |
| juan carlos        | juan          | juan p laster#             | juan c martinez#           | juan c montalba#           | juan morji#                | juan c castillo#         |
| juan carlos        | juan car      | juan cardonas#             | juan carranza#             | juan carrillo#             | juan carodhari#            | juan cardona#            |
| rosa maria         | ro            | robert l moore#            | robert l bresteane#        | roberto c munoz rivera#    | rohit agrawal#             | robert l jr marshall#    |
| rosa maria         | rosa          | rosa a hobbs#              | rosa m demerd#             | rosa m torres#             | rosa kalah#                | rosa a johnson#          |
| rosa maria         | rosa ma       | rosa mannt#                | rosa martinez#             | rosa mata#                 | rosa mani#                 | rosa massey#             |
| francisco javier   | fran          | frank j brown#             | frank e carballo#          | francisco j mendez#        | frana#                     | frank l jr bennett#      |
| francisco javier   | francisc      | francisco a nunez#         | francisco gonzales#        | francisco j mendez#        | francischan#               | francisco j mendez#      |
| francisco javier   | francisco ja  | francisco javen#           | francisco jaza.torrez#     | francisco jaigue#          | francisco jain#            | francisco jaramaz.ramos# |
| maria elena        | ma            | marcus d cook#             | mark a beasley#            | mario a martinez#          | manish kumar  mandal#      | mark a barnes#           |
| maria elena        | maria         | maria murray#              | maria l west#              | maria g serrano#           | mariam  mumtaz#            | maria l lassinorino#     |
| maria elena        | maria el      | maria ellis#               | maria elweow#              | maria elaramdo#            | maria elam#                | maria eller#             |
| brianna            | b             | brandon m fleming#         | brandon l holland#         | bryan santana#             | bharat bhushan#            | brian k bradley#         |
| brianna            | bri           | brian k brickhound#        | brian k saitham#           | brian n acosta#            | brij mohan thakur#         | brian k bradley#         |
| brianna            | brian         | brian k brickhound#        | brian k saitham#           | brian n acosta#            | brianti#                   | brian k bradley#         |
| chloe              | c             | charles e hall#            | christopher m brannon#     | carlos a guerra#           | chander shekhar#           | christopher m mcclendon# |
| chloe              | ch            | charles e hall#            | christopher m brannon#     | christian rivera#          | chander shekhar#           | christopher m mcclendon# |
| chloe              | chl           | chlente chasbin#           | chloe p johnson#           | chlistonae a funsez#       | chlepal#                   | chletu s jr marshall#    |
| destiny            | d             | david l morgan#            | david l mccoy#             | daniel carachure#          | deepak . pardeep#          | david l brown#           |
| destiny            | des           | desmond t baker#           | desiree l barnett#         | dessie santiago#           | desh raj#                  | desmond t harris#        |
| destiny            | desti         | destiny j rios#            | destiny straton#           | destin santiago#           | desti#                     | destiny j brown#         |
| jeremiah           | je            | jermaine l martin#         | jeremy m morrison#         | jesus m salinas#           | jeet pal  .jitu#           | jeremy l coleman#        |
| jeremiah           | jere          | jeremy a lee#              | jeremy m morrison#         | jeremy o burgos#           | jerender singh#            | jeremy l coleman#        |
| jeremiah           | jeremi        | jeremiah d preister#       | jeremiah m conner#         | jeremias v gomez#          | jeremi  rana#              | jeremiah d brown#        |
| josiah             | j             | james l caldwell#          | joseph a bernston#         | jose a correa#             | jai kishan gupta#          | joseph a brown#          |
| josiah             | jos           | joseph l henry#            | joseph a bernston#         | jose a correa#             | josin jomon#               | joseph a brown#          |
| josiah             | josi          | josiah m beachem#          | josian montanez#           | josie a rivera#            | josin jomon#               | josiah m martinez#       |
| undertaker         | un            | undray anderson#           | uney s mendes#             | unesto brito#              | unknown . monu#            | undrae m mcclellan#      |
| undertaker         | under         | underially r ivy#          | underico cabez.garcia#     | under p maldonado#         | under jain#                | underick j collins#      |
| undertaker         | underta       | undertall nix#             | undertan starlir#          | underta romero#            | undertala#                 | undertayshawn king#      |
| yokozuna           | yo            | yolanda y kryger#          | yosmanis a cruz#           | yovanny z bautista#        | yogesh chahar#             | yogesh chandra jo#       |
| yokozuna           | yoko          | yokondrae o hurdolk#       | yoko d chambers#           | yokonnihua f seplano#      | yoko#                      | yokoshania c carter#     |
| yokozuna           | yokozu        | yokozull bowell#           | yokozuma c adams#          | yokozua l uerda#           | yokozuddina#               | yokozua j harris#        |
| andre the giant    | and           | andre l arnold#            | andrew j denny#            | andres r guajardo#         | andhav#                    | andrew j brown#          |
| andre the giant    | andre t       | andre t mccray#            | andre t mccoy#             | andre t salazar#           | andre topta#               | andre t morgan#          |
| andre the giant    | andre the g   | andre the gilmer#          | andre the gorsom#          | andre the garcia#          | andre the gadar#           | andre the getti#         |
| big show           | bi            | billy r jr collier#        | billy j ball#              | billy e caballero reyes#   | birender kuma .r yadav#    | billy j banks#           |
| big show           | big           | big a mitchell#            | big p o carson#            | big mariano#               | big singh#                 | big nemi#                |
| big show           | big sh        | big shardley j jr atturwow | big shulte#                | big shergorio luz#         | big shan#                  | big sharma#              |
| hulk hogan         | hu            | hubert l hunt#             | hugh walthall#             | humberto m malagon#        | husainpreet kour#          | hugh a holt#             |
| hulk hogan         | hulk          | hulk  iv herrit#           | hulk leath#                | hulk rodriguez.gonzale#    | hulk mohd#                 | hulk  kumari#            |
| hulk hogan         | hulk ho       | hulk hornes#               | hulk howstie#              | hulk hoelles.maldonado#    | hulk holoo chand singh#    | hulk holu#               |
| picachu            | p             | patrick l lee#             | paul a branch#             | pedro j salazarlopez#      | parveen kaur . simi#       | paul j boyd#             |
| picachu            | pic           | picky edmores#             | pick d sansom#             | picer g melendez#          | picay . sonu#              | pick d moran#            |
| picachu            | picac         | picace lewis#              | picacco turdlere u alvie#  | picacio burgos#            | picachiram akwal#          | picace a james#          |
| mewtwo             | m             | marcus d cook#             | michael a mckinney#        | mario a martinez#          | manish kumar  mandal#      | mark a barnes#           |
| mewtwo             | mew           | mewille m rodriguez#       | mewald j donnynons#        | mewel achevedo#            | mewa aggarwal#             | mewasui . sanjali#       |
| mewtwo             | mewt          | mewtis f jr baken#         | mewthed shewnert#          | mewtel diaz#               | mewta#                     | mewtilles dison#         |
| jigglypuff         | ji            | jimmy l jr floyd#          | jimmy d booher#            | jimmy r torres#            | jitender sehrawat#         | jimmy l barnett#         |
| jigglypuff         | jiggl         | jiggl jordan#              | jigglan a layhorn#         | jigglan r alvaro#          | jiggla#                    | jiggl r porter#          |
| jigglypuff         | jigglyp       | jigglyphal#                | jigglyp t cedeno#          | jigglype vernarda#         | jigglypri . papa#          | jigglypu g martin#       |
| bulbasaur          | bu            | burnell mckenney#          | buddy l cook#              | bulfrano garcia#           | budh prakash#              | buddy l mccray#          |
| bulbasaur          | bulb          | bulber l hargus#           | bulbernon k manond#        | bulbim m rolon#            | bulbul#                    | bulbul#                  |
| bulbasaur          | bulbas        | bulbas  jr littles#        | bulbas.san s layvoro#      | bulbashumald gasialisanosh | bulbas#                    | bulbas b rivera#         |
| charizard          | ch            | charles e hall#            | christopher m brannon#     | christian rivera#          | chander shekhar#           | christopher m mcclendon# |
| charizard          | char          | charles e hall#            | charles e jr hartsfield#   | charles r murillo#         | charan singh . minchu#     | charles a brown#         |
| charizard          | chariz        | charizelo m laster#        | charizy a berry#           | chariz l martinez#         | chariz#                    | chariz c hill#           |