# NEUIR
Neural network techniques for Information Retrieval

The retrieval is being done on 2 datasets : ``` Nepal earthquake ``` , ``` Italy earthquake ```

## Implemented Models:
* Character Level embeddings
* Word and Character Level embeddings in skipgram setting
* Word embedding with attention over character embedding
* Word embedding with attention over BiLSTM character embedding

## Running types:
* No query expansion
* Query expansion

Mode switiching is unimplemented and for now is being done by changes to source code

## Models:
* CLE : Character Level embeddings that are trained using Character Level context
* WC1 : Word and Character Level embeddings that are both combined together and trained to predict the context of the token ``` skipgram ``` method
* WC2 : Word and Character Level embeddings that are combined after applying attention to character sequence of the token, rest works similar to above
* WC3 : Word embeddings and attention over Character level BiLSTM model for token embedding extraction

The evaluation is run with ``` ./trec eval -q -m <measure standards> <standard> <output> ```

The data is available / was available under ``` FIRE2016 ```

#### These codes are part of a research project and will remain private till released publicly. When released they will be available under MIT license and therefore free for anyone to use till the time the work is cited by whoever who uses it. 

#### The code was written solely by Prannay Khosla during Workshop conducted by Microsoft Research India on Artificial Social Intelligence

NOTE : For access to datasets please contact ```prannay[dot]khosla[at]gmail[dot]com
