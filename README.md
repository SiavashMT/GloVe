Simple Python Implementation of Global Vectors for Word Representation (GloVe)

Example Usage:
```
    import nltk
    from nltk.corpus import brown
    import logging
    logging.getLogger().setLevel(logging.INFO)

    nltk.download('brown')

    data = brown.sents(categories=['news'])[:100]

    glove = GloVe()

    glove.train(data, number_of_iterations=20, optimizer='adagrad')

    print(glove.word_mapping)

```

Implemented Based-on:

[1] http://www.aclweb.org/anthology/D14-1162

[2] http://www.foldl.me/2014/glove-python/ (https://github.com/hans/glove.py)