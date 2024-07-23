# project.py


import pandas as pd
import numpy as np
import os
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    time.sleep(5)
    book = requests.get(url).text.replace('\r\n', '\n')
    started = re.findall(r'\*\*\* START.+\*\*\*(.+)\*\*\* END', book, re.DOTALL)[0]
    return started

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of every paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of every paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens should include no whitespace.
        - Whitespace (e.g. multiple newlines) between two paragraphs of text 
          should be ignored, i.e. they should not appear as tokens.
        - Two or more newlines count as a paragraph break.
        - All punctuation marks count as tokens, even if they are 
          uncommon (e.g. `'@'`, `'+'`, and `'%'` are all valid tokens).


    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """
    paragraphed = re.sub(r'\n{2,}', ' \x03 \x02 ', book_string.strip())
    removenewline = re.sub(r'\s', ' ', paragraphed)
    addstart = re.sub('\A', '\x02 ', removenewline)
    addend = re.sub('\Z', '\x03', addstart)
    return re.findall(r'\\x02|\b[\w\d]+\b|[^\w\d\s]|\\x03', addend)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        token_ser = pd.Series(tokens).unique()
        return pd.Series(1/len(token_ser), token_ser)

        
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        return np.prod([self.mdl[word] if word in self.mdl.index else 0 for word in words])
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        return ' '.join(list(self.mdl.sample(M, replace = True, weights = self.mdl).index))


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        token_ser = pd.Series(tokens).value_counts()
        return token_ser/len(tokens)
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        return np.prod([self.mdl[word] if word in self.mdl.index else 0 for word in words])
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """
        return ' '.join(list(self.mdl.sample(M, replace = True, weights = self.mdl).index))


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """
        if len(tokens) == 2:
            return [(tokens[0], tokens[1])]
        return [tuple(tokens[i:i + self.N]) for i in range(len(tokens) - self.N + 1)]
        
        
    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe with three columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """
        ngramc = pd.Series(ngrams).value_counts()
        n1 = [x[:-1] for x in ngrams]
        ngram1c = pd.Series([x[:-1] for x in ngrams]).value_counts()

        df = pd.DataFrame(columns = ['ngram', 'n1gram','prob'])
        df['n1gram'] = n1
        df['ngram'] = ngrams
        df['prob'] = ngramc[ngrams].values/ngram1c[n1].values
        df['both'] = df['ngram'] + df['n1gram']
        df = df.drop_duplicates(subset=['both'])[['ngram', 'n1gram','prob']]
        df[['ngram', 'n1gram','prob']]
        return df
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two one three one two \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> p = bigrams.probability('two one three'.split())
        >>> np.isclose(p, (1/4) * (1/2) * (1/3))
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """
        gr = self.create_ngrams(words)
        prob = 1
        lst = []
        for i in range(len(gr)):
            x = list(self.mdl[self.mdl['ngram'] == gr[i]].prob.values)
            lst = lst + x
        prev = self.prev_mdl.probability(words[:self.N-1])
        if len(lst) < len(gr):
            return 0
        return np.prod(lst)*prev
    
    def h(self, curr):
        mdl = self
        if len(curr) >= mdl.N - 1:
            length = mdl.N - 1
            currn = curr[-length:]
            options = mdl.mdl[mdl.mdl['n1gram'] == tuple(currn)]
            if len(options) == 0:
                x = ['\x03']
            else:
                rand = np.random.choice(options['ngram'].values, size=1, p=options['prob'].values / sum(options['prob'].values))
                x = list(rand[0][-1:])
            return curr + x

        mdl = self.prev_mdl
        while len(curr) < mdl.N - 1:
            mdl = mdl.prev_mdl
        options = mdl.mdl[mdl.mdl['n1gram'] == tuple(curr)]
        if len(options) == 0:
            rand = ['\x03']
        else:
            rand = np.random.choice(options['ngram'].values, size=1, p=options['prob'].values / sum(options['prob'].values))
        return list(rand[0])


    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial START token.
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """
        # Use a helper function to generate sample tokens of length `length`

        curr =['\x02']
        curr = self.h(curr)
        for i in range(M):
            curr = self.h(curr)
        return ' '.join(curr)