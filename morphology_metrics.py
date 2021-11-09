import sys, math, copy

def read_treebank(filepath):
    trees = []
    tree = []
    with open(filepath, "r") as f:
        for line in f:
            if line == "\n":
                trees.append(tree)
                tree = []
                continue
            items = line.split("\t")
            if "." in items[0] or "-" in items[0] or len(items) != 10:
                continue
            else:
                tree.append(items)
    return trees


def normalised_head_pos_entropy(treebank):
    delexicalised_treebank = delexicalise_treebank(treebank)
    dw_vocab = get_vocab(delexicalised_treebank, 1)
    pos_vocab = get_vocab(delexicalised_treebank, 3)
    probabilities = get_hpos_given_word_probabilities(delexicalised_treebank, dw_vocab,
                                                      pos_vocab)
    total = 0.
    for dw, probs in probabilities.items():
        hpe_w = 0
        instances = 0.
        for pos, p in probs.items():
            if p != 0:
                hpe_w -= p * math.log(p, 2)
            instances += 1
        total += hpe_w / math.log(instances, 2)
    nhpe =  1 - total / len(dw_vocab)
    return nhpe 

def get_vocab(treebank, column):
    vocab = set([])
    for tree in treebank:
        for entry in tree:
            vocab.add(entry[column])
    return vocab

def get_hpos_given_word_probabilities(treebank, dw_vocab, pos_vocab):
    counts = {w: {} for w in dw_vocab}
    for w, c in counts.items():
        for pos in pos_vocab:
            counts[w][pos] = 0. 
    total_counts = {w: 0. for w in dw_vocab}
    for tree in treebank:
        for entry in tree:
            dw = entry[1]
            head = int(entry[6]) - 1
            head_pos = tree[head][3]
            counts[dw][head_pos] += 1
            total_counts[dw] += 1
    probabilities =  {w: {} for w in dw_vocab}
    for w, c in counts.items():
        total = total_counts[w]
        for pos, v in c.items():
            probability = v/total
            probabilities[w][pos] = probability
    return probabilities
   
def delexicalise_treebank(treebank):
    delexicalised_treebank = []
    for tree in treebank:
        delexicalised_tree = [delexicalise_tree(entry) for entry in tree]
        delexicalised_treebank.append(delexicalised_tree)
    return delexicalised_treebank

def delexicalise_tree(entry):
    POS = entry[3]
    MFEATS = entry[5]
    delexicalised_word = "|".join([POS, MFEATS])
    new_entry = copy.deepcopy(entry)
    new_entry[1] = delexicalised_word
    return new_entry

def lemma2form_ratio(treebank, inflected=False):
    lemma_vocab = get_vocab(treebank, 2)
    lemma_form_sets = get_all_forms_of_lemmas(treebank, lemma_vocab)
    if not inflected:
        total = sum([len(form_set) for lemma, form_set in lemma_form_sets.items()])
        vocab_c = len(lemma_vocab)
    else:
        all_lengths =[len(form_set) for lemma, form_set in lemma_form_sets.items() if len(form_set) > 1]
        total = sum(all_lengths)
        vocab_c = len(all_lengths)
    if total != 0:
        ratio = vocab_c / total
    else:
        ratio = 1
    return 1 - ratio
        

def get_all_forms_of_lemmas(treebank, lemma_vocab):
    lemma_form_sets = {lemma: set([]) for lemma in lemma_vocab}
    for tree in treebank:
        for entry in tree:
            lemma = entry[2]
            form = entry[1]
            lemma_form_sets[lemma].add(form)
    return lemma_form_sets

def token2type_ratio(treebank):
    N_tokens =count_tokens(treebank)
    vocab = get_vocab(treebank, 1)
    return len(vocab)/N_tokens

def count_tokens(treebank):
    return sum([len(tree) for tree in treebank])

def normalised_word_entropy(treebank):
    vocab = get_vocab(treebank, 1)
    probabilities = get_word_probabilities(treebank, vocab)
    H = 0.
    for w, p in probabilities.items():
        H -= p * math.log(p, 2)
    return H / math.log(len(vocab), 2)

def get_word_probabilities(treebank, vocab):
    counts = {w: 0. for w in vocab}
    total = count_tokens(treebank)
    for tree in treebank:
        for entry in tree:
            word = entry[1]
            counts[word] += 1
    probabilities = {}
    for w, c in counts.items():
        probability = c/total
        probabilities[w] = probability
    return probabilities

def get_morphological_complexity_score(filepath):
    treebank = read_treebank(filepath)
    NHPE = normalised_head_pos_entropy(treebank) 
    TTR = token2type_ratio(treebank) 
    NWH = normalised_word_entropy(treebank) 
    LF = lemma2form_ratio(treebank) 
    iLF = lemma2form_ratio(treebank, inflected=True) 
    average = (NHPE + TTR + NWH + LF + iLF) / 5.
    return average

if __name__ =="__main__":
    filepath = sys.argv[1]
    treebank = read_treebank(filepath)
    NHPE = normalised_head_pos_entropy(treebank) 
    TTR = token2type_ratio(treebank) 
    NWH = normalised_word_entropy(treebank) 
    LF = lemma2form_ratio(treebank) 
    iLF = lemma2form_ratio(treebank, inflected=True) 
    print("HPE*:\t", format(NHPE, ".5f"))
    print("TTR:\t", format(TTR, ".5f"))
    print("WH*:\t", format(NWH, ".5f"))
    print("F/L*:\t", format(LF, ".5f"))
    print("F/iL*:\t", format(iLF, ".5f"))
    print("----------------------------------------")
    average = (NHPE + TTR + NWH + LF + iLF) / 5.
    print("Morphological complexity:\t", format(average, ".5f"))
