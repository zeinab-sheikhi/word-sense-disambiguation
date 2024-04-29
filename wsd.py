import random

from utils import prettyprint_sense_distribution, calculate_idf, WN_CORRESPONDANCES


class WSDClassifier(object):
    """
    Abstract class for WSD classifiers
    """

    def evaluate(self, instances):
        """
        Evaluates the classifier on a set of instances.
        Returns the accuracy of the classifier, i.e. the percentage of correct predictions.
        
        instances: list[WSDInstance]
        """
                
        true_predictions = sum(1 for ins in instances if ins.sense == self.predict_sense(ins))
        return round(true_predictions / len(instances), 3)
        

class RandomSense(WSDClassifier):
    """
    RandomSense baseline
    """
    
    def __init__(self):
        pass   
     
    def train(self, instances=[]):
        """
        instances: list[WSDInstance]
        """
        pass 

    def predict_sense(self, instance):
        """
        instance: WSDInstance
        """

        senses = list(WN_CORRESPONDANCES[instance.lemma].keys())  # list[string]
        random.shuffle(senses)
        return senses[0]
    
    def __str__(self):
        return "RandomSense"


class MostFrequentSense(WSDClassifier):
    """
    Most Frequent Sense baseline
    """
    
    def __init__(self):
        self.mfs = None  # Should be defined as a dictionary from lemmas to most frequent senses (dict[string -> string]) at training.
    
    def train(self, instances):
        
        """
        instances: list[WSDInstance]
        """
        
        dist = sense_distribution(instances)
        lemmas = set([ins.lemma for ins in instances])
        self.mfs = {lemma: max((key for key in dist if lemma in key), key=lambda k: dist[k]) for lemma in lemmas}

    def predict_sense(self, instance):
        """
        instance: WSDInstance
        """
        return self.mfs[instance.lemma]
        
    def __str__(self):
        return "MostFrequentSense"


class SimplifiedLesk(WSDClassifier):
    """
    Simplified Lesk algorithm
    """
    
    def __init__(self):

        # Should be defined as a dictionary from senses to signatures (dict[string -> set[string]]) at training.
        self.signatures = {}

    def train(self, instances, window_size=-1, use_idf=False):
        """
        instances: list[WSDInstance]
        window_size: int
        idf: bool
        """
        # For the signature of a sense, use (i) the definition of each of the corresponding WordNet synsets,
        # (ii) all of the corresponding examples in WordNet and (iii) the corresponding training instances.
        
        filtered_words = calculate_idf(instances)
        
        for ins in instances:
            
            sense = ins.sense
            context = ins.context
            right_context = ins.right_context[:window_size]
            left_context = ins.left_context[:window_size]
            
            if window_size != -1:
                context = left_context + right_context
            
            if use_idf:
                context = list(set(filtered_words).intersection(set(context)))

            if sense not in self.signatures.keys():
                self.signatures[sense] = get_signature(ins.lemma) + context
            else:
                self.signatures[sense] = context

        self.signatures = {key: set(value) for key, value in self.signatures.items()}

    def predict_sense(self, instance):
        """
        instance: WSDInstance
        """
        max_overlap = 0
        lemma = instance.lemma
        best_sense = None
        context = set(instance.context)

        definitions = {senses: signautures for senses, signautures in self.signatures.items() if lemma in senses}
        
        for sense, signature in definitions.items():
            overlap = len(signature.intersection(context))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sense = sense
        
        if best_sense is None:
            best_sense = random.choice(list(definitions.keys()))
        
        return best_sense
    
    def __str__(self):
        return "SimplifiedLesk"

###############################################################

###############################################################

# The body of this conditional is executed only when this file is directly called as a script (rather than imported from another script).


if __name__ == '__main__':
    
    from twa import WSDCollection
    from optparse import OptionParser
    from utils import sense_distribution, random_data_split, get_signature

    usage = "Comparison of various WSD algorithms.\n%prog TWA_FILE"
    parser = OptionParser(usage=usage)
    (opts, args) = parser.parse_args()
    if (len(args) > 0):
        sensed_tagged_data_file = args[0]
    else:
        exit(usage + '\nYou need to specify in the command the path to a file of the TWA dataset.\n')

    # Loads the corpus.
    instances = WSDCollection(sensed_tagged_data_file).instances
    
    # Displays the sense distributions.
    # prettyprint_sense_distribution(instances)
    
    # Splitting the data
    # test, train = random_data_split(instances)
    # test, train = random_data_split(instances, p=2)
    test, train = random_data_split(instances, n=10, p=1)
    # test, train = random_data_split(instances, n=10, p=3)

    # Evaluation of the random baseline on the whole corpus.
    randome_baseline = RandomSense()
    randome_baseline_acc = randome_baseline.evaluate(instances)
    print(f"Random Baseline {randome_baseline_acc}")

    # Evaluation of the most frequent sense baseline using different splits of the corpus (with `utils.data_split` or `utils.random_data_split`).
    most_frequent_baseline = MostFrequentSense()
    most_frequent_baseline.train(train)
    most_frequent_baseline_acc = most_frequent_baseline.evaluate(test)
    print(f"Frequent Sense Baseline {most_frequent_baseline_acc}")

    # Evaluation of Simplified Lesk (with no fixed window and no IDF values) using different splits of the corpus.
    simple_lesk = SimplifiedLesk()
    simple_lesk.train(train)
    simple_lesk_acc = simple_lesk.evaluate(test)
    print(f"Simplified Lesk {simple_lesk_acc}")
    
    # Evaluation of Simplified Lesk (with a window of size 10 and no IDF values) using different splits of the corpus.
    simple_lesk.train(instances=train, window_size=10)
    simple_lesk_window_acc = simple_lesk.evaluate(test)
    print(f"Simplified Lesk with window {simple_lesk_window_acc}")

    # Evaluation of Simplified Lesk (with IDF values and no fixed window) using different splits of the corpus.
    simple_lesk.train(instances=train[:10], use_idf=True)
    simple_lesk_idf_acc = simple_lesk.evaluate(test)
    print(f"Simplified Lesk with IDF {simple_lesk_idf_acc}")
    








    # Cross-validation
    pass  # TODO
    
    # Naive Bayes classifier
    pass  # TODO
