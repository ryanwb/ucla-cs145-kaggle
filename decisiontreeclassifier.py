"""
decisiontreeclassifier.py

Implements a decision tree classifier
"""

import math
from classifier import Classifier

# TODO: Possibly modify the greed to peek a couple levels deep into the recursion?

# TODO: THIS USES RECURSION... and we probably can't hold all of that in memory

# Decision should be one of the cuisine selections
class LeafNode(object):
    def __init__(self, decision = None):
        self.decision = decision

class InternalNode(object):
    def __init__(self, split_criteria = None, yes_node = None, no_node = None):
        self.split_criteria = split_criteria
        self.yes_node = yes_node
        self.no_node = no_node

# This classifier assumes that every data point has been learn()'d before
# calling classify(); on the first attempt to classify(), we will build the tree
class DecisionTreeClassifier(Classifier):

    def __init__(self):
        self._root = None
        self._data = [] # (ingredients list, cuisine) tuples

    def learn(self, ingredients, cuisine):
        self._data.append((ingredients, cuisine))

    # This uses entropy right now
    # TODO: Try GINI index, etc.
    # TODO: Very small numbers -- do we mess up the math here? Need more precision?
    def pick_partition(self, proj_database, possible_splits):
        # dict mapping a split to the entropy of that split
        split_results = {}
        for split_ingredient in possible_splits:
            # print "Looking at ingredient " + split_ingredient
            # counters for number of yes and number of no if we use this split
            n_yes = 0
            n_no = 0
            # dicts mapping the cuisine to its count within this split
            yes = {}
            no = {}
            for (ingredients, cuisine) in proj_database:
                if split_ingredient in ingredients:
                    n_yes += 1
                    if cuisine in yes:
                        yes[cuisine] += 1
                    else:
                        yes[cuisine] = 1
                else:
                    n_no += 1
                    if cuisine in no:
                        no[cuisine] += 1
                    else:
                        no[cuisine] = 1
            # print str(n_yes) + " yes, " + str(n_no) + " no"
            entropy_yes = 0.0
            entropy_no = 0.0
            for cuisine in yes:
                x = float(yes[cuisine])/float(n_yes)
                entropy_yes -= x * math.log(x, 2)
                # print "x, yes[cuisine], n_yes, entropy_yes = " + str(x) + " " + str(yes[cuisine]) + " " + str(n_yes) + " " + str(entropy_yes)
            for cuisine in no:
                x = float(no[cuisine])/float(n_no)
                entropy_no -= x * math.log(x, 2)
            # print "ENTROPY NO: " + str(entropy_no)
            # print "ENTROPY YES: " + str(entropy_yes)
            entropy = ((float(n_yes)/float(n_no + n_yes)) * entropy_yes) + ((float(n_no)/float(n_no + n_yes)) * entropy_no)
            split_results[split_ingredient] = entropy
            # print "Entropy of splitting at " + split_ingredient + " is " + str(entropy)
        return min(split_results, key = split_results.get)

    # At the top level, just call self.construct_tree()
    # Returns a node
    # TODO: Prevent overfitting
    def construct_tree(self, proj_database = None, possible_splits = None, best_guess_sofar = None):
        if proj_database == None:
            proj_database = self._data
        if possible_splits == None:  # top-level of recusion
            print "Compiling list of possible splits..."

            c = {}
            for (ingredients, cuisine) in proj_database:
                for ingredient in ingredients:
                    if ingredient in c:
                        c[ingredient] += 1
                    else:
                        c[ingredient] = 1
            c_sort = sorted(c, key = c.get, reverse = True)
            # print "TOP INGREDIENTS: "
            # for ingredient in c_sort[0:5]:
            #     print ingredient + " " + str(c[ingredient])

            possible_splits = set()
            # for (ingredients, cuisine) in proj_database:
            #     for ingredient in ingredients:
            #         possible_splits.add(ingredient)
            for ingredient in c_sort[0:200]:
                possible_splits.add(ingredient)

        # Pre-fetch the ratio in our projected database thus far in case we need it
        # print "Getting the current best ratio..."
        cuisines = {}
        total_count = 0
        for (ingredients, cuisine) in proj_database:
            total_count += 1
            if cuisine in cuisines:
                cuisines[cuisine] += 1
            else:
                cuisines[cuisine] = 1
        best_count = 0
        best_cuisine = ""
        for cuisine in cuisines:
            if cuisines[cuisine] > best_count:
                best_count = cuisines[cuisine]
                best_cuisine = cuisine
        # print "Best ratio is " + best_cuisine + " at " + str(float(best_count)/float(total_count)) + " %"

        # TODO: We have a hardcoded threshold here (might want to fiddle with it)
        THRESHOLD = 0.95
        # TODO: What to do here?
        # Probably should use the best guess from the PREVIOUS level!
        # Pass that in during the recursion?
        if len(proj_database) == 0:
            if best_guess_sofar == None:
                best_guess_sofar = "italian"
            print "Undefined! Picking " + best_guess_sofar + " !"
            #raw_input("Press Enter to continue...")
            return LeafNode(best_guess_sofar)
        if len(possible_splits) == 0 or float(best_count)/float(total_count) >= THRESHOLD:
            print "Picking " + best_cuisine + " !"
            #raw_input("Press Enter to continue...")
            return LeafNode(best_cuisine)
        else:
            print "Searching for best partition..."
            split_criteria = self.pick_partition(proj_database, possible_splits)
            print "Deciding to split at " + split_criteria + " !"
            possible_splits.remove(split_criteria)
            no_proj = [x for x in proj_database if not (split_criteria in x[0])]
            yes_proj = [x for x in proj_database if (split_criteria in x[0])]
            print "No: " + str(len(no_proj)) + "; Yes: " + str(len(yes_proj))
            no_node = self.construct_tree(no_proj, possible_splits.copy(), best_cuisine)
            yes_node = self.construct_tree(yes_proj, possible_splits.copy(), best_cuisine)
            return InternalNode(split_criteria = split_criteria, yes_node = yes_node, no_node = no_node)

    def predict(self, ingredients):
        current_node = self._root
        while isinstance(current_node, InternalNode):
            if current_node.split_criteria in ingredients:
                current_node = current_node.yes_node
            else:
                current_node = current_node.no_node
        return current_node.decision

    def classify(self, ingredients):
        if self._root == None:
            self._root = self.construct_tree()
        return self.predict(ingredients)
