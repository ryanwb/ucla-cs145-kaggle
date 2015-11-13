"""
classify.py

Main command line tool to run cuisine classifiers

Example usage:
python classify.py -t -p 0.1 --trainfile train.json -a random
"""
import argparse
from cuisinedatabase import *
from randomclassifier import RandomClassifier

def main():

    parser = argparse.ArgumentParser(description="classifiy ingredients of a recipe to a cuisine type")

    parser.add_argument("-t", "--test", action="store_true",
                    help="re-assign some of the train data to test data with known class")

    parser.add_argument("-p", type=float,
                    help="for --test, probability of assigning data entry to test instead of train")
    
    parser.add_argument("--trainfile",
                    help="input training data file (known class)")

    parser.add_argument("--testfile",
                    help="input testing data file (unknown class)")

    parser.add_argument("-o", "--outputfile",
                    help="output file name (csv file)")
    
    parser.add_argument("-a", "--algorithm",
                    help="the classification algorithm to use")

    args = parser.parse_args()
    
    # parser.add_argument("-v", "--verbose", action="store_true", help="print extra output/data")

    algo = None
    if args.algorithm == "random":
        algo = RandomClassifier()

    if args.test:
        db = TestDatabase(args.trainfile)
    else:
        db = CuisineDatabase(args.trainfile, args.testfile)

    c = Classification()

    print "Starting " + args.algorithm + " training..."

    i = 0
    entry = db.train(i)
    while (entry != None):
        algo.learn(entry["ingredients"], entry["cuisine"])
        i += 1
        entry = db.train(i)

    print "Learned " + str(i) + " training data entries..."

    print "Starting " + args.algorithm + " classification..."

    i = 0
    entry = db.test(i)
    while (entry != None):
        c.save(entry["id"], algo.classify(entry["ingredients"]))
        i += 1
        entry = db.test(i)

    print "Classified " + str(i) + " test data entries..."

    if args.test:
        (n_correct, n_total) = db.accuracy(c)
        accuracy = float(n_correct)/float(n_total)
        print "n_correct: " + str(n_correct)
        print "n_total: " + str(n_total)
        if n_total > 0:
            print "accuracy: " + str(accuracy)

    if args.outputfile != None:
        c.save_csv(args.outputfile)

if __name__ == '__main__':
    main()
