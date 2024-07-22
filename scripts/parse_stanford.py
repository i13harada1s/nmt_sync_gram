import argparse
import re

import nltk
from stanza.server import CoreNLPClient

# from src.tree_utils import process_tree_string

def process_tree_string(tree_string: str):
    return re.sub("[ |\n]+", " ", tree_string)

class StanfordCoreNLPAnnotator:  
    """
    Wrapper of CoreNLPClient
    """
    def __init__(self, client):
        if type(client) is not CoreNLPClient:
            raise RuntimeError("")
        self.client = client

    def get_token(self, text, properties=None):
        ann = self.client.annotate(text, output_format="json")
        tkn = []
        for sentence in ann["sentences"]:
            tkn.append([token["word"] for token in sentence["tokens"]])
        return tkn

    def get_lemma(self, text, properties=None):
        ann = self.client.annotate(
            text, 
            properties=properties, 
            annotators='tokenize,ssplit,lemma,pos', 
            output_format="json"
        )
        lem = []
        for sentence in ann["sentences"]:
            lem.append([token["lemma"] for token in sentence["tokens"]])
        return lem

    def get_parse(self, text, properties=None):
        ann = self.client.annotate(
            text, 
            properties=properties, 
            annotators='tokenize,ssplit,lemma,pos,parse', 
            output_format="json"
        )
        par = []
        for sentence in ann["sentences"]:
            par.append([sentence["parse"]])
        return par

    # output_format=text: binarizedParseTree
    # output_format=json: binaryParse
    def get_binary_parse(self, text, properties=None):
        # properties.update({"parse.binaryTrees": True})
        ann = self.client.annotate(
            text, 
            properties=properties, 
            annotators='tokenize,ssplit,lemma,pos,parse', 
            output_format="json"
        )
        par = []
        for sentence in ann["sentences"]:
            par.append([sentence["binaryParse"]])
        return par

    #TODO: property is "basicDependencies"
    def get_dependency(self, text, properties=None):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="input file")
    parser.add_argument("--output", "-o", required=True, help="output file")
    parser.add_argument("--language", "-l", required=True, type=str, help="target language")
    parser.add_argument("--properties", "-p", default=None, help="properties file")
    parser.add_argument("--timeout", "-t", type=int, default=300000, help="timeout") # default is 300,000ms = 300s = 5m.
    parser.add_argument("--workers", type=int, default=16, help="")
    parser.add_argument("--quiet", action="store_true", help="")
    parser.add_argument("--remove-root", action="store_true", help="")
    parser.add_argument("--sentence-segment", type=str, default='\n', help="")
    args = parser.parse_args()
    
    if args.properties is not None:
        properties = args.properties
    else:
        properties = args.language
        
    with CoreNLPClient( 
        threads=args.workers, 
        timeout=args.timeout, 
        be_quiet=args.quiet
    ) as client:
        with open(args.input, 'r', encoding='utf8') as reader, \
             open(args.output, 'w', encoding='utf8') as writer:
            annotator = StanfordCoreNLPAnnotator(client)
            for line in reader:
                trees = []
                for tree in annotator.get_parse(line, properties=properties):
                    tree = nltk.tree.Tree.fromstring(process_tree_string(str(tree[0])))
                    trees.append(process_tree_string(str(tree[0] if args.remove_root else tree)))
                writer.write(args.sentence_segment.join(trees) + "\n")

if __name__ == "__main__":
    main()
