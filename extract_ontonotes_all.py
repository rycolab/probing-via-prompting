import collections
import json
import logging as log
import os
import sys
from typing import Dict, List, Tuple
import random

import numpy as np
from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from tqdm import tqdm

from utils import LABEL_DICT

CONTROL_MAPPING = {}

def _span_to_string(sentence, span: Tuple[int, int]):
    return " ".join(sentence.split(" ")[span[0]:span[1] + 1])

def _make_target(label: List[str], span1: Tuple[int, int], span2: Tuple[int, int] = None):
    t = {"span1": span1, "label": label}
    if span2 is not None:
        t["span2"] = span2
    return t


def make_record(spans, sentence):
    record = {}
    record["text"] = " ".join(sentence.words)
    record["targets"] = [_make_target(*s) for s in spans]
    return record

def constituents_to_record(parse_tree):
    """Function converting Tree object to dictionary compatible with common JSON format
     copied from ptb_process.py so it doesn't have dependencies
    """
    punctuations = ["-LRB-", "-RRB-", "-LCB-", "-RCB-", "-LSB-", "-RSB-"]

    pos_record = {}
    pos_record["text"] = " ".join(parse_tree.flatten())
    pos_record["targets"] = []

    non_record = {}
    non_record["text"] = " ".join(parse_tree.flatten())
    non_record["targets"] = []

    pos_control_record = {}
    pos_control_record["text"] = " ".join(parse_tree.flatten())
    pos_control_record["targets"] = []
    labels = list(LABEL_DICT['pos'].keys())
    num_labels = len(labels)

    for i, leaf in enumerate(parse_tree.subtrees(lambda t: t.height() == 2)):
        # modify the leafs by adding their index in the parse_tree
        leaf[0] = (leaf[0], str(i))

    for subtree in parse_tree.subtrees():
        assoc_words = subtree.leaves()
        assoc_words = [(i, int(j)) for i, j in assoc_words]
        assoc_words.sort(key=lambda elem: elem[1])
        indices = [int(assoc_words[0][1]), int(assoc_words[-1][1])]
        span = " ".join([word[0] for word in assoc_words])
        
        tmp_tag_list = subtree.label().replace("=", "-").replace("|", "-").split("-")
        label = tmp_tag_list[0]
        # Special cases:
        if len(tmp_tag_list) > 1 and tmp_tag_list[1] == "S":  # Case when we have 'PRP-S' or 'WP-S'
            label = tmp_tag_list[0] + "-" + tmp_tag_list[1]
        if (
            subtree.label() in punctuations
        ):  # Case when we have one of the strange punctions, such as round brackets
            label = subtree.label()
        target = {"span1": indices, "label": label}
        
        if subtree.height() == 2:
            pos_record["targets"].append(target)

            if span not in CONTROL_MAPPING:
                CONTROL_MAPPING[span] = labels[random.randint(0, num_labels - 1)]
            control_label = CONTROL_MAPPING[span]
            control_target = {"span1": indices, "label": control_label}
            
            pos_control_record["targets"].append(control_target)

        elif subtree.height() > 2:
            non_record['targets'].append(target)

    return pos_record, pos_control_record, non_record


def get_frames(sentence):
    for frame, bio_tags in sentence.srl_frames:
        frame_targets = []
        spans = bio_tags_to_spans(bio_tags)
        head_span = None
        other_spans = []
        for (tag, indices) in spans:
            if tag == "V":
                head_span = indices
            else:
                other_spans.append((tag, indices))
        if head_span is None:
            print(frame, bio_tags)
        for span2_tag, span2 in other_spans:
            frame_targets.append((span2_tag, head_span, span2))
        yield frame_targets

def find_links(span_list):
    pairs = []
    for i, span1 in enumerate(span_list):
        for span2 in span_list[i + 1 :]:
            pairs.append((str(span1[0] == span2[0]), span1[1], span2[1]))
    return pairs

def process_ontonotes(ontonotes_reader):
    records = {}
    records['ner'], records['pos'], records['pos_control'], records['const'], records['coref'], records['srl'] = [], [], [], [], [], []
    for sentence in ontonotes_reader:
        # NER
        spans = bio_tags_to_spans(sentence.named_entities)
        if spans:
            records['ner'].append(make_record(spans, sentence))

        # POS and constituent
        if sentence.parse_tree is not None:
            pos_record, pos_control_record, const_record = constituents_to_record(sentence.parse_tree)
            records['pos'].append(pos_record)
            records['pos_control'].append(pos_control_record)
            records['const'].append(const_record)

        # coreference 
        spans = find_links(list(sentence.coref_spans))
        if spans: 
            records['coref'].append(make_record(spans, sentence))

        # SRL
        for frame_spans in get_frames(sentence):
            if frame_spans:
                records['srl'].append(make_record(frame_spans, sentence))
    
    return records

def make_patterns(records, label_dict):
    patterns = []
    for record in records:
        sentence = record['text']
        prompt = f'{sentence}'
        for target in record['targets']:
            span1 = _span_to_string(sentence, target['span1'])
            temp = prompt + f"<sep>{span1}"
            if 'span2' in target:
                span2 = _span_to_string(sentence, target['span2'])
                temp += f"<sep>{span2}"
            patterns.append({"text": temp + f"<|endoftext|>{label_dict[target['label']]}"})
    return patterns

def write_json_data(fname, lines):
    with open(fname, 'w') as fd:
        for line in lines:
            fd.write(json.dumps(line))
            fd.write("\n")


def main(args):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ontonotes",
        type=str,
        default="conll-formatted-ontonotes-5.0",
        help="Path to OntoNotes, e.g. /path/to/conll-formatted-ontonotes-5.0",
    )
    parser.add_argument(
        "--tasks", 
        default=["pos", "const", "coref", "ner", "srl", "pos_control"],
        type=str, nargs="+", help="Tasks, one or more of {pos, const, coref, ner, srl}."
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "development", "test", "conll-2012-test"],
        help="Splits, one or more of {train, development, test, conll-2012-test}.",
    )
    parser.add_argument(
        "-o", dest="output_dir", type=str, default="ontonotes/", help="Output directory for JSON files."
    )
    args = parser.parse_args(args)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    ontonotes = Ontonotes()
    for split in args.splits:
        source_path = os.path.join(args.ontonotes, "data", split)
        ontonotes_reader = ontonotes.dataset_iterator(file_path=source_path)
        converted_records = process_ontonotes(tqdm(ontonotes_reader))
        for task in args.tasks:
            pp_task_dir = os.path.join(args.output_dir, "pp", task)
            dp_task_dir = os.path.join(args.output_dir, "dp", task)
            if not os.path.isdir(pp_task_dir):
                os.makedirs(pp_task_dir)
            if not os.path.isdir(dp_task_dir):
                os.makedirs(dp_task_dir)

            write_json_data(os.path.join(dp_task_dir, f"{split}.json"), converted_records[task])

            if 'control' in task:
                label_dict = LABEL_DICT[task.replace("_control", "")]
            else:
                label_dict = LABEL_DICT[task]
            patterns = make_patterns(converted_records[task], label_dict)
            write_json_data(os.path.join(pp_task_dir, f"{split}.json"), patterns)



if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)