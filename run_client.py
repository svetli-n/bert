import collections
import json
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2
import grpc
from elasticsearch import Elasticsearch

import tokenization
from run_classifier import QnliProcessor, from_record_to_tf_example

from flask import Flask, request

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

es = Elasticsearch()
es_index_name = 'test-index'

model_request = predict_pb2.PredictRequest()
model_request.model_spec.name = 'my_model'
model_request.model_spec.signature_name = 'serving_default'

max_seq_length = 128
vocab_file = '/Users/svetlin/workspace/q-and-a/bert-data/cased_L-12_H-768_A-12/vocab.txt'
data_dir = '.'

data_file = '/Users/svetlin/workspace/q-and-a/glue_data/QNLI/dev.tsv'

tf_serving_host = 'localhost:8500'


def make_tf_request(q_and_a=None):
    if not q_and_a:
        raise ValueError('q_and_a can not be empty')

    channel = grpc.insecure_channel(tf_serving_host)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    processor = QnliProcessor(data_dir)
    label_list = processor.get_labels()

    # skipping the first header line
    input_example = processor._create_examples([None] + q_and_a, 'test')

    model_input = []

    for example in input_example:
        tf_example = from_record_to_tf_example(3, example, label_list, max_seq_length, tokenizer)
        model_input.append(tf_example.SerializeToString())

    tensor_proto = tf.contrib.util.make_tensor_proto(model_input, shape=[len(input_example)])
    model_request.inputs['examples'].CopyFrom(tensor_proto)
    # start = time.time()
    predictions = stub.Predict(model_request, 10.0)
    # end = time.time(); took = end - start; print(f'Hello, World! took: {took} secs')
    probs = tf.make_ndarray(predictions.outputs['output'])
    q_and_a = np.array(q_and_a)
    predictions = np.concatenate((probs, q_and_a), axis=1)
    df = pd.DataFrame(predictions)
    for i in [0, 1]:
        df[i] = df[i].astype('float32')
    df.sort_values(by=0, ascending=False, inplace=True)
    return df


def index(filename):
    with open(filename) as dev_fh:
        dev_lines = dev_fh.readlines()

    # skipping the first header line
    for line in dev_lines[1:]:
        items = line.split('\t')
        doc = {
            'id': items[0],
            'question': items[1],
            'answer': items[2],
            'label': items[3][:-1],
        }
        # consider bulk index api instead
        res = es.index(index=es_index_name, doc_type='_doc', id=None, body=doc)
        app.logger.debug(res['result'])
    es.indices.refresh(index=es_index_name)


def search(question, num_answers=None):
    result = es.search(
        index=es_index_name,
        body={'query': {'match_phrase': {'question': question}}},
        size=num_answers or 10
    )
    app.logger.debug('Got %d results:' % len(result['hits']['hits']))
    return [[f'guid-{i}', question, hit['_source']['answer'], hit['_source']['label']] for i, hit in
            enumerate(result['hits']['hits'])]


@app.route('/predict', methods=['GET'])
def predict():
    question = request.args.get('question')
    q_and_a = search(question)
    df = make_tf_request(q_and_a)
    df_dict = collections.OrderedDict(df.to_dict())
    return json.dumps(df_dict)


if __name__ == '__main__':
    '''
    HOWTO run in console:
        FLASK_DEBUG=1 python -m flask run
    '''
    index(data_file)
