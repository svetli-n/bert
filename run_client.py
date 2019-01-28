import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2
import grpc
from elasticsearch import Elasticsearch

import tokenization
from run_classifier import QnliProcessor, from_record_to_tf_example


def make_tf_request(answers=None):

    if not answers:
        raise ValueError('answers can not be empty')

    max_seq_length = 128
    vocab_file = '/Users/svetlin/workspace/q-and-a/bert-data/cased_L-12_H-768_A-12/vocab.txt'
    data_dir = '.'

    channel = grpc.insecure_channel("localhost:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    processor = QnliProcessor(data_dir)
    label_list = processor.get_labels()


    model_input = []

    inputExample = processor._create_examples([None]+answers, 'test')

    for example in inputExample:
        tf_example = from_record_to_tf_example(3, example, label_list, max_seq_length, tokenizer)
        model_input.append(tf_example.SerializeToString())

    model_request = predict_pb2.PredictRequest()
    model_request.model_spec.name = 'my_model'
    model_request.model_spec.signature_name = 'serving_default'
    tensor_proto = tf.contrib.util.make_tensor_proto(model_input, shape=[len(inputExample)])
    model_request.inputs['examples'].CopyFrom(tensor_proto)
    result = stub.Predict(model_request, 10.0)  # 10 secs timeout
    result = tf.make_ndarray(result.outputs["output"])
    # pretty_result = "Predicted Label: " + label_list[result[0].argmax(axis=0)]
    # tf.logging.info("Predicted Label: %s", label_list[result[0].argmax(axis=0)])
    # tf.logging.info('Result: %s', pretty_result)
    print(result)
        # print(pretty_result)


def index():
    es = Elasticsearch()
    dev_file = '/Users/svetlin/workspace/q-and-a/glue_data/QNLI/dev.tsv.short'
    with open(dev_file) as dev_fh:
        dev_lines = dev_fh.readlines()

    for line in dev_lines[1:]:
        items = line.split('\t')
        doc = {
            'id': items[0],
            'question': items[1],
            'answer': items[2],
            'label': items[3][:-1],
        }
        res = es.index(index="test-index", doc_type='_doc', id=doc['id'], body=doc)
        print(res['result'])
    es.indices.refresh(index="test-index")


def search(question):
    es = Elasticsearch()
    res = es.search(index="test-index", body={"query": {"match": {'answer': question}}})
    print("Got %d Hits:" % res['hits']['total'])
    return [[f'guid-{i}', question, hit["_source"]['answer'], 'entailment'] for i, hit in enumerate(res['hits']['hits'])]


if __name__ == '__main__':
    # make_tf_request()
    # index()
    answers = search('to')
    make_tf_request(answers)
