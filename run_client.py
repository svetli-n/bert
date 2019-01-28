import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2
import grpc

import tokenization
from run_classifier import QnliProcessor, from_record_to_tf_example


def make_request():
    max_seq_length = 128
    vocab_file = '/Users/svetlin/workspace/q-and-a/bert-data/cased_L-12_H-768_A-12/vocab.txt'
    data_dir = '.'
    dev_file = '/Users/svetlin/workspace/q-and-a/glue_data/QNLI/dev.tsv.short'
    train_file = '/Users/svetlin/workspace/q-and-a/glue_data/QNLI/train.tsv.short'

    channel = grpc.insecure_channel("localhost:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # Parse Description
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    processor = QnliProcessor(data_dir)
    label_list = processor.get_labels()

    with open(dev_file) as dev_fh:
        dev_lines = dev_fh.readlines()

    with open(train_file) as train_fh:
        train_lines = train_fh.readlines()

    for dev_line, train_line in zip(dev_lines[1:], train_lines[1:]):
        train_example = train_line[:-1].split('\t')
        dev_example = dev_line[:-1].split('\t')

        input_li = [dev_example for _ in range(10)]
        input_li.extend([train_example for _ in range(10)])
        inputExample = processor._create_examples(input_li, 'test')
        model_input = []

        for example in inputExample:
            tf_example = from_record_to_tf_example(3, example, label_list, max_seq_length, tokenizer)
            model_input.append(tf_example.SerializeToString())

        # Send request
        # See prediction_service.proto for gRPC request/response details.
        model_request = predict_pb2.PredictRequest()
        model_request.model_spec.name = 'my_model'
        model_request.model_spec.signature_name = 'serving_default'

        # dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=2)]
        # tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
        # tensor_proto = tensor_pb2.TensorProto(
        #     dtype=types_pb2.DT_STRING,
        #     tensor_shape=tensor_shape_proto,
        #     string_val=[model_input])

        tensor_proto = tf.contrib.util.make_tensor_proto(model_input, shape=[len(inputExample)])

        model_request.inputs['examples'].CopyFrom(tensor_proto)
        result = stub.Predict(model_request, 10.0)  # 10 secs timeout
        result = tf.make_ndarray(result.outputs["output"])
        # pretty_result = "Predicted Label: " + label_list[result[0].argmax(axis=0)]
        # tf.logging.info("Predicted Label: %s", label_list[result[0].argmax(axis=0)])
        # tf.logging.info('Result: %s', pretty_result)
        print(result)
        # print(pretty_result)


if __name__ == '__main__':
    make_request()
