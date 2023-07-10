# triton + batch + http client + ensemble pipeline
#!/usr/bin/env python

import argparse, os, sys
import numpy as np
import tritonclient.http as client
from line_profiler import LineProfiler

profile = LineProfiler()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('--batch_size', type=int, required=False, default=1,
                        help='Batch size')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('-m', '--model_name', type=str, required=False, default="dali_backend",
                        help='Model name')
    parser.add_argument('-i', '--input_name', type=str, required=False, default="INPUT",
                        help='Input name')
    parser.add_argument('-o', '--output_name', type=str, required=False, default="OUTPUT",
                        help='Output name. For multi output layer, ":" is default delimiter, e.g OUTPUT1:OUTPUT2')
    parser.add_argument('--statistics', action='store_true', required=False, default=False,
                        help='Print tritonserver statistics after inferring')
    return parser.parse_args()

def string_to_np(string) -> np.array:
    encoded = np.array([str(x).encode('utf-8') for x in string],
                       dtype=np.object_)
    return encoded

def generate_inputs(input_name, input_shape, input_dtype):
    return [client.InferInput(input_name, input_shape, input_dtype)]

def generate_outputs(output_names):
    output_layer = []
    for output_layer_name in output_names.split(":"):
        output_layer.append(client.InferRequestedOutput(output_layer_name))
    return output_layer


@profile
def main():
    FLAGS = parse_args()
    try:
        triton_client = client.InferenceServerClient(url=FLAGS.url,
                                                     verbose=FLAGS.verbose)
    except Exception as e:
        print("[ FAIL ] - channel creation failed: " + str(e))
        sys.exit()


    model_name = FLAGS.model_name
    model_version = -1

    # create raw data
    # print("Loading input")
    text_data = ["Hello, I'm Machine Learning Engineer, my duty is "]*FLAGS.batch_size
    batch = string_to_np(text_data).reshape(FLAGS.batch_size,-1)

    # Initialize the data 
    inputs = generate_inputs(FLAGS.input_name, batch.shape, "BYTES")
    outputs = generate_outputs(FLAGS.output_name)

    # Store the data
    inputs[0].set_data_from_numpy(batch, binary_data=True)
    # Test with outputs
    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    # Get the output arrays from the results
    outputs = []
    for output_layer_name in FLAGS.output_name.split(":"):
        outputs.append(results.as_numpy(output_layer_name))

    # for idx, data in enumerate(outputs):
    #     print(f"[Output] - {idx} : ",data)

    # statistics = triton_client.get_inference_statistics(model_name="opt_125m_tokenizer")
    statistics = triton_client.get_inference_statistics(model_name=FLAGS.model_name)

    if len(statistics["model_stats"]) != 1:
        print("[ FAIL ]: No Inference Statistics")
        sys.exit(1)
    if FLAGS.statistics:
        print(statistics)

    print(f'[ PASS ]: infer - {len(outputs)}, shape: {[x.shape for x in outputs]}')


if __name__ == '__main__':
    main()
    # profile.print_stats()
