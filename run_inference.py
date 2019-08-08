from __future__ import print_function, division

import logging
import os
import sys
from argparse import ArgumentParser, SUPPRESS
from datetime import datetime
import cv2
from openvino.inference_engine import IENetwork, IECore

logging.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=logging.INFO,
                    stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to a image/video file. (Specify 'cam' to work with "
                                            "camera)", required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Labels mapping file", default=None, type=str)
    args.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering", default=0.4, type=float)
    args.add_argument("-ni", "--number_iter", help="Optional. Number of inference iterations", default=1, type=int)
    args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=False,
                      action="store_true")
    args.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
                      default=False, action="store_true")
    return parser


def main():
    args = build_argparser().parse_args()

    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + '.bin'

    # Plugin initialization for specified device and load extensions
    # library if specified.
    log.info("Creating Inference Engine...")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    n, c, w, h = net.inputs['image'].shape
    net.batch_size = n

    log.info('Loading IR to the plugin...')
    exec_net = ie.load_network(network=net, device_name=args.device,
                               num_requests=2)
    del net

    frame = cv2.imread(args.input)
    frame = cv2.resize(frame, (w, h))
    frame = frame.reshape((n, c, h, w))

    log.info("Start inference")
    start_time = datetime.now()
    print(start_time)

    for i in range(60):
        exec_net.infer({'image': frame})

    end_time = datetime.now()
    infer_time = end_time - start_time
    print(end_time)

    log.info("Finish inference")
    log.info("Inference time is {}".format(infer_time))

    del exec_net
    del ie


if __name__ == '__main__':
    sys.exit(main() or 0)
