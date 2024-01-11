import argparse

# Create the argument parser
def parse_option():
    parser = argparse.ArgumentParser(description='Parser for AutoEncoder')

    # Add arguments
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--savedir', type=str, help='Save model directory')
    parser.add_argument('--testname', type=str, help='Name of this test (or model)')
    parser.add_argument('--output', type=str, help='Output file path')
    # parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Mode (train or test)')
    parser.add_argument('--modelpath', type=str, help='Model path')
    parser.add_argument('--imagepath', type=str, help='Image save path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batchs', type=int, default=16, help='Number of batch size')
    parser.add_argument('--lr', type=int, default=0.001, help='Number of learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of threads')
    parser.add_argument('--encodesize', type=int, default=16, help='Encoder input image size (Square)')
    parser.add_argument('--decodesize', type=int, default=128, help='Decoder input image size (Square)')
    parser.add_argument('--devices', type=str, default="1", help='Selcet devices ("-1" for CPU, default is GPU)')
    # visulize
    parser.add_argument('--th_percent', type=float, default=None, help='filter smura')
    parser.add_argument('--min_area', type=int, default=None, help='Filter noise')

    return parser.parse_args()