from pathlib import Path

import odsyntds
import logging

def parse_arguments():
    """
    Parse the cli arguments
    """

    parser = argparse.ArgumentParser(
        description="Overlay objects images onto backgrounds images with associated bboxes with or without augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-t",
        "--type",
        help="specify the type of overlay to be performed",
        type=str,
        choices=["overlay", "augmented_overlay"],
        default="overlay",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--objects",
        required=False,
        type=str,
        default="data/rawObjects",
        help="Path to objects' images to be overlayed",
    )
    parser.add_argument(
        "-b",
        "--background",
        required=False,
        type=str,
        default="data/backgrounds",
        help="Path to backgrounds' images",
    )
    parser.add_argument(
        "-r",
        "--resize",
        required=False,
        type=int,
        default=None,
        help="Resize the biggest dimension to specified size in px",
    )
    parser.add_argument(
        "--no_alpha",
        required=False,
        action="store_true",
        help="Convert the image with objects overlayed to rgb instead of rgba",
    )
    parser.add_argument(
        "--save_bboxes",
        required=False,
        action="store_true",
        help="save the image with objects overlayed with bbox displayed",
    )

    parser.add_argument(
        "-f",
        "--format",
        required=False,
        default="yolo",
        help="choose bounding box format",
        choices=["yolo", "SSD"],
    )

    parser.add_argument(
        "-n",
        "--number",
        required=False,
        default="1",
        help="Number of training samples to generate per background",
    )

    parser.add_argument(
        "-s",
        "--save",
        required=False,
        type=str,
        default="trainingSet",
        help="Path to save dataset",
    )

    return parser.parse_args()

logger = logging.getLogger()
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logFile = Path('script.log')
if logFile.exists():
    logFile.unlink()
if not logFile.parent.exists():
    logFile.parent.mkdir(parents=True)
fh = logging.FileHandler(logFile)
fh.setFormatter(fmt)
sh = logging.StreamHandler() #sys.stdout
sh.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(sh)
logger.setLevel(logging.INFO)

print(f"Log file is available at {fh.baseFilename}")

args = parse_arguments()

logger.info("Executing odsyntds create_ds function")

aug = False
if args.type == 'augmented_overlay':
    aug = True
    
odsyntds.create_ds(args.objects,
                args.background,
                save_path = args.save,
                perBkg=args.number,
                format=args.format,
                aug=aug,
                resize=args.resize,
                no_alpha=args.no_alpha,
                save_bboxes=args.save_bboxes)