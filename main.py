import os
import argparse
import generate_dataset
import generate_tfrecord
import helper

parser = argparse.ArgumentParser(description='Create synthetic output_train data for object detection algorithms.')
# directory based arguments
parser.add_argument("-bkg", "--backgrounds", type=str, default="input/backgrounds/",
                    help="Path to background images folder.")
parser.add_argument("-obj", "--objects", type=str, default="input/objects/",
                    help="Path to object images folder.")
parser.add_argument("-img", "--images", type=str, default="synthetic/images/",
                    help="Path to synthetic images folder.")
parser.add_argument("-dat", "--data", type=str, default="synthetic/data/",
                    help="Path to synthetic images folder.")
# Image based arguments
parser.add_argument("-grp", "--groups", type=bool, default=True,
                    help="Include groups of objects in output_train set?")
parser.add_argument("-cnt", "--count", type=int, default=4, help="Number of locations for each object size")
parser.add_argument("-sze", "--sizes", type=helper.restricted_float, default=[0.7, 0.9, 1.1, 1.3], nargs='+',
                    help="A list of floats to define at what size ratio the object will be placed on the background."
                         "Note: must be between 0.5 and 1.5")
# data based arguments
parser.add_argument("-ann", "--annotate", type=bool, default=True,
                    help="Generate csv and json annotations with labels and bounding box co-ordinates")
parser.add_argument("-tfr", "--tfrecord", type=bool, default=True,
                    help="Generate tfrecord files")
# cloud based arguments
parser.add_argument("-cld", "--cloud", type=bool, default=False,
                    help="set True if running in GCP.")
args = parser.parse_args()

bucket = os.environ['BUCKET'] if cloud else None

# adjust paths based on whether we are running in the cloud
bkgs_path = f"gs://{bucket}/{args.backgrounds}" if cloud else args.backgrounds
objs_path = f"gs://{bucket}/{args.objects}" if cloud else args.objects
output_images = f"gs://{bucket}/{args.images}" if cloud else args.images
output_data = f"gs://{bucket}/{args.data}" if cloud else args.data
label_map_input = f"gs://{bucket}/synthetic/data/labelmap.pbtxt" if cloud else "synthetic/data/labelmap.pbtxt"
csv_input = f"gs://{bucket}/synthetic/data/annotations.csv" if cloud else "synthetic/data/annotations.csv"
output_path = f"gs://{bucket}" + "/synthetic/data/{}_{}_{}.record" if cloud else "synthetic/data/{}_{}_{}.record"

# generate the dataset
generate_dataset.main(bkgs_path=bkgs_path, objs_path=objs_path, output_images=output_images,
                      output_data=output_data, annotate=args.annotate, groups=args.groups,
                      count_per_size=args.count, sizes=args.sizes, cloud=args.cloud)

if args.tfrecord:
    generate_tfrecord.main(label_map_input=label_map_input, images_path=output_images,
                           csv_input=csv_input, output_path=output_path, cloud=args.cloud)

