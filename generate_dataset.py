import os
import json

from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from tqdm import tqdm
import cv2
import tensorflow as tf

from helper import load_file_from_gcp, save_image_to_gcp, save_file_to_gcp

# list for the annotations
annotations = []


# Helper functions
def get_obj_positions(obj, bkg, sizes, count=1):
    obj_w, obj_h = [], []
    x_positions, y_positions = [], []
    bkg_w, bkg_h = bkg.size
    # Rescale our obj to have a couple different sizes
    obj_sizes = [tuple([int(s * x) for x in obj.size]) for s in sizes]
    for w, h in obj_sizes:
        obj_w.extend([w] * count)
        obj_h.extend([h] * count)
        max_x, max_y = bkg_w - w, bkg_h - h
        x_positions.extend(list(np.random.randint(0, max_x, count)))
        y_positions.extend(list(np.random.randint(0, max_y, count)))
    return obj_h, obj_w, x_positions, y_positions


def get_box(obj_w, obj_h, max_x, max_y):
    x1, y1 = np.random.randint(0, max_x, 1), np.random.randint(0, max_y, 1)
    x2, y2 = x1 + obj_w, y1 + obj_h
    return [x1[0], y1[0], x2[0], y2[0]]


# check if two boxes intersect
def intersects(box, new_box):
    box_x1, box_y1, box_x2, box_y2 = box
    x1, y1, x2, y2 = new_box
    return not (box_x2 < x1 or box_x1 > x2 or box_y1 > y2 or box_y2 < y1)


def get_group_obj_positions(obj_group, bkg, objs_path, obj_images):
    bkg_w, bkg_h = bkg.size
    boxes = []
    objs = [Image.open(objs_path + obj_images[i]) for i in obj_group]
    obj_sizes = [tuple([int(0.6 * x) for x in i.size]) for i in objs]
    for w, h in obj_sizes:
        # set background image boundaries
        max_x, max_y = bkg_w - w, bkg_h - h
        # get new box coordinates for the obj on the bkg
        while True:
            new_box = get_box(w, h, max_x, max_y)
            for box in boxes:
                res = intersects(box, new_box)
                if res:
                    break

            else:
                break  # only executed if the inner loop did NOT break
            # print("retrying a new obj box")
            continue  # only executed if the inner loop DID break
        # append our new box
        boxes.append(new_box)
    return obj_sizes, boxes


def adjust_brightness(img, value):
    img = cv2.imread(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def main(bkgs_path, objs_path, output_images, output_data,
         annotate, groups, count_per_size, sizes, cloud):
    bkg_images = tf.io.gfile.listdir(bkgs_path) if cloud else [f for f in os.listdir(bkgs_path) if not f.startswith(".")]
    obj_images = tf.io.gfile.listdir(objs_path) if cloud else [f for f in os.listdir(objs_path) if not f.startswith(".")]
    total_images = (len(bkg_images) * len(obj_images) * len(sizes) * count_per_size) + \
                   (2 * len(obj_images) * len(bkg_images))
    n = 1
    print(f"[STARTING] Generating {total_images} synthetic images...", flush=True)

    with tqdm(total=total_images) as pbar:
        # Make synthetic output_train data
        for bkg in bkg_images:
            # Load the background image
            bkg_path = bkgs_path + bkg
            bkg_img = Image.open(load_file_from_gcp(bkg_path) if cloud else bkg_path).convert("RGBA")
            bkg_x, bkg_y = bkg_img.size

            # Do single objs first
            for i in obj_images:
                # Load the single obj
                i_path = objs_path + i
                obj_img = Image.open(load_file_from_gcp(i_path) if cloud else i_path).convert("RGBA")
                # Get an array of random obj positions (from top-left corner)
                obj_h, obj_w, x_pos, y_pos = get_obj_positions(
                    obj=obj_img, bkg=bkg_img, sizes=sizes, count=count_per_size
                )

                # Create synthetic images based on positions
                for h, w, x, y in zip(obj_h, obj_w, x_pos, y_pos):
                    # Copy background
                    bkg_w_obj = bkg_img.copy()
                    # Adjust obj size
                    new_obj = obj_img.resize(size=(w, h))
                    # Adjust objects brightness
                    new_obj = ImageEnhance.Brightness(new_obj).enhance(
                        np.around(np.random.uniform(0.6, 1.2, 1), 1)[0]
                    )
                    # Paste on the obj
                    bkg_w_obj.paste(new_obj, (x, y), mask=new_obj)
                    output_fp = output_images + str(n) + ".png"
                    filename = str(n) + ".png"
                    # Save the image
                    if cloud:
                        save_image_to_gcp(bkg_w_obj, output_fp)
                    else:
                        bkg_w_obj.save(fp=output_fp, format="png")

                    # update the progress bar
                    pbar.update(1)
                    if annotate:
                        annotation = {
                            'filename': filename,
                            'height': bkg_y,
                            'width': bkg_x,
                            'xmin': int(x),
                            'xmax': int(x + w),
                            'ymin': int(y),
                            'ymax': int(y + h),
                            'class': i.split(".png")[0]
                        }
                        # Save the annotation data
                        annotations.append(annotation)
                    # print(n)
                    n += 1

            if groups:
                # 24 Groupings of 2-4 objs together on a single background
                groups = [np.random.randint(0, len(obj_images) - 1, np.random.randint(2, 5, 1)) for r in
                          range(2 * len(obj_images))]
                # For each group of objs
                for group in groups:
                    # Get sizes and positions
                    obj_sizes, boxes = get_group_obj_positions(
                        group, bkg_img, objs_path, obj_images)
                    bkg_w_obj = bkg_img.copy()

                    # For each obj in the group
                    for i, size, box in zip(group, obj_sizes, boxes):
                        # Get the obj
                        obj = Image.open(
                            objs_path + obj_images[i]).convert("RGBA")
                        # adjust the brightness of the object
                        obj = ImageEnhance.Brightness(obj).enhance(
                            np.around(np.random.uniform(0.6, 1.2, 1), 1)[0]
                        )
                        obj_w, obj_h = size
                        # Resize it as needed
                        new_obj = obj.resize((obj_w, obj_h))
                        x_pos, y_pos = box[:2]
                        output_fp = output_images + str(n) + ".png"
                        filename = str(n) + ".png"
                        if annotate:
                            # annotations
                            annotation = {
                                'filename': filename,
                                'height': bkg_y,
                                'width': bkg_x,
                                'xmin': int(x_pos),
                                'xmax': int(x_pos + obj_w),
                                'ymin': int(y_pos),
                                'ymax': int(y_pos + obj_h),
                                'class': obj_images[i].split(".png")[0]
                            }
                            # Add obj annotation
                            annotations.append(annotation)
                        # Paste the obj to the background
                        bkg_w_obj.paste(new_obj, (x_pos, y_pos), mask=new_obj)

                    # Save image
                    if cloud:
                        save_image_to_gcp(bkg_w_obj, output_fp)
                    else:
                        bkg_w_obj.save(fp=output_fp, format="png")
                    # update the progress bar
                    pbar.update(1)
                    # print(n)
                    n += 1

        if annotate:
            # Save annotations
            if cloud:
                save_file_to_gcp(annotations, output_data + "annotations.json", type='json')
            else:
                with open(output_data + "annotations.json", "w") as f:
                    f.write(json.dumps(annotations))
            # load into data frame
            df = pd.read_json(load_file_from_gcp(output_data + "annotations.json") if cloud else
                              output_data + "annotations.json")
            # create csv files
            if cloud:
                csv = df.to_csv(index=None)
                save_file_to_gcp(csv, output_data + "annotations.csv", type='csv')
            else:
                df.to_csv(output_data + "annotations.csv", index=None)
            # create a labelmap (.ppbtxt)
            categories = df["class"].unique()
            end = '\n'
            s = ' '
            out = ''
            for ID, name in enumerate(categories):
                out += 'item' + s + '{' + end
                out += s * 2 + 'id:' + ' ' + (str(ID + 1)) + end
                out += s * 2 + 'name:' + ' ' + '\'' + name + '\'' + end
                out += '}' + end * 2
            if cloud:
                save_file_to_gcp(out, output_data + "labelmap.pbtxt", type='txt')
            else:
                with open(output_data + "labelmap.pbtxt", 'w') as f:
                    f.write(out)

    print(f"[SUCCESS] Successfully created {total_images} synthetic images", flush=True)
