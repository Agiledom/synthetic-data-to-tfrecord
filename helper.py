import argparse


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.5 or x > 1.5:
        raise argparse.ArgumentTypeError("%r not in range [0.5, 1.5]" % (x,))
    return x


def load_file_from_gcp(path):
    with tf.python.lib.io.file_io.FileIO(path, 'rb') as f:
        return f


def save_image_to_gcp(file, path):
    with tf.python.lib.io.file_io.FileIO(path, 'wb') as f:
        file.save(fp=f, format="png")


def save_file_to_gcp(content, path, type=''):
    with tf.python.lib.io.file_io.FileIO(path, 'wb') as f:
        if type == 'json':
            f.write(json.dumps(content))
        else:
            f.write(content)
