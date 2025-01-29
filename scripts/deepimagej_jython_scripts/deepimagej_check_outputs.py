"""
Python script that checks that the wanted objects have actually been created and correspond to the expected ones
"""

import json
import os
import argparse

from net.imglib2.img.display.imagej import ImageJFunctions

from ij import IJ


def find_expected_output(outputs_dir, name):
    for ff in os.listdir(outputs_dir):
        if ff.endswith("_" + name + ".tif") or ff.endswith("_" + name + ".tiff"):
            return os.path.join(outputs_dir, ff)
    raise Exception("Expected output for " + name + " not found")


def main(model_dir):
    with open(os.path.join(model_dir, os.getenv("JSON_OUTS_FNAME")), 'r') as f:
        expected_outputs = json.load(f)

        for output in expected_outputs:
            name = output["name"]
            dij_output_path = output["dij"]
            expected_output = output["expected"]
            if not os.path.exists(dij_output_path) or len(os.listdir(dij_output_path)) != len(expected_outputs):
                raise Exception("Output " + name + " was not generated by deepimagej")
            elif not os.path.exists(expected_output):
                raise Exception("Cannot find expected output " + name)
            dij_output = find_expected_output(dij_output_path, name)
            dij_rai = ImageJFunctions.wrap(IJ.openImage(dij_output))
            expected_rai = ImageJFunctions.wrap(IJ.openImage(expected_output))
            dij_shape = dij_rai.dimensionsAsLongArray()
            expected_shape = expected_rai.dimensionsAsLongArray()
            assert dij_shape == expected_shape, "Output " + name + " in deepimagej has different shape " + str(dij_shape) + " vs " + str(expected_shape)
            dij_cursor = dij_rai.cursor()
            expected_cursor = expected_rai.cursor()
            while dij_cursor.hasNext():
                dij_cursor.fwd()
                expected_cursor.fwd()
                if dij_cursor.get().getRealFloat() - expected_cursor.get().getRealFloat() > (1.5 * 1e-4) \
                    or expected_cursor.get().getRealFloat() - dij_cursor.get().getRealFloat() > (1.5 * 1e-4):
                    raise Exception("Values of output " + name  + " differ")


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument('-model_dir', type=str, required=True)


    # Parse the arguments
    args = parser.parse_args()

    model_dir = args.model_dir
    main(model_dir)