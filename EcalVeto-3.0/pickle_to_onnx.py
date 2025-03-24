import pickle
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import argparse
from ast import arg
import os

print('All packages imported successfully')

parser = argparse.ArgumentParser(description='Converts a pkl filetype to onnx filetype.')
parser.add_argument('-i','--infile', type=str, action='store', dest='infile', help='Absolute path to input file')
parser.add_argument('-o','--outdir', type=str, action='store', dest='outdir', help='Directory of output file, absolute path')
args = parser.parse_args()

with open(args.infile, "rb") as f:
    clf = pickle.load(f)
    outfilename = os.path.basename(args.infile).replace('.pkl', '.onnx')
    feature_names = clf.feature_names

# the xgboost converter in onnxmltools can only parse default feature names assigned
# by xgboost, i.e. f0, f1, f2, etc. The below block of code checks whether the feature
# names assigned in the process of training the model are default or otherwise out of
# order and thus unparseable by the onnx converter. If they are non-default, it
# temporarily assigns default feature names in the same order as the read-in feature
# names (changing the feature names in memory but NOT in the pickle file or the model
# itself) so that the onnx converter can properly convert the xgboost.Booster() object
# to an onnx filetype.

if not all(feat == 'f%d' % i for i, feat in enumerate(feature_names)):
    print('Feature names are either non-default or out of order.\nAssigning default names for onnx converter...')
    clf.feature_names = ['f%d' % i for i in range(len(feature_names))]
    print('Feature names are assigned. NOTE: This does not change the source pickle file.')
    # print(clf.feature_names)
else:
    print('Feature names are default and in the correct order.')

outfilepath = os.path.join(args.outdir, outfilename)

n_features = clf.num_features()
onnx_model = onnxmltools.convert_xgboost(clf, initial_types=[('input', FloatTensorType([None, n_features]))])

with open(outfilepath, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"ONNX model saved as {outfilename} in {args.outdir}")
