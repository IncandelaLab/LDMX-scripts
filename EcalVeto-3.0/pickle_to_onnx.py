import pickle
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

with open("/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/pyEcalVeto/bdt_test_0/bdt_test_0_weights.pkl", "rb") as f:
    clf = pickle.load(f)

n_features = 47 
onnx_model = onnxmltools.convert_xgboost(clf, initial_types=[('input', FloatTensorType([None, n_features]))])
with open("bdt_test_0_weights_f.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model saved as bdt_test_0_weights_f.onnx")