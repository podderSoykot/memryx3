from deepface.basemodels.ArcFace import FacialRecognition
import memryx as mx
import numpy as np

class ArcFaceMXClient(FacialRecognition):
    """
    ArcFace running on the MXA (MemryX Accelerator)
    """
    def __init__(self):
        self.accl = mx.SyncAccl('ArcFace.dfp')
        self.model_name = "ArcFace"
        self.input_shape = (112, 112)
        self.output_shape = 512

    def forward(self, img: np.ndarray) -> list:
        # Reshape inputs and perform inference
        ifmap = np.squeeze(img)[:,:,None,:]
        outputs = self.accl.run(ifmap)
        return np.squeeze(outputs).tolist() 