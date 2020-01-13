# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# -*- coding: utf-8 -*-
<<<<<<< HEAD

from .utils import (Mesh2PVE, Generate5tt, Generate5ttMask, BrainMask,
                    TensorMetrics, ComputeTDI, TCK2VTK, MRMath, MRConvert,
                    DWIExtract, PopulationTemplate, MRRegister,
                    MRTransform)
from .preprocess import (ResponseSD, ACTPrepareFSL, ReplaceFSwithFIRST,
                         DWIDenoise, AverageResponse, MTNormalise,
                         DWINormalise)
from .tracking import Tractography, SIFT, TCKConvert, TCKEdit
=======
"""MRTrix3 provides software tools to perform various types of diffusion MRI analyses."""
from .utils import (
    Mesh2PVE,
    Generate5tt,
    BrainMask,
    TensorMetrics,
    ComputeTDI,
    TCK2VTK,
    MRMath,
    MRConvert,
    MRResize,
    DWIExtract,
)
from .preprocess import (
    ResponseSD,
    ACTPrepareFSL,
    ReplaceFSwithFIRST,
    DWIDenoise,
    MRDeGibbs,
    DWIBiasCorrect,
)
from .tracking import Tractography
>>>>>>> 5d2fe1df7b7ac10be41aefb7a0f3de3d87da829c
from .reconst import FitTensor, EstimateFOD
from .connectivity import LabelConfig, LabelConvert, BuildConnectome
