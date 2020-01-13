# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..surface import ProbeVolumeWithModel


def test_ProbeVolumeWithModel_inputs():
    input_map = dict(
        InputModel=dict(argstr="%s", extensions=None, position=-2,),
        InputVolume=dict(argstr="%s", extensions=None, position=-3,),
        OutputModel=dict(argstr="%s", hash_files=False, position=-1,),
        args=dict(argstr="%s",),
        environ=dict(nohash=True, usedefault=True,),
    )
    inputs = ProbeVolumeWithModel.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_ProbeVolumeWithModel_outputs():
    output_map = dict(OutputModel=dict(extensions=None, position=-1,),)
    outputs = ProbeVolumeWithModel.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
