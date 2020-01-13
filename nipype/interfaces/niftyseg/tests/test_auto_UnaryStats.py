# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..stats import UnaryStats


def test_UnaryStats_inputs():
    input_map = dict(
        args=dict(argstr="%s",),
        environ=dict(nohash=True, usedefault=True,),
        in_file=dict(argstr="%s", extensions=None, mandatory=True, position=2,),
        larger_voxel=dict(argstr="-t %f", position=-3,),
        mask_file=dict(argstr="-m %s", extensions=None, position=-2,),
        operation=dict(argstr="-%s", mandatory=True, position=4,),
    )
    inputs = UnaryStats.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_UnaryStats_outputs():
    output_map = dict(output=dict(),)
    outputs = UnaryStats.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
