# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# -*- coding: utf-8 -*-
from __future__ import (print_function, division, unicode_literals,
                        absolute_import)

import os.path as op

from ..base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec,
                    File, isdefined, Undefined, Directory, InputMultiObject,
                    InputMultiPath, OutputMultiPath)
from .base import MRTrix3BaseInputSpec, MRTrix3Base


class BrainMaskInputSpec(MRTrix3BaseInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-2,
        desc='input diffusion weighted images')
    out_file = File(
        'brainmask.mif',
        argstr='%s',
        mandatory=True,
        position=-1,
        usedefault=True,
        desc='output brain mask')


class BrainMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output response file')


class BrainMask(CommandLine):
    """
    Convert a mesh surface to a partial volume estimation image


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> bmsk = mrt.BrainMask()
    >>> bmsk.inputs.in_file = 'dwi.mif'
    >>> bmsk.cmdline                               # doctest: +ELLIPSIS
    'dwi2mask dwi.mif brainmask.mif'
    >>> bmsk.run()                                 # doctest: +SKIP
    """

    _cmd = 'dwi2mask'
    input_spec = BrainMaskInputSpec
    output_spec = BrainMaskOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class Mesh2PVEInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-3,
        desc='input mesh')
    reference = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-2,
        desc='input reference image')
    in_first = File(
        exists=True,
        argstr='-first %s',
        desc='indicates that the mesh file is provided by FSL FIRST')

    out_file = File(
        'mesh2volume.nii.gz',
        argstr='%s',
        mandatory=True,
        position=-1,
        usedefault=True,
        desc='output file containing SH coefficients')


class Mesh2PVEOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output response file')


class Mesh2PVE(CommandLine):
    """
    Convert a mesh surface to a partial volume estimation image


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> m2p = mrt.Mesh2PVE()
    >>> m2p.inputs.in_file = 'surf1.vtk'
    >>> m2p.inputs.reference = 'dwi.mif'
    >>> m2p.inputs.in_first = 'T1.nii.gz'
    >>> m2p.cmdline                               # doctest: +ELLIPSIS
    'mesh2pve -first T1.nii.gz surf1.vtk dwi.mif mesh2volume.nii.gz'
    >>> m2p.run()                                 # doctest: +SKIP
    """

    _cmd = 'mesh2pve'
    input_spec = Mesh2PVEInputSpec
    output_spec = Mesh2PVEOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class Generate5ttInputSpec(MRTrix3BaseInputSpec):
    algorithm = traits.Enum(
        'fsl',
        'gif',
        'freesurfer',
        argstr='%s',
        position=-3,
        mandatory=True,
        desc='tissue segmentation algorithm')
    in_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-2,
        desc='input image')
    out_file = File(
        argstr='%s', mandatory=True, position=-1, desc='output image')


class Generate5ttOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output image')


class Generate5tt(MRTrix3Base):
    """
    Generate a 5TT image suitable for ACT using the selected algorithm


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> gen5tt = mrt.Generate5tt()
    >>> gen5tt.inputs.in_file = 'T1.nii.gz'
    >>> gen5tt.inputs.algorithm = 'fsl'
    >>> gen5tt.inputs.out_file = '5tt.mif'
    >>> gen5tt.cmdline                             # doctest: +ELLIPSIS
    '5ttgen fsl T1.nii.gz 5tt.mif'
    >>> gen5tt.run()                               # doctest: +SKIP
    """

    _cmd = '5ttgen'
    input_spec = Generate5ttInputSpec
    output_spec = Generate5ttOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class Generate5ttMaskInputSpec(MRTrix3BaseInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-2,
        desc='input 5tt segmented anatomical image'
    )
    out_file = File(
        'mask_5tt.mif',
        usedefault=True,
        argstr='%s',
        mandatory=True,
        position=-1,
        desc='output mask image'
    )
    mask_in = File(
        argstr='-mask_in %s',
        position=1,
        desc='Filter input mask image according to voxel in GM/WM boundary'
    )


class Generate5ttMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output mask image')


class Generate5ttMask(MRTrix3Base):
    """
    Generate a mask image suitable appropriate for seeding streamlines on the
    grey matter-white matter interface


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> gen5ttmask = mrt.Generate5ttMask()
    >>> gen5ttmask.inputs.in_file = '5tt.mif'
    >>> gen5ttmask.inputs.out_file = 'mask.mif'
    >>> gen5ttmask.cmdline                             # doctest: +ELLIPSIS
    '5tt2gmwmi 5tt.mif mask.mif'
    >>> gen5ttmask.run()                               # doctest: +SKIP
    """

    _cmd = '5tt2gmwmi'
    input_spec = Generate5ttMaskInputSpec
    output_spec = Generate5ttMaskOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class PopulationTemplateInputSpec(MRTrix3BaseInputSpec):
    in_dir = Directory(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-2,
        desc='input directory containing all images to build the template')
    out_file = File(
        'template.mif',
        argstr='%s',
        mandatory=True,
        position=-1,
        usedefault=True,
        desc='output image'
    )
    # general options
    tfm_type = traits.Enum(
        'rigid',
        'affine',
        'nonlinear',
        'rigid_affine',
        'rigid_nonlinear',
        'affine_nonlinear',
        'rigid_affine_nonlinear',
        argstr='-type %s',
        desc='types of registration stages to perform'
    )
    vox_size = traits.List(
        traits.Int, argstr='-voxel_size %s', sep=',', desc='voxel dimensions')
    init_align = traits.Enum(
        'mass',
        'geometric',
        'none',
        argstr='-initial_alignment %s'
    )
    mask_dir = Directory(
        argstr='-mask_dir %s',
        desc='input masks in single directory'
    )
    warp_dir = Directory(
        argstr='-warp_dir %s',
        desc='output directory containing warps for each input'
    )
    tfm_dir = Directory(
        argstr='-transformed_dir %s',
        desc='output directory with transforms for each input'
    )
    lin_tfm_dir = Directory(
        argstr='-linear_transformations_dir %s',
        desc='output directory with linear transforms to generate template'
    )
    template_mask = File(
        argstr='-template_mask %s',
        desc='output template mask'
    )
    # non-lin options
    nl_scale = InputMultiObject(
        traits.Float,
        argstr='-nl_scale %s',
        desc='specify multi-resolution pyramid to build non-linear template'
    )
    nl_lmax = InputMultiObject(
        traits.Int,
        argstr='-nl_lmax %s',
        desc='specify lmax used for non-linear registration'
    )
    nl_niter = InputMultiObject(
        traits.Int,
        argstr='-nl_niter %s',
        desc='specify number of registration iterations at each level'
    )
    nl_update_smooth = traits.Float(
        argstr='-nl_update_smooth %f',
        desc='regularise gradient update field with Gaussian smoothing'
    )
    nl_disp_smooth = traits.Float(
        argstr='-nl_disp_smooth %f',
        desc='regularise displacement field with Gaussian smoothing'
    )
    nl_grad_step = traits.Float(
        argstr='-nl_grad_step %f',
        desc='gradient step size for non-linear registration'
    )
    # linear options
    lin_estimator = traits.String(
        argstr='-linear_estimator %s',
        desc='choose estimator for intensity difference metric'
    )
    rigid_scale = InputMultiObject(
        traits.Float,
        argstr='-rigid_scale %s',
        desc='specify multi-resolution pyramid to build rigid template'
    )
    rigid_lmax = InputMultiObject(
        traits.Int,
        argstr='-rigid_lmax %s',
        desc='specify lmax used for rigid registration'
    )
    rigid_niter = InputMultiObject(
        traits.Int,
        argstr='-rigid_niter %s',
        desc='specify number of registration iterations at each level'
    )
    affine_scale = InputMultiObject(
        traits.Float,
        argstr='-affine_scale %s',
        desc='specify multi-resolution pyramid to build affine template'
    )
    affine_lmax = InputMultiObject(
        traits.Int,
        argstr='-affine_lmax %s',
        desc='specify lmax used for affine registration'
    )
    affine_niter = InputMultiObject(
        traits.Int,
        argstr='-affine_niter %s',
        desc='specify number of registration iterations at each level'
    )


class PopulationTemplateOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output image')
    warp_dir = Directory(desc='output directory with warps')
    tfm_dir = Directory(desc='output directory with transforms')
    lin_tfm_dir = Directory(desc='output directory with linear transforms')
    template_mask = File(desc='output template mask')


class PopulationTemplate(MRTrix3Base):
    """
    Generates unbiased group-average template from series of images

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> popTemplate = mrt.PopulationTemplate()
    >>> popTemplate.inputs.in_dir = 'in_folder'
    >>> popTemplate.inputs.out_file = 'template.mif'
    >>> popTemplate.cmdline                             # doctest: +ELLIPSIS
    'population_template in_folder template.mif'
    >>> popTemplate.run()                               # doctest: +SKIP
    """

    _cmd = 'population_template'
    input_spec = PopulationTemplateInputSpec
    output_spec = PopulationTemplateOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        # Conditional outputs
        if isdefined(self.inputs.warp_dir) and self.inputs.warp_dir:
            outputs['warp_dir'] = op.abspath(self.inputs.warp_dir)
        if isdefined(self.inputs.tfm_dir) and self.inputs.tfm_dir:
            outputs['tfm_dir'] = op.abspath(self.inputs.tfm_dir)
        if isdefined(self.inputs.lin_tfm_dir) and self.inputs.lin_tmf_dir:
            outputs['lin_tfm_dir'] = op.abspath(self.inputs.lin_tfm_dir)
        if isdefined(self.inputs.template_mask) and self.inputs.template_mask:
            outputs['template_mask'] = op.abspath(self.inputs.template_mask)
        return outputs


class TensorMetricsInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-1,
        desc='input DTI image')

    out_fa = File(argstr='-fa %s', desc='output FA file')
    out_adc = File(argstr='-adc %s', desc='output ADC file')
    out_rd = File(argstr='-rd %s', desc='output RD file')
    out_ad = File(argstr='-ad %s', desc='output AD file')
    out_evec = File(
        argstr='-vector %s', desc='output selected eigenvector(s) file')
    out_eval = File(
        argstr='-value %s', desc='output selected eigenvalue(s) file')
    component = traits.List(
        [1],
        usedefault=True,
        argstr='-num %s',
        sep=',',
        desc=('specify the desired eigenvalue/eigenvector(s). Note that '
              'several eigenvalues can be specified as a number sequence'))
    in_mask = File(
        exists=True,
        argstr='-mask %s',
        desc=('only perform computation within the specified binary'
              ' brain mask image'))
    modulate = traits.Enum(
        'FA',
        'none',
        'eval',
        argstr='-modulate %s',
        desc=('how to modulate the magnitude of the'
              ' eigenvectors'))

    nthreads = traits.Int(
        argstr='-nthreads %d',
        desc=('use this number of threads in multi-threaded applications'))


class TensorMetricsOutputSpec(TraitedSpec):
    out_fa = File(desc='output FA file')
    out_adc = File(desc='output ADC file')
    out_rd = File(desc='output RD file')
    out_ad = File(desc='output AD file')
    out_evec = File(desc='output selected eigenvector(s) file')
    out_eval = File(desc='output selected eigenvalue(s) file')


class TensorMetrics(CommandLine):
    """
    Compute metrics from tensors


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> comp = mrt.TensorMetrics()
    >>> comp.inputs.in_file = 'dti.mif'
    >>> comp.inputs.out_fa = 'fa.mif'
    >>> comp.cmdline                               # doctest: +ELLIPSIS
    'tensor2metric -num 1 -fa fa.mif dti.mif'
    >>> comp.run()                                 # doctest: +SKIP
    """

    _cmd = 'tensor2metric'
    input_spec = TensorMetricsInputSpec
    output_spec = TensorMetricsOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()

        for k in list(outputs.keys()):
            if isdefined(getattr(self.inputs, k)):
                outputs[k] = op.abspath(getattr(self.inputs, k))

        return outputs


class ComputeTDIInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-2,
        desc='input tractography')
    out_file = File(
        'tdi.mif',
        argstr='%s',
        usedefault=True,
        position=-1,
        desc='output TDI file')
    reference = File(
        exists=True,
        argstr='-template %s',
        desc='a reference'
        'image to be used as template')
    vox_size = traits.List(
        traits.Int, argstr='-vox %s', sep=',', desc='voxel dimensions')
    data_type = traits.Enum(
        'float',
        'unsigned int',
        argstr='-datatype %s',
        desc='specify output image data type')
    use_dec = traits.Bool(argstr='-dec', desc='perform mapping in DEC space')
    dixel = File(
        argstr='-dixel %s',
        desc='map streamlines to'
        'dixels within each voxel. Directions are stored as'
        'azimuth elevation pairs.')
    max_tod = traits.Int(
        argstr='-tod %d',
        desc='generate a Track Orientation '
        'Distribution (TOD) in each voxel.')

    contrast = traits.Enum(
        'tdi',
        'length',
        'invlength',
        'scalar_map',
        'scalar_map_conut',
        'fod_amp',
        'curvature',
        argstr='-constrast %s',
        desc='define the desired '
        'form of contrast for the output image')
    in_map = File(
        exists=True,
        argstr='-image %s',
        desc='provide the'
        'scalar image map for generating images with '
        '\'scalar_map\' contrasts, or the SHs image for fod_amp')

    stat_vox = traits.Enum(
        'sum',
        'min',
        'mean',
        'max',
        argstr='-stat_vox %s',
        desc='define the statistic for choosing the final'
        'voxel intesities for a given contrast')
    stat_tck = traits.Enum(
        'mean',
        'sum',
        'min',
        'max',
        'median',
        'mean_nonzero',
        'gaussian',
        'ends_min',
        'ends_mean',
        'ends_max',
        'ends_prod',
        argstr='-stat_tck %s',
        desc='define the statistic for choosing '
        'the contribution to be made by each streamline as a function of'
        ' the samples taken along their lengths.')

    fwhm_tck = traits.Float(
        argstr='-fwhm_tck %f',
        desc='define the statistic for choosing the'
        ' contribution to be made by each streamline as a function of the '
        'samples taken along their lengths')

    map_zero = traits.Bool(
        argstr='-map_zero',
        desc='if a streamline has zero contribution based '
        'on the contrast & statistic, typically it is not mapped; use this '
        'option to still contribute to the map even if this is the case '
        '(these non-contributing voxels can then influence the mean value in '
        'each voxel of the map)')

    upsample = traits.Int(
        argstr='-upsample %d',
        desc='upsample the tracks by'
        ' some ratio using Hermite interpolation before '
        'mappping')

    precise = traits.Bool(
        argstr='-precise',
        desc='use a more precise streamline mapping '
        'strategy, that accurately quantifies the length through each voxel '
        '(these lengths are then taken into account during TWI calculation)')
    ends_only = traits.Bool(
        argstr='-ends_only',
        desc='only map the streamline'
        ' endpoints to the image')

    tck_weights = File(
        exists=True,
        argstr='-tck_weights_in %s',
        desc='specify'
        ' a text scalar file containing the streamline weights')
    nthreads = traits.Int(
        argstr='-nthreads %d',
        desc='number of threads. if zero, the number'
        ' of available cpus will be used',
        nohash=True)


class ComputeTDIOutputSpec(TraitedSpec):
    out_file = File(desc='output TDI file')


class ComputeTDI(MRTrix3Base):
    """
    Use track data as a form of contrast for producing a high-resolution
    image.

    .. admonition:: References

      * For TDI or DEC TDI: Calamante, F.; Tournier, J.-D.; Jackson, G. D. &
        Connelly, A. Track-density imaging (TDI): Super-resolution white
        matter imaging using whole-brain track-density mapping. NeuroImage,
        2010, 53, 1233-1243

      * If using -contrast length and -stat_vox mean: Pannek, K.; Mathias,
        J. L.; Bigler, E. D.; Brown, G.; Taylor, J. D. & Rose, S. E. The
        average pathlength map: A diffusion MRI tractography-derived index
        for studying brain pathology. NeuroImage, 2011, 55, 133-141

      * If using -dixel option with TDI contrast only: Smith, R.E., Tournier,
        J-D., Calamante, F., Connelly, A. A novel paradigm for automated
        segmentation of very large whole-brain probabilistic tractography
        data sets. In proc. ISMRM, 2011, 19, 673

      * If using -dixel option with any other contrast: Pannek, K., Raffelt,
        D., Salvado, O., Rose, S. Incorporating directional information in
        diffusion tractography derived maps: angular track imaging (ATI).
        In Proc. ISMRM, 2012, 20, 1912

      * If using -tod option: Dhollander, T., Emsell, L., Van Hecke, W., Maes,
        F., Sunaert, S., Suetens, P. Track Orientation Density Imaging (TODI)
        and Track Orientation Distribution (TOD) based tractography.
        NeuroImage, 2014, 94, 312-336

      * If using other contrasts / statistics: Calamante, F.; Tournier, J.-D.;
        Smith, R. E. & Connelly, A. A generalised framework for
        super-resolution track-weighted imaging. NeuroImage, 2012, 59,
        2494-2503

      * If using -precise mapping option: Smith, R. E.; Tournier, J.-D.;
        Calamante, F. & Connelly, A. SIFT: Spherical-deconvolution informed
        filtering of tractograms. NeuroImage, 2013, 67, 298-312 (Appendix 3)



    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> tdi = mrt.ComputeTDI()
    >>> tdi.inputs.in_file = 'dti.mif'
    >>> tdi.cmdline                               # doctest: +ELLIPSIS
    'tckmap dti.mif tdi.mif'
    >>> tdi.run()                                 # doctest: +SKIP
    """

    _cmd = 'tckmap'
    input_spec = ComputeTDIInputSpec
    output_spec = ComputeTDIOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class TCK2VTKInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-2,
        desc='input tractography')
    out_file = File(
        'tracks.vtk',
        argstr='%s',
        usedefault=True,
        position=-1,
        desc='output VTK file')
    reference = File(
        exists=True,
        argstr='-image %s',
        desc='if specified, the properties of'
        ' this image will be used to convert track point positions from real '
        '(scanner) coordinates into image coordinates (in mm).')
    voxel = File(
        exists=True,
        argstr='-image %s',
        desc='if specified, the properties of'
        ' this image will be used to convert track point positions from real '
        '(scanner) coordinates into image coordinates.')

    nthreads = traits.Int(
        argstr='-nthreads %d',
        desc='number of threads. if zero, the number'
        ' of available cpus will be used',
        nohash=True)


class TCK2VTKOutputSpec(TraitedSpec):
    out_file = File(desc='output VTK file')


class TCK2VTK(MRTrix3Base):
    """
    Convert a track file to a vtk format, cave: coordinates are in XYZ
    coordinates not reference

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> vtk = mrt.TCK2VTK()
    >>> vtk.inputs.in_file = 'tracks.tck'
    >>> vtk.inputs.reference = 'b0.nii'
    >>> vtk.cmdline                               # doctest: +ELLIPSIS
    'tck2vtk -image b0.nii tracks.tck tracks.vtk'
    >>> vtk.run()                                 # doctest: +SKIP
    """

    _cmd = 'tck2vtk'
    input_spec = TCK2VTKInputSpec
    output_spec = TCK2VTKOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class DWIExtractInputSpec(MRTrix3BaseInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-2,
        desc='input image')
    out_file = File(
        argstr='%s', mandatory=True, position=-1, desc='output image')
    bzero = traits.Bool(argstr='-bzero', desc='extract b=0 volumes')
    nobzero = traits.Bool(argstr='-nobzero', desc='extract non b=0 volumes')
    singleshell = traits.Bool(
        argstr='-singleshell', desc='extract volumes with a specific shell')
    shell = traits.List(
        traits.Float,
        sep=',',
        argstr='-shell %s',
        desc='specify one or more gradient shells')


class DWIExtractOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output image')


class DWIExtract(MRTrix3Base):
    """
    Extract diffusion-weighted volumes, b=0 volumes, or certain shells from a
    DWI dataset

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> dwiextract = mrt.DWIExtract()
    >>> dwiextract.inputs.in_file = 'dwi.mif'
    >>> dwiextract.inputs.bzero = True
    >>> dwiextract.inputs.out_file = 'b0vols.mif'
    >>> dwiextract.inputs.grad_fsl = ('bvecs', 'bvals')
    >>> dwiextract.cmdline                             # doctest: +ELLIPSIS
    'dwiextract -bzero -fslgrad bvecs bvals dwi.mif b0vols.mif'
    >>> dwiextract.run()                               # doctest: +SKIP
    """

    _cmd = 'dwiextract'
    input_spec = DWIExtractInputSpec
    output_spec = DWIExtractOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class MRConvertInputSpec(MRTrix3BaseInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-2,
        desc='input image')
    out_file = File(
        'dwi.mif',
        argstr='%s',
        mandatory=True,
        position=-1,
        usedefault=True,
        desc='output image')
    coord = traits.List(
        traits.Float,
        sep=' ',
        argstr='-coord %s',
        desc='extract data at the specified coordinates')
    vox = traits.List(
        traits.Float,
        sep=',',
        argstr='-vox %s',
        desc='change the voxel dimensions')
    axes = traits.List(
        traits.Int,
        sep=',',
        argstr='-axes %s',
        desc='specify the axes that will be used')
    scaling = traits.List(
        traits.Float,
        sep=',',
        argstr='-scaling %s',
        desc='specify the data scaling parameter')


class MRConvertOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output image')


class MRConvert(MRTrix3Base):
    """
    Perform conversion between different file types and optionally extract a
    subset of the input image

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> mrconvert = mrt.MRConvert()
    >>> mrconvert.inputs.in_file = 'dwi.nii.gz'
    >>> mrconvert.inputs.grad_fsl = ('bvecs', 'bvals')
    >>> mrconvert.cmdline                             # doctest: +ELLIPSIS
    'mrconvert -fslgrad bvecs bvals dwi.nii.gz dwi.mif'
    >>> mrconvert.run()                               # doctest: +SKIP
    """

    _cmd = 'mrconvert'
    input_spec = MRConvertInputSpec
    output_spec = MRConvertOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class MRMathInputSpec(MRTrix3BaseInputSpec):
    in_file = InputMultiObject(
        traits.Either(File(exists=True), traits.List),
        argstr='%s',
        mandatory=True,
        position=-3,
        desc='input image(s)')
    out_file = File(
        argstr='%s', mandatory=True, position=-1, desc='output image')
    operation = traits.Enum(
        'mean',
        'median',
        'sum',
        'product',
        'rms',
        'norm',
        'var',
        'std',
        'min',
        'max',
        'absmax',
        'magmax',
        argstr='%s',
        position=-2,
        mandatory=True,
        desc='operation to computer along a specified axis')
    axis = traits.Int(
        0,
        argstr='-axis %d',
        desc='specfied axis to perform the operation along')


class MRMathOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output image')


class MRMath(MRTrix3Base):
    """
    Compute summary statistic on image intensities
    along a specified axis of a single image

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> mrmath = mrt.MRMath()
    >>> mrmath.inputs.in_file = 'dwi.mif'
    >>> mrmath.inputs.operation = 'mean'
    >>> mrmath.inputs.axis = 3
    >>> mrmath.inputs.out_file = 'dwi_mean.mif'
    >>> mrmath.inputs.grad_fsl = ('bvecs', 'bvals')
    >>> mrmath.cmdline                             # doctest: +ELLIPSIS
    'mrmath -axis 3 -fslgrad bvecs bvals dwi.mif mean dwi_mean.mif'
    >>> mrmath.run()                               # doctest: +SKIP
    """

    _cmd = 'mrmath'
    input_spec = MRMathInputSpec
    output_spec = MRMathOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class MRRegisterInputSpec(MRTrix3BaseInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-2,
        desc='input image 1 ("moving")'
    )
    ref_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-1,
        desc='input image 2 ("template")'
    )
    # general options
    tfm_type = traits.Enum(
        'rigid',
        'affine',
        'nonlinear',
        'rigid_affine',
        'rigid_nonlinear',
        'affine_nonlinear',
        'rigid_affine_nonlinear',
        argstr='-type %s',
        desc='registration type'
    )
    tfm_img = File(
        argstr='-transformed %s',
        desc='moving image after transform to template space'
    )
    tfm_midway = traits.List(
        File(exists=False),
        argstr='-transformed_midway %s',
        desc='moving and template image after transform to midway space'
    )
    mask1 = File(
        argstr='-mask1 %s',
        desc='mask to define region of moving image for optimization'
    )
    mask2 = File(
        argstr='-mask2 %s',
        desc='mask to define region of template image for optimization'
    )
    # linear options
    rigid = File(
        argstr='-rigid %s',
        desc='output file containing 4x4 transformation matrix'
    )
    rigid_1tomidway = File(
        argstr='-rigid_1tomidway %s',
        desc='output file containing 4x4 transformation aligning '
             'moving to template image at common midway'
    )
    rigid_2tomidway = File(
        argstr='-rigid_2tomidway %s',
        desc='output file containing 4x4 transformation aligning '
             'template to moving image at common midway'
    )
    rigid_init_translate = traits.Enum(
        'mass',
        'geometric',
        argstr='-rigid_init_translation %s',
        desc='initialize translation and centre of rotation'
    )
    rigid_init_rotation = traits.Enum(
        'search',
        'mass',
        'none',
        argstr='-rigid_init_rotation %s',
        desc='initialize rotation'
    )
    rigid_init_matrix = File(
        argstr='-rigid_init_matrix %s',
        desc='initialize registration with supplied rigid '
             'transformation; note this overrides '
             'rigid_init_translation and rigid_init_rotation'
    )
    rigid_scale = InputMultiObject(
        traits.Float,
        argstr='-rigid_scale %s',
        desc='define scale factor of each level of multi-resolution '
             'scheme'
    )
    affine = File(
        argstr='-affine %s',
        desc='output file containing affine transformation (4x4)'
    )
    affine_1tomidway = File(
        argstr='-affine_1tomidway %s',
        desc='output file containing affine transformation aligning '
             'moving to template image at common midway'
    )
    affine_2tomidway = File(
        argstr='-affine_2tomidway %s',
        desc='output file containing affine transformation aligning '
             'template to moving image at common midway'
    )
    affine_init_translate = traits.Enum(
        'mass',
        'geometric',
        'none',
        argstr='-affine_init_translation %s',
        desc='initialize translatation and centre of rotation'
    )
    affine_init_rotation = traits.Enum(
        'search',
        'moments',
        'none',
        argstr='-affine_init_rotation %s',
        desc='initialize rotation'
    )
    affine_init_matrix = File(
        argstr='-rigid_init_matrix %s',
        desc='initialize registration with supplied affine '
             'transformation; note this overrides '
             'affine_init_translation and affine_init_rotation'
    )
    affine_scale = InputMultiObject(
        traits.Float,
        argstr='-affine_scale %s',
        desc='define scale factor of each level of multi-resolution '
             'scheme'
    )
    affine_niter = InputMultiObject(
        traits.Int,
        argstr='-affine_niter %s',
        desc='maximum number of gradient descent iterations per stage'
    )
    affine_metric = traits.Enum(
        'diff',
        argstr='-affine_metric %s',
        desc='valid choices are: diff (intensity differences)'
    )
    affine_metric_estimator = traits.Enum(
        'l1',
        'l2',
        'lp',
        argstr='-affine_metric.diff.estimator %s',
        desc='valid choices are l1 (least absolute: |x|), '
             'l2 (ordinary least squares), lp (least powers: |x|^1.2)'
    )
    affine_lmax = InputMultiObject(
        traits.Int,
        argstr='-affine_lmax %s',
        desc='explicitly set lmax to be used per scale factor'
    )
    affine_log = File(
        argstr='-affine_log %s',
        desc='write gradient descent parameter evolution to log file'
    )
    # advanced linear transform initialization options
    init_translation_unmasked1 = traits.Bool(
        argstr='-init_translation.unmasked1',
        desc='disregard mask1 for translation'
    )
    init_translation_unmasked2 = traits.Bool(
        argstr='-init_translationl.unmasked2',
        desc='disregard mask2 for translation'
    )
    init_rotation_unmasked1 = traits.Bool(
        argstr='-init_rotation.unmasked1',
        desc='disregard mask1 for rotation'
    )
    init_rotation_unmasked2 = traits.Bool(
        argstr='-init_rotation.unmasked2',
        desc='disregard mask2 for rotation'
    )
    init_rotation_angles = InputMultiObject(
        traits.Int,
        argstr='-init_rotation.search.angles %s',
        desc='rotation angles for local search in degrees between 0 '
             'and 180'
    )
    init_rotation_scale = traits.Float(
        argstr='-init_rotation.search.scale %f',
        desc='relative size of images used for rotation search'
    )
    init_rotation_direction = traits.Int(
        argstr='-init_rotation.search.directions %d',
        desc='number of rotation axis for local search'
    )
    init_rotation_global = traits.Bool(
        argstr='-init_rotation.search.run_global',
        desc='perform a global search'
    )
    init_rotation_iteration = traits.Int(
        argstr='-init_rotation.search.global.iterations %d',
        desc='number of rotations to investigate'
    )
    # advanced linear registration stage options
    linstage_iterations = InputMultiObject(
        traits.Int,
        argstr='-linstage.iterations %s',
        desc='number of iterations for each registration stage'
    )
    linstage_first_opt = traits.Enum(
        'bbgd',
        'gd',
        argstr='-linstage.optimiser.first %s',
        desc='cost function to use at first iteration of all stages'
    )
    linstage_last_opt = traits.Enum(
        'bbgd',
        'gd',
        argstr='-linstage.optimiser.last %s',
        desc='cost function to use at last iteration of all stages'
    )
    linstage_default_opt = traits.Enum(
        'bbgd',
        'gd',
        argstr='-linstage.optimiser.default %s',
        desc='cost function to use at any stage other than first '
             'or last'
    )
    linstage_diagnostic = File(
        argstr='-linstage.diagnostics.prefix %s diagnostic',
        desc='generate diagnostic images after each registration stage'
    )
    # non-linear registration options
    nl_warp = InputMultiPath(
        File,
        argstr='-nl_warp %s ',
        desc='non-linear warp output as two deformation fields, where '
             'warp1 transforms moving to template and warp2 transforms '
             'template to moving image'
    )
    nl_warp_full = traits.File(
        argstr='-nl_warp_full %s',
        desc='output all warps during registration'
    )
    nl_init = File(
        argstr='-nl_init %s',
        desc='initialize non-linear registration with supplied warped '
             'image'
    )
    nl_scale = InputMultiObject(
        traits.Float,
        argstr='-nl_scale %s',
        desc='defining scale factor for each level of multi-resolution '
             'scheme'
    )
    nl_niter = InputMultiObject(
        traits.Int,
        argstr='-nl_niter %s',
        desc='maximum number of iterations'
    )
    nl_update_smooth = traits.Float(
        argstr='-nl_update_smooth %f',
        desc='regularise gradient update field with Gausisan smoothing'
    )
    nl_dis_smooth = traits.Float(
        argstr='-nl_dis_smooth %f',
        desc='regularise displacement field with Gaussian smoothing'
    )
    nl_grad_step = traits.Float(
        argstr='-nl_grad_step %f',
        desc='gradient step size for non-linear registration'
    )
    nl_lmax = InputMultiObject(
        traits.Int,
        argstr='-nl_lmax %s',
        desc='lax to be used per scale factor in non-linear FOD '
             'registration'
    )
    # fod registration options
    directions = File(
        argstr='-directions %s',
        desc='directions used for FOD reorientation using apodised PSF'
    )
    noreorientation = traits.Bool(
        argstr='-noreorientation',
        desc='turn off FOD reorientation'
    )


class MRRegisterOutputSpec(TraitedSpec):
    tfm_img = File(desc='moving image after transform to template space')
    tfm_midway = File(desc='moving and template image after transform in '
                           'midway space')
    rigid = File(desc='file containing rigid 4x4 transformation matrix')
    rigid_1tomidway = File(desc='file containing rigid 4x4 '
                                'transform matrix aligning moving at common '
                                'midway')
    rigid_2tomidway = File(desc='file containing rigid 4x4 '
                                'transform matrix aligning template at common '
                                'midway')
    affine = File(desc='file containing affine 4x4 transformation matrix')
    affine_1tomidway = File(desc='file containing affine 4x4 transform matrix '
                                 'aligning moving image at common midway')
    affine_2tomidway = File(desc='file containing affine 4x4 transform matrix '
                                 'aligning template image at common midway')
    affine_log = File(desc='log with gradient descent parameter evolution')
    linstage_diagnostic = File(desc='diagnostic image after each registration '
                                    'stage')
    nl_warp = OutputMultiPath(File, desc='non-linear warp output as two '
                                         'deformation fields')
    nl_warp_full = File(desc='output all warps used during registration')


class MRRegister(MRTrix3Base):
    """
    Register two images together using a symmetric rigid, affine, or non-linear
    transformation model

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> mrregister = mrt.MRRegister()
    >>> mrregister.inputs.in_file = 'moving.mif'
    >>> mrregister.inputs.ref_file = 'template.mif'
    >>> mrregister.inputs.mask1 = 'mask1.mif'
    >>> mrregister.inputs.nl_warp = ['mv-tmp_warp', 'tmp-mv_warp']
    >>> mrregister.cmdline
    'mrregister -mask1 mask1.mif -nl_warp mv-tmp_warp tmp-mv_warp moving.mif \
    template.mif'
    >>> mrregister.run()
    """

    _cmd = 'mrregister'
    input_spec = MRRegisterInputSpec
    output_spec = MRRegisterOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if self.inputs.tfm_img != Undefined:
            outputs['tfm_img'] = op.abspath(self.inputs.tfm_img)
        if self.inputs.tfm_midway != Undefined:
            outputs['tfm_midway'] = op.abspath(self.inputs.tfm_midway)
        if self.inputs.rigid != Undefined:
            outputs['rigid'] = op.abspath(self.inputs.rigid)
        if self.inputs.rigid_1tomidway != Undefined:
            outputs['rigid_1tomidway'] = op.abspath(
                                            self.inputs.rigid_1tomidway)
        if self.inputs.rigid_2tomidway != Undefined:
            outputs['rigid_2tomidway'] = op.abspath(
                                            self.inputs.rigid_2tomidway)
        if self.inputs.affine != Undefined:
            outputs['affine'] = op.abspath(self.inputs.affine)
        if self.inputs.affine_1tomidway != Undefined:
            outputs['affine_1tomidway'] = op.abspath(
                                            self.inputs.affine_1tomidway)
        if self.inputs.affine_2tomidway != Undefined:
            outputs['affine_2tomidway'] = op.abspath(
                                            self.inputs.affine_2tomidway)
        if self.inputs.affine_log != Undefined:
            outputs['affine_log'] = op.abspath(self.inputs.affine_log)
        if self.inputs.linstage_diagnostic != Undefined:
            outputs['linstage_diagnostic'] = op.abspath(
                self.inputs.linstage_diagnostic)
        if self.inputs.nl_warp != Undefined:
            out_files = []
            for warp in self.inputs.nl_warp:
                out_files.append(op.abspath(warp))
            outputs['nl_warp'] = out_files
        if self.inputs.nl_warp_full != Undefined:
            outputs['nl_warp_full'] = self.inputs.nl_warp_full

        return outputs


class MRTransformInputSpec(MRTrix3BaseInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-2,
        desc='Input images to be transformed')
    out_file = File(
        genfile=True,
        argstr='%s',
        position=-1,
        desc='Output image')
    # Affine transformation options
    linear_transform = File(
        argstr='-linear %s',
        position=1,
        desc="Linear transform to apply mapping points in template image to "
             "moving image")
    flip = InputMultiObject(
        traits.Int,
        argstr='-flip %s',
        desc='flip specified axes, provided as list (0:x, 1:y, 2:z)')
    invert = traits.Bool(
        argstr='-inverse',
        position=1,
        desc="Invert the specified transform before using it")
    half = traits.Bool(
        argstr='-half',
        position=1,
        desc="Apply matrix square root of transformation")
    replace_transform = File(
        argstr='-replace %s',
        position=1,
        desc="Replace the current transform by that specified, rather than "
             "applying it to the current transform")
    identity = traits.Bool(
        argstr='-identity',
        position=1,
        desc="Set header transform of image to identity matrix")
    # Regridding
    template_image = File(
        exists=True,
        argstr='-template %s',
        position=1,
        desc='Reslice the input image to match specified template image grid')
    midway_space = traits.Bool(
        argstr='-midway_space',
        position=1,
        desc='Reslice input image to midway space. Requires template or warp '
             'option.')
    interpolation = traits.Enum(
        'nearest',
        'linear',
        'cubic',
        'sinc',
        argstr='-interp %s',
        position=1,
        desc='set interpolation method to use when reslicing')
    oversample = InputMultiObject(
        traits.Int,
        argstr='-oversample %s',
        position=1,
        desc='Set amount of over-sampling in target space to perform when '
             'regridding.')
    # Non-linear transformation
    warp = File(
        argstr='-warp %s',
        position=1,
        desc='apply non-linear 4d deformation field to warp input image')
    warp_full = File(
        argstr='-warp_full %s',
        position=1,
        desc='warp input image using a 5d warp file output from mrregister')
    # FOD handling options
    modulate = traits.Bool(
        argstr='-modulate',
        position=1,
        desc='modulate FODs during reorientation to preserve apparent fibre '
             'density across fibre bundle widths')
    directions = File(
        argstr='-directions %s',
        position=1,
        desc='directions defining the number and orientation of apodised PSF')
    noreorientation = traits.Bool(
        argstr='-noreorientation',
        position=1,
        desc='turn off FOD reorientation')


class MRTransformOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output image')


class MRTransform(MRTrix3Base):
    """
    Apply spatial transformations to an image

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> mrtransform = mrt.MRTransform()
    >>> mrtransform.inputs.in_file = 'input.mif'
    >>> mrtransform.inputs.out_file = 'output.mif'
    >>> mrtransform.inputs.warp = 'warp.mif'
    >>> mrtransform.cmdline
    'mrtransform -warp warp.mif input.mif output.mif'
    """

    _cmd = 'mrtransform'
    input_spec = MRTransformInputSpec
    output_spec = MRTransformOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)

        return outputs
