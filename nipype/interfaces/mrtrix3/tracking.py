# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# -*- coding: utf-8 -*-
from __future__ import (print_function, division, unicode_literals,
                        absolute_import)

import os.path as op

from ..base import (traits, TraitedSpec, File, InputMultiObject, Directory,
                    Undefined, CommandLineInputSpec, CommandLine, isdefined,
                    InputMultiPath)
from .base import MRTrix3BaseInputSpec, MRTrix3Base


class TractographyInputSpec(MRTrix3BaseInputSpec):
    sph_trait = traits.Tuple(
        traits.Float,
        traits.Float,
        traits.Float,
        traits.Float,
        argstr='%f,%f,%f,%f')

    in_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=-2,
        desc='input file to be processed')

    out_file = File(
        'tracked.tck',
        argstr='%s',
        mandatory=True,
        position=-1,
        usedefault=True,
        desc='output file containing tracks')

    algorithm = traits.Enum(
        'iFOD2',
        'FACT',
        'iFOD1',
        'Nulldist',
        'SD_Stream',
        'Tensor_Det',
        'Tensor_Prob',
        usedefault=True,
        argstr='-algorithm %s',
        desc='tractography algorithm to be used')

    # ROIs processing options
    roi_incl = traits.Either(
        File(exists=True),
        sph_trait,
        argstr='-include %s',
        desc=('specify an inclusion region of interest, streamlines must'
              ' traverse ALL inclusion regions to be accepted'))
    roi_excl = traits.Either(
        File(exists=True),
        sph_trait,
        argstr='-exclude %s',
        desc=('specify an exclusion region of interest, streamlines that'
              ' enter ANY exclude region will be discarded'))
    roi_mask = traits.Either(
        File(exists=True),
        sph_trait,
        argstr='-mask %s',
        desc=('specify a masking region of interest. If defined,'
              'streamlines exiting the mask will be truncated'))

    # Streamlines tractography options
    step_size = traits.Float(
        argstr='-step %f',
        desc=('set the step size of the algorithm in mm (default is 0.1'
              ' x voxelsize; for iFOD2: 0.5 x voxelsize)'))
    angle = traits.Float(
        argstr='-angle %f',
        desc=('set the maximum angle between successive steps (default '
              'is 90deg x stepsize / voxelsize)'))
    n_tracks = traits.Int(
        argstr='-select %d',
        desc=('set the desired number of tracks. The program will continue'
              ' to generate tracks until this number of tracks have been '
              'selected and written to the output file'))
    max_tracks = traits.Int(
        argstr='-maxnum %d',
        desc=('set the maximum number of tracks to generate. The program '
              'will not generate more tracks than this number, even if '
              'the desired number of tracks hasn\'t yet been reached '
              '(default is 100 x number)'))
    max_length = traits.Float(
        argstr='-maxlength %f',
        desc=('set the maximum length of any track in mm (default is '
              '100 x voxelsize)'))
    min_length = traits.Float(
        argstr='-minlength %f',
        desc=('set the minimum length of any track in mm (default is '
              '5 x voxelsize)'))
    cutoff = traits.Float(
        argstr='-cutoff %f',
        desc=('set the FA or FOD amplitude cutoff for terminating '
              'tracks (default is 0.1)'))
    cutoff_init = traits.Float(
        argstr='-initcutoff %f',
        desc=('set the minimum FA or FOD amplitude for initiating '
              'tracks (default is the same as the normal cutoff)'))
    n_trials = traits.Int(
        argstr='-trials %d',
        desc=('set the maximum number of sampling trials at each point'
              ' (only used for probabilistic tracking)'))
    unidirectional = traits.Bool(
        argstr='-unidirectional',
        desc=('track from the seed point in one direction only '
              '(default is to track in both directions)'))
    init_dir = traits.Tuple(
        traits.Float,
        traits.Float,
        traits.Float,
        argstr='-initdirection %f,%f,%f',
        desc=('specify an initial direction for the tracking (this '
              'should be supplied as a vector of 3 comma-separated values'))
    noprecompt = traits.Bool(
        argstr='-noprecomputed',
        desc=('do NOT pre-compute legendre polynomial values. Warning: this '
              'will slow down the algorithm by a factor of approximately 4'))
    power = traits.Int(
        argstr='-power %d',
        desc=('raise the FOD to the power specified (default is 1/nsamples)'))
    n_samples = traits.Int(
        4, usedefault=True,
        argstr='-samples %d',
        desc=('set the number of FOD samples to take per step for the 2nd '
              'order (iFOD2) method'))
    use_rk4 = traits.Bool(
        argstr='-rk4',
        desc=('use 4th-order Runge-Kutta integration (slower, but eliminates'
              ' curvature overshoot in 1st-order deterministic methods)'))
    stop = traits.Bool(
        argstr='-stop',
        desc=('stop propagating a streamline once it has traversed all '
              'include regions'))
    downsample = traits.Float(
        argstr='-downsample %f',
        desc='downsample the generated streamlines to reduce output file size')

    # Anatomically-Constrained Tractography options
    act_file = File(
        exists=True,
        argstr='-act %s',
        desc=('use the Anatomically-Constrained Tractography framework during'
              ' tracking; provided image must be in the 5TT '
              '(five - tissue - type) format'))
    backtrack = traits.Bool(
        argstr='-backtrack', desc='allow tracks to be truncated')

    crop_at_gmwmi = traits.Bool(
        argstr='-crop_at_gmwmi',
        desc=('crop streamline endpoints more '
              'precisely as they cross the GM-WM interface'))

    # Tractography seeding options
    seed_sphere = traits.Tuple(
        traits.Float,
        traits.Float,
        traits.Float,
        traits.Float,
        argstr='-seed_sphere %f,%f,%f,%f',
        desc='spherical seed')
    seed_image = File(
        exists=True,
        argstr='-seed_image %s',
        desc='seed streamlines entirely at random within mask')
    seed_rnd_voxel = traits.Tuple(
        File(exists=True),
        traits.Int(),
        argstr='-seed_random_per_voxel %s %d',
        xor=['seed_image', 'seed_grid_voxel'],
        desc=('seed a fixed number of streamlines per voxel in a mask '
              'image; random placement of seeds in each voxel'))
    seed_grid_voxel = traits.Tuple(
        File(exists=True),
        traits.Int(),
        argstr='-seed_grid_per_voxel %s %d',
        xor=['seed_image', 'seed_rnd_voxel'],
        desc=('seed a fixed number of streamlines per voxel in a mask '
              'image; place seeds on a 3D mesh grid (grid_size argument '
              'is per axis; so a grid_size of 3 results in 27 seeds per'
              ' voxel)'))
    seed_rejection = File(
        exists=True,
        argstr='-seed_rejection %s',
        desc=('seed from an image using rejection sampling (higher '
              'values = more probable to seed from'))
    seed_gmwmi = File(
        exists=True,
        argstr='-seed_gmwmi %s',
        requires=['act_file'],
        desc=('seed from the grey matter - white matter interface (only '
              'valid if using ACT framework)'))
    seed_dynamic = File(
        exists=True,
        argstr='-seed_dynamic %s',
        desc=('determine seed points dynamically using the SIFT model '
              '(must not provide any other seeding mechanism). Note that'
              ' while this seeding mechanism improves the distribution of'
              ' reconstructed streamlines density, it should NOT be used '
              'as a substitute for the SIFT method itself.'))
    max_seed_attempts = traits.Int(
        argstr='-max_seed_attempts %d',
        desc=('set the maximum number of times that the tracking '
              'algorithm should attempt to find an appropriate tracking'
              ' direction from a given seed point'))
    out_seeds = File(
        'out_seeds.nii.gz', usedefault=True,
        argstr='-output_seeds %s',
        desc=('output the seed location of all successful streamlines to'
              ' a file'))


class TractographyOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output filtered tracks')
    out_seeds = File(
        desc=('output the seed location of all successful'
              ' streamlines to a file'))


class Tractography(MRTrix3Base):
    """
    Performs streamlines tractography after selecting the appropriate
    algorithm.

    .. [FACT] Mori, S.; Crain, B. J.; Chacko, V. P. & van Zijl,
      P. C. M. Three-dimensional tracking of axonal projections in the
      brain by magnetic resonance imaging. Annals of Neurology, 1999,
      45, 265-269

    .. [iFOD1] Tournier, J.-D.; Calamante, F. & Connelly, A. MRtrix:
      Diffusion tractography in crossing fiber regions. Int. J. Imaging
      Syst. Technol., 2012, 22, 53-66

    .. [iFOD2] Tournier, J.-D.; Calamante, F. & Connelly, A. Improved
      probabilistic streamlines tractography by 2nd order integration
      over fibre orientation distributions. Proceedings of the
      International Society for Magnetic Resonance in Medicine, 2010, 1670

    .. [Nulldist] Morris, D. M.; Embleton, K. V. & Parker, G. J.
      Probabilistic fibre tracking: Differentiation of connections from
      chance events. NeuroImage, 2008, 42, 1329-1339

    .. [Tensor_Det] Basser, P. J.; Pajevic, S.; Pierpaoli, C.; Duda, J.
      and Aldroubi, A. In vivo fiber tractography using DT-MRI data.
      Magnetic Resonance in Medicine, 2000, 44, 625-632

    .. [Tensor_Prob] Jones, D. Tractography Gone Wild: Probabilistic Fibre
      Tracking Using the Wild Bootstrap With Diffusion Tensor MRI. IEEE
      Transactions on Medical Imaging, 2008, 27, 1268-1274


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> tk = mrt.Tractography()
    >>> tk.inputs.in_file = 'fods.mif'
    >>> tk.inputs.roi_mask = 'mask.nii.gz'
    >>> tk.inputs.seed_sphere = (80, 100, 70, 10)
    >>> tk.cmdline                               # doctest: +ELLIPSIS
    'tckgen -algorithm iFOD2 -samples 4 -output_seeds out_seeds.nii.gz \
-mask mask.nii.gz -seed_sphere \
80.000000,100.000000,70.000000,10.000000 fods.mif tracked.tck'
    >>> tk.run()                                 # doctest: +SKIP
    """

    _cmd = 'tckgen'
    input_spec = TractographyInputSpec
    output_spec = TractographyOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if 'roi_' in name and isinstance(value, tuple):
            value = ['%f' % v for v in value]
            return trait_spec.argstr % ','.join(value)

        return super(Tractography, self)._format_arg(name, trait_spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class SIFTInputSpec(MRTrix3BaseInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        position=-3,
        mandatory=True,
        desc='input track file'
    )

    in_fod = File(
        exists=True,
        argstr='%s',
        position=-2,
        mandatory=True,
        desc='input image containing spherical harmonics of the fibre '
             'orientation distrubtions'
    )

    out_file = File(
        'tracks_filtered.tck',
        argstr='%s',
        position=-1,
        usedefault=True,
        desc='output filtered tracks file'
    )

    # Options
    nofilter = traits.Bool(
        argstr='-nofilter',
        position=1,
        desc='do NOT perform track filtering - just construct model to '
             'provide output debugging images'
    )
    output_at_counts = InputMultiObject(
        traits.Int,
        argstr='-output_at_counds %d',
        position=1,
        desc='output filtered track files at specific numbers of remaining '
             'streamlines; provide as comma-seperated list of integers'
    )
    # Options for processing mask
    proc_mask = File(
        argstr='-proc_mask %s',
        position=1,
        desc='provide image containign processing mask weights for the model '
             'image spatial dimensions must match the fixel image'
    )
    act_img = File(
        argstr='-act %s',
        position=1,
        desc='use an ACT five-tissue-type segmented anatomical image to '
             'derive the processing mask'
    )
    # Options for SIFT model
    fd_scale_gm = traits.Bool(
        argstr='-fd_scale_gm',
        position=1,
        desc='provide this option (in conjunction with -act) to heuristically '
             'downsize the fibre density estimates based on the presence of '
             'GM in the voxel. this can assist in reducing tissue interface '
             'effects when using a single-tissue deconvolution algorithm'
    )
    no_dilate = traits.Bool(
        argstr='-no_dilate_lut',
        position=1,
        desc='do NOT dilate FOD lobe lookup tables; only map streamlines to '
             'FOD lobes if the precise tangent lies within the angular spread '
             'of that lobe'
    )
    null_lobes = traits.Bool(
        argstr='-make_null_lobes',
        position=1,
        desc='add an additional FOD lobe to each voxel, with zero integral, '
             'that covers all directions with zero / negative FOD amplitudes'
    )
    remove_untracked = traits.Bool(
        argstr='-remove_untracked',
        position=1,
        desc='remove FOD lobes that do not have any streamline density '
             'attributed to them; this improves filtering slightly, at the '
             'expense of longer computation time (and you can no longer do '
             'quantitative comparisons between reconstructions if this is '
             'enabled)'
    )
    fd_thresh = traits.Float(
        argstr='-fd_thresh %f',
        position=1,
        desc='fibre density threshold; excluse an FOD lobe from filtering '
             'processing if its integral is less than this amount ( '
             'streamlines will still be mapped to it, but it will not '
             'contribute to the cost function or the filtering)'
    )
    # Options for additional output files
    csv_file = File(
        argstr='-csv %s',
        position=1,
        desc='output statistics of execution per iteration to a .csv file'
    )
    out_mu = File(
        argstr='-out_mu %s',
        position=1,
        desc='output the final value of SIFT proportionality coefficient mu '
             'to a text file'
    )
    out_debug = traits.Bool(
        argstr='-output_debug',
        position=1,
        desc='provide various output images for assessing & debugging '
             'performance etc.'
    )
    out_selection = Directory(
        argstr='-out_selection %s',
        position=1,
        desc='output a text file containing the binary selection of '
             'streamlines'
    )
    # Options on termination
    term_number = traits.Int(
        argstr='-term_number %d',
        position=1,
        desc='number of streamlines - continue filtering until this nubmer of '
             'streamlines remain'
    )
    term_ratio = traits.Float(
        argstr='-term_ratio %f',
        position=1,
        desc='termination ratio - defined as the ration between reduction in '
             'cost function, and reduction in density of streamlines. smaller '
             'values result in more streamlines being filtered out.'
    )
    term_mu = traits.Float(
        argstr='-term_mu %f',
        position=1,
        desc='terminate filtering once the SIFT proportionality coefficient '
             'reaches a given value'
    )


class SIFTOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output filtered tracks file')
    csv_file = File(desc='output statistics of execution per iteration')
    out_mu = File(desc='output final value of SIFT proportionality '
                       'coefficient')
    out_selection = Directory(desc='output a text file containing binary '
                                   'selection of streamlines')


class SIFT(MRTrix3Base):
    """
    Filter a whole-brain fibre-trackign data set such that the streamline
    densities match the FOD lobe integrals

    .. Smith, R. E.; Tournier, J.-D.; Calamante, F. & Connelly, A.
       SIFT: Spherical-deconvolution informed filtering of tractograms.
       NeuroImage, 2013, 67, 298-312

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> sift = mrt.SIFT()
    >>> sift.inputs.in_file = 'tracks.tck'
    >>> sift.inputs.in_fod = 'fods.mif'
    >>> sift.inputs.out_file = 'tracks_filtered.tck'
    >>> tk.cmdline                               # doctest: +ELLIPSIS
    'tcksift tracks.tck fods.mif tracks_filtered.tck'
    >>> tk.run()                                 # doctest: +SKIP
    """

    _cmd = "tcksift"
    input_spec = SIFTInputSpec
    output_spec = SIFTOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)

        # Conditional outputs
        if self.inputs.csv_file != Undefined:
            outputs['csv_file'] = op.abspath(self.inputs.csv_file)
        if self.inputs.out_mu != Undefined:
            outputs['out_mu'] = op.abspath(self.inputs.out_mu)
        if self.inputs.out_selection != Undefined:
            outputs['out_selection'] = op.abspath(self.inputs.out_selection)
        return outputs


class TCKConvertInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        position=-2,
        mandatory=True,
        desc='input track file'
    )
    out_file = File(
        argstr='%s',
        position=-1,
        mandatory=True,
        desc='output track file'
    )
    # Options
    scanner2voxel = File(
        argstr='-scanner2voxel %s',
        position=1,
        desc='properties of this image will be uesd to convert track point '
             'positions from real (scanner) coordinates into voxel coordinates'
    )
    scanner2image = File(
        argstr='-scanner2image %s',
        position=1,
        desc='properties of this image will be used to convert track point '
             'positions from real (scanner) coordinates into image '
             'coordinates (in mm)'
    )
    voxel2scanner = File(
        argstr='-voxel2scanner %s',
        position=1,
        desc='properties of this image will be used to convert track point '
             'positions from voxel coordinates into real (scanner) coordinates'
    )
    image2scanner = File(
        argstr='-image2scanner %s',
        position=1,
        desc='properties of this image will be used to convert track point '
             'positions from image coordinates (in mm) into real (scanner) '
             'coordinates'
    )
    # Ply writer options
    sides = traits.Int(
        argstr='-sides %d',
        position=1,
        desc='number of sides for streamlines',
    )
    increment = traits.Int(
        argstr='-increment %d',
        position=1,
        desc='generate streamline points at every (increment) points'
    )
    # RIB writer options
    dec = traits.Bool(
        argstr='-dec',
        position=1,
        desc='add DEC as a primvar'
    )
    # Writer options for both RIB and PLY
    radius = traits.Float(
        argstr='-radius %f',
        position=1,
        desc='radius of the streamlines'
    )

    nthreads = traits.Int(
        argstr='-nthreads %d',
        desc='number of threads. if zero, the number'
        ' of available cpus will be used',
    )


class TCKConvertOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output track file')


class TCKConvert(MRTrix3Base):
    """
    Convert between different track file formats

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> tckconvert = mrt.tckconvert()
    >>> tckconvert.inputs.in_file = 'tracks.tck'
    >>> tckconvert.inputs.out_file = 'tracks.vtk'
    >>> tckconert..cmdline                               # doctest: +ELLIPSIS
    'tckconvert tracks.tck tracks.vtk
    >>> tckconvert.run()                                 # doctest: +SKIP
    """
    _cmd = 'tckconvert'
    input_spec = TCKConvertInputSpec
    output_spec = TCKConvertOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class TCKEditInputSpec(CommandLineInputSpec):
    in_file = InputMultiPath(
        File(exists=True),
        argstr='%s',
        mandatory=True,
        position=-2,
        desc='input track file(s)'
        )
    out_file = File(
        'tracks_selected.tck',
        argstr='%s',
        mandatory=True,
        position=-1,
        usedefault=True,
        desc='output track file'
    )
    # ROI options
    include = traits.Either(
        File(exists=True),
        InputMultiObject(traits.Float),
        argstr='-include %s',
        position=1,
        desc='inclusion ROI as either binary mask image, or as a sphere '
             'using 4 comma-seperated values (x,y,z, radius). Streamlines '
             'must traverse all inclusion regions to be accepted'
    )
    exclude = traits.Either(
        File(exists=True),
        InputMultiObject(traits.Float),
        argstr='-include %s',
        position=1,
        desc='inclusion ROI as either binary mask image, or as a sphere '
             'using 4 comma-seperated values (x,y,z,radius). Streamlines '
             'that enter ANY exclude region will be discarded'
    )
    mask = traits.Either(
        File(exists=True),
        InputMultiObject(traits.Float),
        argstr='-mask %s',
        position=1,
        desc='specify a masking ROI, as either a binary mask image, or as a '
             'sphere using 4 comma-seperated values (x,y,z,radius). '
             'If defined, streamlines existing the mask will be truncated.'
    )
    # Streamline threshold options
    maxlength = traits.Float(
        argstr='-maxlength %f',
        position=1,
        desc='set the maximum length of any streamline in mm'
    )
    minlength = traits.Float(
        argstr='-minlength %f',
        position=1,
        desc='set the minimum length of any streamline in mm'
    )
    # Streamline truncation options
    number = traits.Int(
        argstr='-number %d',
        position=1,
        desc='set the desired number of selected streamlines to be '
             'propogated to the output file'
    )
    skip = traits.Int(
        argstr='-skip %d',
        position=1,
        desc='omit this number of selected straemlines before commencing '
             'writing to the output file'
    )
    # Streamline weighting
    maxvalue = traits.Float(
        argstr='-maxweight %f',
        position=1,
        desc='set maximum weight of any streamline'
    )
    minvalue = traits.Float(
        argstr='-minweight %f',
        position=1,
        desc='set minimum weight of any streamline'
    )
    tckweights_in = File(
        exists=True,
        argstr='-tck_weights_in %s',
        position=1,
        desc='specify a text scalar file containing the streamline weights'
    )
    tckweights_out = Directory(
        argstr='-tck_weights_out %s',
        position=1,
        desc='specify the path for an output text scalar file containing '
             'streamline weights'
    )
    # tckedit specific options
    inverse = traits.Bool(
        argstr='-inverse',
        position=1,
        desc='output inverse selection of streamlines based on criteria '
             'provided, i.e. only those streamlines that fail at least one '
             'criterion will be written to file '
    )
    ends_only = traits.Bool(
        argstr='-ends_only',
        position=1,
        desc='only test the ends of each streamline against the provided '
             'include/exclude ROIs'
    )

    nthreads = traits.Int(
        argstr='-nthreads %d',
        desc='number of threads. if zero, the number of available cpus will '
             'be used',
    )


class TCKEditOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output track file')
    tckweights_out = Directory(desc='path for output text scalar file')


class TCKEdit(CommandLine):
    """
    Perform various editing operations on track files


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> tckedit = mrt.TCKEdit()
    >>> tckedit.inputs.in_file = 'in_tracks.tck'
    >>> tckedit.inputs.out_file = 'out_tracks.tck'
    >>> tckedit.inputs.number = 20000
    >>> tckedit.cmdline                               # doctest: +ELLIPSIS
    'tckedit -number 20000 in_tracks.tck out_tracks.tck'
    >>> tckedit.run()                                 # doctest: +SKIP
    """

    _cmd = 'tckedit'
    input_spec = TCKEditInputSpec
    output_spec = TCKEditOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        # Conditional output
        if isdefined(self.inputs.tckweights_out) and self.inputs.tck_weights_out:
            outputs['tckweights_out'] = op.abspath(self.inputs.tckweight_out)
        return outputs
