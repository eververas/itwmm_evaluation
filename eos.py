from functools import lru_cache
from pathlib import Path
from tempfile import mkdtemp

import menpo.io as mio
import menpo3d.io as m3io

from menpo.landmark import face_ibug_49_to_face_ibug_49
from menpo.shape import PointCloud, TexturedTriMesh

import kinectfusion as kf
from kinectfusion import run_command
from menpo.shape import ColouredTriMesh

THIS_DIR = Path(__file__).parent


@lru_cache()
def load_eos_low_res_lm_index():
    return mio.import_pickle(
        THIS_DIR / 'eos_landmark_settings.pkl')['ibug_49_index']


@lru_cache()
def load_basel_kf_trilist():
    return mio.import_pickle(kf.kf_sequence_path(1, 'neutral') /
                             'registered_mesh.pkl').trilist


@lru_cache()
def load_fw_on_eos_low_res_settings():
    d = mio.import_pickle(THIS_DIR / 'fw_on_eos_low_res_settings.pkl')
    bc_fw_on_eos_low_res = d['bc_fw_on_eos_low_res']
    tri_index_fw_on_eos_low_res = d['tri_index_fw_on_eos_low_res']
    return bc_fw_on_eos_low_res, tri_index_fw_on_eos_low_res


def upsample_eos_low_res_to_fw(eos_mesh_low_res):
    bc_fw_on_eos_low_res, tri_index_fw_on_eos_low_res = load_fw_on_eos_low_res_settings()
    effective_fw_pc = eos_mesh_low_res.project_barycentric_coordinates(
        bc_fw_on_eos_low_res, tri_index_fw_on_eos_low_res)
    tcoords = eos_mesh_low_res.barycentric_coordinate_interpolation(eos_mesh_low_res.tcoords.points,
                                                                    bc_fw_on_eos_low_res,
                                                                    tri_index_fw_on_eos_low_res)

    effective_fw = TexturedTriMesh(effective_fw_pc.points, tcoords,
                                   eos_mesh_low_res.texture, trilist=load_basel_kf_trilist())
    return effective_fw


def upsample_eos_low_res_to_fw_no_texture(eos_mesh_low_res):
    bc_fw_on_eos_low_res, tri_index_fw_on_eos_low_res = load_fw_on_eos_low_res_settings()
    effective_fw_pc = eos_mesh_low_res.project_barycentric_coordinates(
        bc_fw_on_eos_low_res, tri_index_fw_on_eos_low_res)

    effective_fw = ColouredTriMesh(effective_fw_pc.points,
                                   trilist=load_basel_kf_trilist())
    return effective_fw


def eos_command(image_path, output_dir, landmarks_path=None):
    if landmarks_path is None:
        landmarks_path = Path(image_path).with_suffix('.pts')
    image_stem = Path(image_path).stem

    eos_prefix = '/vol/atlas/homes/jab08/itw3dmm_3rd_party/eos/install/'
    bin_ = eos_prefix + 'bin/'
    share = eos_prefix + 'share/'

    return [
        bin_ + 'fit-model',
        '--model',
        share + 'sfm_shape_3448.bin',
        '--mapping',
        share + 'ibug2did.txt',
        '--image',
        str(image_path),
        '--landmarks',
        str(landmarks_path),
        '--output',
        str(Path(output_dir) / image_stem)
    ]


def load_eos_output(image_path, output_dir):
    image_stem = Path(image_path).stem
    mesh = m3io.import_mesh(Path(output_dir) / (image_stem + '.obj'))

    lms = face_ibug_49_to_face_ibug_49(PointCloud(
        mesh.points[load_eos_low_res_lm_index()]))
    mesh_upsampled = upsample_eos_low_res_to_fw(mesh)
    mesh_upsampled.landmarks['ibug49'] = lms
    return {
        'raw_fit': mesh,
        'corresponded_fit': mesh_upsampled
    }


def eos_fit_and_output(image_path, output_dir,
                       landmarks_path=None, verbose=False):
    return run_command(eos_command(image_path, output_dir,
                                   landmarks_path=landmarks_path),
                       verbose=verbose)


def fit_eos(image, verbose=False):
    temp_dir = mkdtemp()
    # save out the input landmarks to the temp dir
    lms_path = Path(temp_dir) / 'landmarks.pts'
    mio.export_landmark_file(image.landmarks['PTS'], lms_path)
    image_path = image.path
    eos_fit_and_output(image_path, temp_dir, landmarks_path=lms_path,
                       verbose=verbose)
    return load_eos_output(image_path, temp_dir)
