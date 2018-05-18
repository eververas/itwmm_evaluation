from pathlib import Path
import subprocess
import numpy as np

import menpo.io as mio
import menpo3d.io as m3io
from menpo.landmark import face_ibug_49_to_face_ibug_49
from menpo.shape import PointCloud
from menpo.base import LazyList
from menpo.landmark import face_ibug_68_to_face_ibug_49
from menpo.transform import AlignmentSimilarity
from menpo3d.vtkutils import trimesh_to_vtk, VTKClosestPointLocator


def kf_sequence_path(i, expression):
    return Path('/vol/atlas/databases/kinectfusionitw/{:02d}/{}'.format(i, expression))


def kf_sequence_image_paths(i, expression):
    return LazyList.init_from_iterable(mio.image_paths(kf_sequence_path(i, expression)))


def subjects_and_expressions_with_registered_gt_mesh():
    dirs_with_registered_meshes = [p.parent for p in Path('/vol/atlas/databases/kinectfusionitw/').glob('**/registered_mesh.pkl')]
    return [(int(p.parent.name), p.name) for p in dirs_with_registered_meshes]


def kf_sequence_image_paths_with_landmarks(i, expression):
    return LazyList.init_from_iterable(
        sorted([p.with_suffix('.png')
                for p in kf_sequence_path(i, expression).glob('*.pts')]))


def load_registered_gt_mesh_from_kf(i, expression, distance=1):
    gt_mesh = mio.import_pickle(kf_sequence_path(i, expression) / 'registered_mesh.pkl')
    return landmark_and_mask_gt_mesh(gt_mesh, distance=distance)


def fit_kf_image(fitter, image):
    initial_shape = image.landmarks['PTS'].lms
    result = fitter.fit_from_shape(image, initial_shape,
                                   camera_update=True, max_iters=[10, 30],
                                   return_costs=True,
                                   init_shape_params_from_lms=False,
                                   focal_length_update=True)
    return result.final_mesh


def landmark_and_mask_gt_mesh(gt_mesh, distance=1):
    gt_mesh.landmarks['ibug49'] = face_ibug_49_to_face_ibug_49(gt_mesh.landmarks['ibug49'])
    gt_mesh.landmarks['nosetip'] = PointCloud(gt_mesh.landmarks['ibug49'].get_label('nose').points[-6][None, :])
    gt_mesh.landmarks['eye_corners'] = PointCloud(gt_mesh.landmarks['ibug49'].points[[36 - 17, 45 - 17]])
    eval_mask = gt_mesh.distance_to(gt_mesh.landmarks['nosetip']).ravel() < distance
    return gt_mesh, eval_mask


def calculate_dense_error(fit_3d_aligned, gt_mesh):
    fit_vtk = trimesh_to_vtk(fit_3d_aligned)
    closest_points_on_fit = VTKClosestPointLocator(fit_vtk)
    nearest_points, tri_indices = closest_points_on_fit(gt_mesh.points)
    err_per_vertex = np.sqrt(np.sum((nearest_points -
                                     gt_mesh.points) ** 2, axis=1))

    # normalize by inter-oc
    b, a = gt_mesh.landmarks['eye_corners'].lms.points
    inter_occ_distance = np.sqrt(((a - b) ** 2).sum())
    print('norm: {}'.format(inter_occ_distance))
    return err_per_vertex / inter_occ_distance


def align_dense_fit_to_gt(fit_3d, gt_mesh):
    return AlignmentSimilarity(fit_3d, gt_mesh).apply(fit_3d)


def mask_align_and_calculate_dense_error(fit_3d, gt_mesh, mask):
    fit_3d_masked = fit_3d.from_mask(mask)
    gt_mesh_masked = gt_mesh.from_mask(mask)

    fit_3d_masked_aligned = align_dense_fit_to_gt(
        fit_3d_masked,
        gt_mesh_masked
    )

    return calculate_dense_error(fit_3d_masked_aligned, gt_mesh_masked), fit_3d_masked_aligned, gt_mesh_masked


def mask_align_and_dense_errors(image, gt_mesh, mask, fitters):
    fits = {k: fitters[k](image) for k in fitters}
    return {k: mask_align_and_calculate_dense_error(fits[k], gt_mesh, mask)
            for k in fitters}


def run_command(command, verbose=False):
    if verbose:
        print('> ' + ' '.join(command))
    try:
        output = subprocess.check_output(command)
        if verbose:
            print(output.decode().strip())
    except subprocess.CalledProcessError as e:
        raise ValueError(e.output.decode().strip())
