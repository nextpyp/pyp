import functools
import math
import sys

import numpy as np

from pyp.analysis.geometry import transformations as vtk
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def calcSpikeNormXYZ(spike_x, spike_y, spike_z, virion_x, virion_y, virion_z):
    """
    This function calculates Euler rotations - normX, normY, normZ
    that will transform spike's vector vertically.

    spike_x,y,z - coordinate of spike 
    virion_x,y,z - coordinate of virion
    """
    vector = np.array([spike_x - virion_x, spike_y - virion_y, spike_z - virion_z])
    norm = np.linalg.norm(vector)
    vector = np.array([vector[0] / norm, vector[1] / norm, vector[2] / norm])
    vertical = np.array([0, 0, 1])

    a, b = vector, vertical
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    if r[2, 2] < 1 - np.nextafter(0, 1):

        if r[2, 2] > -1 + np.nextafter(0, 1):
            x = math.acos(r[2, 2])
            z = math.atan2(r[2, 0] / math.sin(x), r[2, 1] / -math.sin(x))
        else:
            x = math.pi
            z = math.atan2(r[0, 1], r[0, 0])
    else:
        x = 0
        z = math.atan2(r[0, 1], r[0, 0])

    normX = np.degrees(x)
    normZ = np.degrees(z)

    return normX, 0, normZ


def eulerZXZtoZYZ(matrixZXZ):
    """This function converts rotation matrix from ZXZ to ZYZ.
    Converting matrix from ZXZ to ZYZ is not straightforward, and
    it can only be done within three axis-rotation matrices.
    ZXZ has to be converted to ZYZ before going into more than three matrix multiplcation (e.g. spa_euler_angles)
    
    *** ZXZ convention is used by 3DAVG (NormXYZ), EMAN2 (az/alt/phi) etc.
    *** ZYZ convention is used by Frealign, RELION etc.
    """

    # decompose matrix into triple Euler angles - z1 * x * z2
    # Matrix(ZXZ) -> R(z1) * R(x) * R(z2)
    # *** LEFT hand
    #            cz1*cz2 - sz1*cx*sz2    cz1*sz2 + sz1*cx*cz2        sinZ1*sinX
    # Matrix = [ -sz1*cz2 - cz1*cx*sz2   -sz1*sz2 + cz1*cx*cz2       cosZ1*sinX ]
    #                 sinX*sinZ2             -sinX*cosZ2                 cosX

    if matrixZXZ[2, 2] < 1 - np.nextafter(0, 1):

        if matrixZXZ[2, 2] > -1 + np.nextafter(0, 1):
            x = math.acos(matrixZXZ[2, 2])
            z1 = math.atan2(
                matrixZXZ[0, 2] / math.sin(x), matrixZXZ[1, 2] / math.sin(x)
            )
            z2 = math.atan2(
                matrixZXZ[2, 0] / math.sin(x), -matrixZXZ[2, 1] / math.sin(x)
            )
        else:  # x == pi
            x = math.pi
            z2 = math.atan2(matrixZXZ[0, 1], matrixZXZ[0, 0])
            z1 = 0
    else:  # x == 0
        x = 0
        z1 = 0
        z2 = math.atan2(matrixZXZ[0, 1], matrixZXZ[0, 0])

    [z1_degree, x_degree, z2_degree] = list(map(math.degrees, [z1, x, z2]))
    # print( z1_degree, x_degree, z2_degree )

    # compute again matrix but in right-handedness
    z1 = vtk.rotation_matrix(np.radians(z1_degree), [0, 0, 1])
    x = vtk.rotation_matrix(np.radians(x_degree), [1, 0, 0])
    z2 = vtk.rotation_matrix(np.radians(z2_degree), [0, 0, 1])
    m = np.dot(z1, np.dot(x, z2))
    # print(m)

    # decompose matrix into triple Euler angles - z1, y, z2
    # Matrix(ZYZ) -> R(z1) * R(y) * R(z2)
    # *** RIGHT hand
    #            cy*cz1*cz2 - sz1*sz2     -cz2*sz1 - cycz1*sz2   sycz1
    # Matrix = [ cycz2*sz1 + cz1*sz2      cz1*cz2 - cysz1*sz2    sysz1  ]
    #                  -sy*cz2                  sy*sz2            cy
    if m[2, 2] < 1 - np.nextafter(0, 1):

        if m[2, 2] > -1 + np.nextafter(0, 1):
            y = math.acos(m[2, 2])
            z2 = math.atan2(m[2, 1] / math.sin(y), -m[2, 0] / math.sin(y))
            z1 = math.atan2(m[1, 2] / math.sin(y), m[0, 2] / math.sin(y))
        else:
            y = math.pi
            z1 = -math.atan2(m[1, 0], m[1, 1])
            z2 = 0
    else:
        y = 0
        z1 = math.atan2(m[1, 0], m[1, 1])
        z2 = 0

    [z1_degree, y_degree, z2_degree] = list(map(math.degrees, [z1, y, z2]))
    # print( z1_degree, y_degree, z2_degree )

    # compose matrix ZYZ in left handedness
    z1 = vtk.rotation_matrix(np.radians(-z1_degree), [0, 0, 1])
    y = vtk.rotation_matrix(np.radians(-y_degree), [0, 1, 0])
    z2 = vtk.rotation_matrix(np.radians(-z2_degree), [0, 0, 1])

    m = np.dot(z1, np.dot(y, z2))
    return m


def eulerZYZtoZXZ(matrixZYZ):
    """This function converts rotation matrix from ZYZ to ZXZ.
    """

    # Left handedness ZYZ
    m = matrixZYZ
    
    # decompose matrix into triple Euler angles - z1 * x * z2
    # Matrix(ZXZ) -> R(z1) * R(x) * R(z2)
    # *** RIGHT hand
    #            cz1*cz2 - sz1*cx*sz2    -cz1*sz2 - sz1*cx*cz2        sinZ1*sinX
    # Matrix = [ sz1*cz2 + cz1*cx*sz2   -sz1*sz2 + cz1*cx*cz2       -cosZ1*sinX ]
    #                 sinX*sinZ2             sinX*cosZ2                 cosX

    if m[2, 2] < 1 - np.nextafter(0, 1):

        if m[2, 2] > -1 + np.nextafter(0, 1):
            x = math.acos(m[2, 2])
            z1 = math.atan2(
                m[0, 2] / math.sin(x), -m[1, 2] / math.sin(x)
            )
            z2 = math.atan2(
                m[2, 0] / math.sin(x), m[2, 1] / math.sin(x)
            )
        else:  # x == pi
            x = math.pi
            z2 = math.atan2(-m[0, 1], m[0, 0])
            z1 = 0
    else:  # x == 0
        x = 0
        z1 = 0
        z2 = math.atan2(-m[0, 1], m[0, 0])

    [z1_degree, x_degree, z2_degree] = list(map(math.degrees, [z1, x, z2]))
    # print( z1_degree, x_degree, z2_degree )

    # compute again zxz matrix in left-handedness
    z1 = vtk.rotation_matrix(np.radians(-z1_degree), [0, 0, 1])
    x = vtk.rotation_matrix(np.radians(-x_degree), [1, 0, 0])
    z2 = vtk.rotation_matrix(np.radians(-z2_degree), [0, 0, 1])
    m = np.dot(z1, np.dot(x, z2))

    return m



def eulerTwoZYZtoOneZYZ(matrixZYZZYZ):

    # Matrix(ZYZ) -> R(z1) * R(y) * R(z2)
    # *** LEFT handedness
    #            cz1*cy*cz2 - sz1*sz2       cz1*cy*sz2 + sz1*cz2         -cz1*sy
    # Matrix = [ -sz1*cy*cz2 - cz1*sz2      -sz1*cy*sz2 + cz1*cz2         sz1*sy ]
    #                 sy*cz2                       sy*sz2                   cy

    if matrixZYZZYZ[2, 2] < 1 - np.nextafter(0, 1):
        if matrixZYZZYZ[2, 2] > -1 + np.nextafter(0, 1):
            y = math.acos(matrixZYZZYZ[2, 2])
            z2 = math.atan2(
                matrixZYZZYZ[2, 1] / math.sin(y), matrixZYZZYZ[2, 0] / math.sin(y)
            )
            z1 = math.atan2(
                matrixZYZZYZ[1, 2] / math.sin(y), -matrixZYZZYZ[0, 2] / math.sin(y)
            )
        else:
            y = math.pi
            z1 = math.atan2(-matrixZYZZYZ[0, 1], -matrixZYZZYZ[0, 0])
            z2 = 0

    else:
        y = 0
        z1 = math.atan2(matrixZYZZYZ[0, 1], matrixZYZZYZ[0, 0])
        z2 = 0

    [z1_degree, y_degree, z2_degree] = list(map(math.degrees, [z1, y, z2]))
    z1 = vtk.rotation_matrix(np.radians(-z1_degree), [0, 0, 1])
    y = vtk.rotation_matrix(np.radians(-y_degree), [0, 1, 0])
    z2 = vtk.rotation_matrix(np.radians(-z2_degree), [0, 0, 1])
    # print z1_degree, y_degree, z2_degree

    m = np.dot(z1, np.dot(y, z2))
    return m


def get_degrees_from_matrix(matrix: np.ndarray):

    if matrix[2, 2] < 1 - np.nextafter(0, 1):
        if matrix[2, 2] > -1 + np.nextafter(0, 1):
            theta = math.acos(matrix[2, 2])
            psi = math.atan2(matrix[2, 1] / math.sin(theta), matrix[2, 0] / math.sin(theta))
            phi = math.atan2(matrix[1, 2] / math.sin(theta), -matrix[0, 2] / math.sin(theta))
        else:
            theta = math.pi
            phi = math.atan2(-matrix[0, 1], -matrix[0, 0])
            psi = 0
    else:
        theta = 0
        phi = math.atan2(matrix[0, 1], matrix[0, 0])
        psi = 0

    frealign = np.degrees(np.array([psi, theta, phi]))

    # frealign does not use negative angles, so we add 360 to each negative angle
    frealign = np.where(frealign < 0, frealign + 360.0, frealign)

    psi, theta, phi = frealign[0], frealign[1], frealign[2]

    return psi, theta, phi



def spa_euler_angles(tilt_angle, tilt_axis_angle, normal, m, cutOffset):

    """
    This function converts the 3D geometries solved by sub-tomogram averaging (either EMAN2 or 3DAVG) to 
    2D corresponding geometries used by Frealign, for further SPA refinement. 


    The geometries include:
    1. One tilt axis rotation angle (solved by IMOD)
    2. One tilt angle (solved by IMOD)
    3. Three Euler angles (normX, normY, normZ used by 3DAVG) 
    4. Three Euler angles (three rotations)
    5. Three shifts (x, y, z)

    All we know about these rotations are "extrinsic", which means "world axes", as opposed to local axes that are changed during Euler rotation, are used. 
    To compute rotation matrices, we should "post-multiply" these rotaion matrices in sequence. 
    
    2D particle projection undergoes these rotations in order, before becoming the final structure of sub-tomogram averaging.
    ** 2D projection -> R( tilt_axis_rotation ) -> R( tilt_angle ) -> R( normXYZ ) -> R( refinement_rotations ) -> T( translations ) -> final structure
    
    1. Tilt axis rotation is rotating around world z axis
    2. Tilt angle is rotating around world y axis
    3. Norm X, Y, Z are rotating around world X, Z2, Z1 respectively, in which normZ->normX->normY. 
    4. Rotation can be decomposed into three Euler angles az, alt, phi that are rotating around Z1, X, Z2 respectively. az->alt->phi.
    5. Shift is intuitive and it is the last one. 


    However, different programs use different conventions. Here we only introduce LEFT and RIGHT handedness for rotation. 
    
    * Left handedness is used by 3DAVG, EMAN2, Frealign, including normXYZ, normXYZ and Euler angles of rotation.
    Its rotation matrix, if rotating around z axis, should be 
                   cos(angle)    sin(angle)  0
    R( angle ) = [ -sin(angle)    cos(angle)  0 ] 
                        0           0        1
    
    * Right handedness is used by IMOD and vtk module, including tilt axis rotation angle and tilt angle.
    Its rotation matrix, if rotating around z axis, should be 
                   cos(angle)    -sin(angle)  0
    R( angle ) = [ sin(angle)    cos(angle)  0 ] 
                        0           0        1
    
    Shifts are really intuitive (positive value means moving up), except the ones used by 3DAVG (observed in IMOD)
    ---- 3DAVG ------
    +x: move object left (negative in x axis)
    +y: move object up (positive in y axis)
    +z: move object down (negative in z axis)


    All the transformations you saw above are applied to "sub-volumes", but the three Euler angles and two shifts we want to derive
    for Frealign refinement are applied to "reference" instead. 
    
    Therefore, to calculate geometries for reference, we need to think in this way:
    ** Final result -> T( -translations ) -> R( -refinement_rotations ) -> R( -normXYZ ) -> R( -tilt_angle ) -> R( -tilt_axis_rotation ) -> 2D projection

    Rotation matrix will be:
    R( -phi )*R( -alt )*R( -az )*R( -normY )*R( -normX )*R( -normZ )*R( tilt_angle )*R( -tilt_axis_rotation )
    
    We can treat shifts as vector, which are rotated in sequence. It can be calculated as:
    R( -tilt_axis_rotation )*R( tilt_angle )*R( -normZ )*R( -normX )*R( -normY )*R( -az )*R( -alt )*R( -phi )*T( -translation )


    One last thing to be careful, these multiple rotation matrices CANNOT be simplified into three Euler angles 
    if simply decoding cos() and sin() in a multiplied matrix.
    
    This way of decoding Euler angles only works if having less than or equal to three axis rotaion, like 
    Z1 * Y * Z2 = Z1' * Y' * Z2'

    It is tricky if not in the same Euler rotation, like
    Z1 * X * Z2 = Z1' * Y * Z2'
    
    To solve this, we have to address the triples step by step:
    1. Convert ZXZ to ZYZ
       (Z*X*Z) * (Z*X*Z) * Y * Z =>
    2. Merge two ZYZ into one
       (Z*Y*Z) * (Z*Y*Z) * Y * Z =>
    3. It can now be solved by decoding sin and cos in matrix
       (Z*Y*Z) * Y * Z
    
    These conversions are using eulerZXZtoZYZ() and eulerTwoZYZtoOneZYZ()


    Now we can start calculating PSI, THETA, PHI, shift_x, shift_y for Frealign
    but be careful of the order, sign and handedness, especially in rotation.
    """

    # Refinement matrix, including rotation and translation, which is from 3DAVG or EMAN2
    refinement = np.matrix(
        [
            [m[0], m[1], m[2], m[3]],
            [m[4], m[5], m[6], m[7]],
            [m[8], m[9], m[10], m[11]],
            [m[12], m[13], m[14], m[15]],
        ]
    )
    # Rotation matrix only ( Z1 * X * Z2 )
    refinement_rotation = np.matrix(
        [
            [m[0], m[1], m[2], 0],
            [m[4], m[5], m[6], 0],
            [m[8], m[9], m[10], 0],
            [m[12], m[13], m[14], m[15]],
        ]
    )

    # invert the order of rotation matrix ( Z2 * X * Z1 )
    refinement_rotation_reverse = np.matrix(
        [
            [m[0], -m[4], m[8], 0],
            [-m[1], m[5], -m[9], 0],
            [m[2], -m[6], m[10], 0],
            [m[12], m[13], m[14], m[15]],
        ]
    )

    # translation matrix only
    refinement_translation = np.dot(np.linalg.inv(refinement_rotation), refinement)

    # correct translations x and z to make sense
    refinement_translation[0, 3] = -refinement_translation[0, 3]
    refinement_translation[2, 3] = -refinement_translation[2, 3]

    # compute rotation matrix of norm X Y Z, be careful that vtk uses right handedness but 3DAVG uses left handedness
    normX = vtk.rotation_matrix(np.radians(-normal[0]), [1, 0, 0])
    normY = vtk.rotation_matrix(np.radians(-normal[1]), [0, 0, 1])
    normZ = vtk.rotation_matrix(np.radians(-normal[2]), [0, 0, 1])
    norm = np.dot(normZ, np.dot(normX, normY))
    norm_reverse = np.dot(normY, np.dot(normX, normZ))

    # compute matrix of Tilt axis rotation (around Z)
    tilt_axis_rotation_matrix = vtk.rotation_matrix(
        np.radians(tilt_axis_angle), [0, 0, 1]
    )
    # compute matrix of tilt angle (around Y)
    tilt_angle_matrix = vtk.rotation_matrix(np.radians(tilt_angle), [0, 1, 0])

    ###################
    # Rotation matrix #
    ###################
    # R( -phi )*R( -alt )*R( -az )*R( -normY )*R( -normX )*R( -normZ )*R( tilt_angle )*R( -tilt_axis_rotation )
    #      Z         X        Z          Z           X           Z             Y                   Z
    r = np.linalg.inv(tilt_axis_rotation_matrix)
    r = np.dot(tilt_angle_matrix, r)

    # convert ZXZ to ZYZ and then merge two ZYZs into one ZYZ
    n = eulerZXZtoZYZ(np.linalg.inv(norm))
    n = np.dot(eulerZXZtoZYZ(np.linalg.inv(refinement_rotation)), n)
    # n = np.dot(np.linalg.inv(refinement_rotation), n)
    n = eulerTwoZYZtoOneZYZ(n)

    # now we combine all the rotations into particle rotation (in parfile)
    ppsi, ptheta, pphi = get_degrees_from_matrix(n)

    r = np.dot(n, r)

    ######################
    # Translation matrix #
    ######################
    # R( -tilt_axis_rotation )*R( tilt_angle )*R( -normZ )*R( -normX )*R( -normY )*R( -az )*R( -alt )*R( -phi )*T( -translation )

    t = vtk.translation_matrix([0, 0, -cutOffset])
    t = np.dot(np.linalg.inv(refinement_translation), t)
    t = np.dot(np.linalg.inv(refinement_rotation_reverse), t)
    t = np.dot(np.linalg.inv(norm_reverse), t)

    # now we store particle shifts directly into parfile without norm, matrix
    px = -t[0, 3]
    py = -t[1, 3]
    pz = -t[2, 3]

    t = np.dot(tilt_angle_matrix, t)
    t = np.dot(np.linalg.inv(tilt_axis_rotation_matrix), t)

    # extract euler angles from transformation matrix by decoding its sin and cos

    # Frealign first rotates the "reference" by PHI -> THETA -> PSI extrinsically
    # then moves/translates the reference to match 2D particle projections

    # Matrix(ZYZ) -> R(z1) * R(y) * R(z2)
    # z1 = phi
    # y = theta
    # z2 = psi

    # *** LEFT handedness
    #            cz1*cy*cz2 - sz1*sz2       cz1*cy*sz2 + sz1*cz2         -cz1*sy
    # Matrix = [ -sz1*cy*cz2 - cz1*sz2      -sz1*cy*sz2 + cz1*cz2         sz1*sy ]
    #                 sy*cz2                       sy*sz2                   cy

    if r[2, 2] < 1 - np.nextafter(0, 1):
        if r[2, 2] > -1 + np.nextafter(0, 1):
            theta = math.acos(r[2, 2])
            psi = math.atan2(r[2, 1] / math.sin(theta), r[2, 0] / math.sin(theta))
            phi = math.atan2(r[1, 2] / math.sin(theta), -r[0, 2] / math.sin(theta))
        else:
            theta = math.pi
            phi = math.atan2(-r[0, 1], -r[0, 0])
            psi = 0
    else:
        theta = 0
        phi = math.atan2(r[0, 1], r[0, 0])
        psi = 0

    psi, theta, phi = get_degrees_from_matrix(r)

    return [psi, theta, phi, t[0, 3], t[1, 3]], [-ppsi, -ptheta, -pphi, px, py, pz]


def getShiftsForRecenter(normal, m, cutOffset):

    # Refinement matrix, including rotation and translation, which is from 3DAVG or EMAN2
    refinement = np.matrix(
        [
            [m[0], m[1], m[2], m[3]],
            [m[4], m[5], m[6], m[7]],
            [m[8], m[9], m[10], m[11]],
            [m[12], m[13], m[14], m[15]],
        ]
    )
    # Rotation matrix only ( Z1 * X * Z2 )
    refinement_rotation = np.matrix(
        [
            [m[0], m[1], m[2], 0],
            [m[4], m[5], m[6], 0],
            [m[8], m[9], m[10], 0],
            [m[12], m[13], m[14], m[15]],
        ]
    )

    # invert the order of rotation matrix ( Z2 * X * Z1 )
    refinement_rotation_reverse = np.matrix(
        [
            [m[0], -m[4], m[8], 0],
            [-m[1], m[5], -m[9], 0],
            [m[2], -m[6], m[10], 0],
            [m[12], m[13], m[14], m[15]],
        ]
    )

    # translation matrix only
    refinement_translation = np.dot(np.linalg.inv(refinement_rotation), refinement)

    # correct translations x and z to make sense
    refinement_translation[0, 3] = -refinement_translation[0, 3]
    refinement_translation[2, 3] = -refinement_translation[2, 3]

    # compute rotation matrix of norm X Y Z, be careful that vtk uses right handednes but 3DAVG uses left handedness
    normX = vtk.rotation_matrix(np.radians(-normal[0]), [1, 0, 0])
    normY = vtk.rotation_matrix(np.radians(-normal[1]), [0, 0, 1])
    normZ = vtk.rotation_matrix(np.radians(-normal[2]), [0, 0, 1])
    norm = np.dot(normZ, np.dot(normX, normY))
    norm_reverse = np.dot(normY, np.dot(normX, normZ))

    ######################
    # Translation matrix #
    ######################
    # R( -tilt_axis_rotation )*R( tilt_angle )*R( -normZ )*R( -normX )*R( -normY )*R( -az )*R( -alt )*R( -phi )*T( -translation )

    t = vtk.translation_matrix([0, 0, -cutOffset])
    t = np.dot(np.linalg.inv(refinement_translation), t)
    t = np.dot(np.linalg.inv(refinement_rotation_reverse), t)
    t = np.dot(np.linalg.inv(norm_reverse), t)

    return [t[0, 3], t[1, 3], t[2, 3]]

def findSpecimenBounds(particle_coordinates, dim_tomogram):
    """ Find the boundaries in x, y, z of 'specimen', which will be used to divide a tomogram into several grids for frame refinement
        (z is determined by the particle coordinates)

    Parameters
    ----------
    particle_coordinates : list[str]
        Particle 3D coordinates from 3dboxes file 
    dim_tomogram : list[float]
        Dimension of tomogram - x, y, z

    Returns
    -------
    list[list[int]]
        List that stores two 3D cooridnates; one is bottom left corner, one is top right corner of tomogram 
    """

    min_x = dim_tomogram[0]
    min_y = dim_tomogram[1]
    min_z = dim_tomogram[2]

    max_x, max_y, max_z = 0, 0, 0

    # particle coordinates - x, y, z are in index 1, 2, 3 respectively
    for coord in particle_coordinates:
        x, y, z = float(coord[1]), float(coord[2]), float(coord[3])

        if x < min_x:
            min_x = math.floor(x)
        elif x > max_x:
            max_x = math.ceil(x)

        if y < min_y:
            min_y = math.floor(y)
        elif y > max_y:
            max_y = math.ceil(y)

        if z < min_z:
            min_z = math.floor(z)
        elif z > max_z:
            max_z = math.ceil(z)

    # use original size for x and y
    min_x = min_y = 0
    max_x = dim_tomogram[0]
    max_y = dim_tomogram[1]

    bottom_left_corner = [min_x, min_y, min_z]
    top_right_corner = [max_x, max_y, max_z]

    return [bottom_left_corner, top_right_corner]


def divide2regions(
    bottom_left_corner, top_right_corner, split_x=4, split_y=4, split_z=1, overlap=0.0
):
    """Divide the tomogram into several grids to sort local particles for frame refinement

    Parameters
    ----------
    bottom_left_corner : list[float]
        The 3D coordinate of one corner of tomogram
    top_right_corner : list[float]
        The 3D coordinate of the other corner of tomogram
    split_x : int, optional
        The number of grids to be divided in x direction, by default 4
    split_y : int, optional
        The number of grids to be divided in y direction, by default 4
    split_z : int, optional
        The number of grids to be divided in z direction, by default 1
    overlap : float, optional
        The percentage of overlapped area between grids, 0.0 - 1.0, by default 0.2

    Returns
    -------
    list[list[float]], list[float]]
        The first list stores 3D coordinate of the bottom-left corners for every divided square
        The second list stores the size of squares [ x, y, z ]
    """

    if split_x < 1 or split_y < 1 or split_z < 1:
        logger.error(f"Split x/y/z has to be greater than zero.")
        sys.exit()

    corners_squares = []

    tomosize_x = top_right_corner[0] - bottom_left_corner[0]
    tomosize_y = top_right_corner[1] - bottom_left_corner[1]
    tomosize_z = top_right_corner[2] - bottom_left_corner[2]

    if split_x - split_x * overlap + overlap != 0:
        patchsize_x = tomosize_x / (split_x - split_x * overlap + overlap)
    else:
        patchsize_x = tomosize_x

    if split_y - split_y * overlap + overlap != 0:
        patchsize_y = tomosize_y / (split_y - split_y * overlap + overlap)
    else:
        patchsize_y = tomosize_y

    if split_z - split_z * overlap + overlap != 0:
        patchsize_z = tomosize_z / (split_z - split_z * overlap + overlap)
    else:
        patchsize_z = tomosize_z

    curr_x, curr_y, curr_z = bottom_left_corner
    increment_x, increment_y, increment_z = (
        patchsize_x * (1 - overlap),
        patchsize_y * (1 - overlap),
        patchsize_z * (1 - overlap),
    )

    for i in range(split_x):
        for j in range(split_y):
            for k in range(split_z):
                corners_squares.append(
                    [
                        curr_x + (increment_x * i),
                        curr_y + (increment_y * j),
                        curr_z + (increment_z * k),
                    ]
                )

    logger.info(
        f"Dividing into {len(corners_squares)} region(s) for CSPT region-based refinement"
    )

    return corners_squares, [patchsize_x, patchsize_y, patchsize_z]


def findSpecimenBounds(particle_parameters, dim_tomogram):
    """ Find the boundaries in x, y, z of 'specimen', which will be used to divide a tomogram into several grids for frame refinement
        (z is determined by the particle coordinates)

    Parameters
    ----------
    particle_coordinates : list[str]
        Particle 3D coordinates from 3dboxes file 
    dim_tomogram : list[float]
        Dimension of tomogram - x, y, z

    Returns
    -------
    list[list[int]]
        List that stores two 3D cooridnates; one is bottom left corner, one is top right corner of tomogram 
    """

    min_x = dim_tomogram[0]
    min_y = dim_tomogram[1]
    min_z = dim_tomogram[2]

    max_x, max_y, max_z = 0, 0, 0

    # particle coordinates - x, y, z are in index 1, 2, 3 respectively
    for particle_index in particle_parameters:
        particle = particle_parameters[particle_index]
        x, y, z = particle.x_position_3d, particle.y_position_3d, particle.z_position_3d

        if x < min_x:
            min_x = math.floor(x)
        elif x > max_x:
            max_x = math.ceil(x)

        if y < min_y:
            min_y = math.floor(y)
        elif y > max_y:
            max_y = math.ceil(y)

        if z < min_z:
            min_z = math.floor(z)
        elif z > max_z:
            max_z = math.ceil(z)

    # use original size for x and y
    min_x = min_y = 0
    max_x = dim_tomogram[0]
    max_y = dim_tomogram[1]

    bottom_left_corner = [min_x, min_y, min_z]
    top_right_corner = [max_x, max_y, max_z]

    return [bottom_left_corner, top_right_corner]


def DefocusOffsetFromCenter(
    particleCoord, tomoCenter, tilt_angle, T_inverse, specimen_z_offset, handedness=-1
):

    particle_pos3D = np.matrix(
        [
            [1, 0, 0, particleCoord[0]],
            [0, 1, 0, particleCoord[1]],
            [0, 0, 1, particleCoord[2] - specimen_z_offset],
            [0, 0, 0, 1],
        ]
    )

    center_pos3D = np.matrix(
        [
            [1, 0, 0, tomoCenter[0]],
            [0, 1, 0, tomoCenter[1]],
            [0, 0, 1, tomoCenter[2]],
            [0, 0, 0, 1],
        ]
    )

    z_offset = np.matrix(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, specimen_z_offset], [0, 0, 0, 1]]
    )

    toImodOrigin = np.matrix(
        [
            [1, 0, 0, -tomoCenter[0]],
            [0, 1, 0, -tomoCenter[1]],
            [0, 0, 1, -tomoCenter[2]],
            [0, 0, 0, 1],
        ]
    )

    tiltAngle = vtk.rotation_matrix(np.radians(tilt_angle), [0, 1, 0])

    axisAngleAndShift = np.matrix(
        [
            [T_inverse[0], T_inverse[1], 0, T_inverse[4]],
            [T_inverse[2], T_inverse[3], 0, T_inverse[5]],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    toRawImage = np.matrix(
        [
            [1, 0, 0, tomoCenter[0]],
            [0, 1, 0, tomoCenter[1]],
            [0, 0, 1, tomoCenter[2]],
            [0, 0, 0, 1],
        ]
    )

    particle_raw = functools.reduce(
        np.matmul,
        [
            toRawImage,
            axisAngleAndShift,
            tiltAngle,
            toImodOrigin,
            z_offset,
            particle_pos3D,
        ],
    )
    center_raw = functools.reduce(
        np.matmul,
        [
            toRawImage,
            axisAngleAndShift,
            tiltAngle,
            toImodOrigin,
            z_offset,
            center_pos3D,
        ],
    )

    defocus_offset = particle_raw[2, 3] - center_raw[2, 3]

    if handedness == -1:
        defocus_offset *= -1
    elif handedness == 1:
        defocus_offset = defocus_offset
    else:
        raise Exception("Don't understand handedness other than 1 or -1.")

    return defocus_offset



def getRelionMatrix(tilt_angle, xf, thickness, origin_image_dim, tomox, tomoy):
    """ Combine per-tilt tilt-series alignment and other transformations into single matrix that will be used by RELION

    Args:
        tilt_angle (int): Tilt angle in degree
        xf (numpy array): Tilted image affine transform in 2D matrix 
        thickness (int): Thickness (Z) of unbinned tomogram 
        origin_image_dim (int): raw image dimension [x, y]
        tomox, tomoy (int): Dimension of unbinned tomogram

    Returns:
        numpy array: per-tilt 2D matrix further used by RELION
    """
    # affineXforms * tiltProjs * toImodOrigin3D * Off * YzFlip
    tilt = math.radians(tilt_angle)
    original_center_x = (origin_image_dim[0] - 1.0) / 2.0
    original_center_y = (origin_image_dim[1] - 1.0) / 2.0
    ali_center_x = (tomox - 1.0) / 2.0
    ali_center_y = (tomoy - 1.0) / 2.0

    yzflip = np.matrix(
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, thickness-1],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ]
            )

    # we normally don't shift tomogram in x and z in PYP
    off = np.matrix(
                [
                    [1, 0, 0, 0], 
                    [0, 1, 0, 0], 
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )

    to_imod_orgin = np.matrix(
                [
                    [1, 0, 0, -1],
                    [0, 1, 0, -thickness/2.0],
                    [0, 0, 1, -1],
                    [0, 0, 0, 1],
                ]
            )

    tilt_matrix = np.matrix(
                [
                    [math.cos(tilt), -math.sin(tilt), 0, ali_center_x],
                    [0, 0, 1, ali_center_y],
                    [-math.sin(tilt), -math.cos(tilt), 0, 0],
                    [0, 0, 0, 1],
                ]
            )

    to_origin = np.matrix(
                [
                    [1, 0, 0, -ali_center_x],
                    [0, 1, 0, 0],
                    [0, 0, 1, -ali_center_y],
                    [0, 0, 0, 1],
                ]
            )

    tilt_proj = np.matmul(tilt_matrix, to_origin)

    p = np.matrix(
                [
                    [1, 0, 0, original_center_x],
                    [0, 1, 0, original_center_y],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )

    xf_matrix = np.matrix(
                [
                    [xf[0], xf[1], 0, xf[4]],
                    [xf[2], xf[3], 0, xf[5]],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
    xf_invert = np.linalg.inv(xf_matrix)

    q = np.matrix(
                [
                    [1, 0, 0, -ali_center_x],
                    [0, 1, 0, -ali_center_y],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
    affine = functools.reduce(np.matmul, [p, xf_invert, q])

    ret = functools.reduce(
            np.matmul,
            [
                affine,
                tilt_proj,
                to_imod_orgin,
                off, 
                yzflip
            ],
        )


    return ret

def spk2Relion(x, y, z, binning, tomox, tomoy, thickness, tomo_x_bin=512, tomo_y_bin=512, tomo_z_bin=256, shiftz=0):
    """ Convert particle coordinate from PYP to RELION

    Args:
        x (int): PYP particle x position (bottom left is origin)
        y (int): PYP particle y position (bottom left is origin)
        z (int): PYP particle z position (bottom left is origin)
        binning (int): Binning factor applied to our binned tomogram 
        tomox, tomoy, thickness (int): Dimension of full size tomogram 
        tomo_x_bin (int, optional): X of PYP binned tomogram. Defaults to 512.
        tomo_z_bin (int, optional): Z of PYP binned tomogram. Defaults to 256.
        shiftz (int, optional): Shifts applied to tomogram. Defaults to 0.

    Returns:
        int: Particle X Y Z in RELION
    """
    
    relion_z_bin = tomo_z_bin / 2 - z  
    relion_y_bin = y - (tomo_y_bin / 2)
    relion_x_bin = x - (tomo_x_bin / 2)

    relion_x = relion_x_bin * binning + (tomox / 2.0)
    relion_y = relion_y_bin * binning + (tomoy / 2.0)
    relion_z = relion_z_bin * binning + (thickness / 2.0 - shiftz)

    return int(relion_x), int(relion_y), int(relion_z)

def relion2Spk(x, y, z, binning, relion_x, relion_y, relion_z, tomo_x_bin=512, tomo_z_bin=256, shiftx=0, shiftz=0):
    """ Convert particle coordinate from RELION to PYP

    Args:
        x (int): RELION particle x position (center is origin)
        y (int): RELION particle y position (center is origin)
        z (int): RELION particle z position (center is origin)
        binning (int): Binning factor applied to PYP binned tomogram 
        relion_x (int): X of RELION tomogram 
        relion_y (int): Y of RELION tomogram 
        relion_z (int): Z of RELION tomogram 
        tomo_x_bin (int, optional): X of PYP tomogram. Defaults to 512.
        tomo_z_bin (int, optional): Z of PYP tomogram. Defaults to 256.
        shiftx (int, optional): X shift applied to RELION tomogram. Defaults to 0.
        shiftz (int, optional): Z shift applied to RELION tomogram. Defaults to 0.

    Returns:
        int: Particle X Y Z in PYP
    """
    ox = float(x - (relion_x / 2.0) + shiftx) / binning
    oy = float(y - (relion_y / 2.0)) / binning
    oz = float(z - (relion_z / 2.0) + shiftz) / binning
    
    new_x = (ox + tomo_x_bin/2)
    new_y = (oy + tomo_x_bin/2)
    new_z = (tomo_z_bin - (oz + tomo_z_bin/2))
    
    return new_x, new_y, new_z

def cistem2_alignment2Relion(ppsi, ptheta, pphi, px, py, pz):

    # input: per particle rotation and shifts from extended Particles object
    # rotation 
    # R( normZ )*R( normX )*R( normY )*R( az )*R( alt )*R( phi )*R( -ppsi )*R( -ptheta )*R( -pphi ) (all left handedness)
    #      Z          X          Z         Z       X        Z          Z           Y           Z 

    mpsi = vtk.rotation_matrix(np.radians(-ppsi), [0, 0, 1])
    mtheta = vtk.rotation_matrix(np.radians(-ptheta), [0, 1, 0])
    mphi = vtk.rotation_matrix(np.radians(-pphi), [0, 0, 1])
    refine_rotation = functools.reduce(np.matmul, [mpsi, mtheta, mphi])
    
    m = eulerTwoZYZtoOneZYZ(refine_rotation)

    if m[2, 2] < 1 - np.nextafter(0, 1):

        if m[2, 2] > -1 + np.nextafter(0, 1):
            y = math.acos(m[2, 2])
            z2 = math.atan2(m[2, 1] / math.sin(y), -m[2, 0] / math.sin(y))
            z1 = math.atan2(m[1, 2] / math.sin(y), m[0, 2] / math.sin(y))
        else:
            y = math.pi
            z1 = -math.atan2(m[1, 0], m[1, 1])
            z2 = 0
    else:
        y = 0
        z1 = math.atan2(m[1, 0], m[1, 1])
        z2 = 0
  
    rot, tilt, psi = np.degrees(np.array([z2, y, z1])) 
    
    return rot, tilt, psi, -px, -py, -pz


def alignment2Relion(matrix, ppsi, ptheta, pphi, normX, normY, normZ):
    """ Convert TOMO particle alignment from PYP to RELION
        PYP contains:
            - one matrix that includes rotation and translation from SVA
            - PPSI, PTHETA, PPHI that are refined particle rotations by CSP
            - norm X/Y/Z 
        RELION: 
            - rot, tilt, psi which are particle rotation 
            - dx, dy, dz which are particle translation 
    Args:
        matrix (numpy array): Matrix
        ppsi (int): Euler angle in degree
        ptheta (int): Euler angle in degree
        pphi (int): Euler angle in degree
        normX (int): Euler angle in degree
        normY (int): Euler angle in degree
        normZ (int): Euler angle in degree

    Returns:
        int: Three rotation angles and three translations
    """
    refinement_sva = np.matrix([
        [matrix[0], matrix[1], matrix[2], matrix[3]], 
        [matrix[4], matrix[5], matrix[6], matrix[7]],
        [matrix[8], matrix[9], matrix[10], matrix[11]],
        [0,0,0,1]
    ])

    reverse_rotation = np.matrix([
        [matrix[0], -matrix[4], matrix[8], 0], 
        [-matrix[1], matrix[5], -matrix[9], 0],
        [matrix[2], -matrix[6], matrix[10], 0],
        [0,0,0,1]
    ])

    rotation = np.matrix([
        [matrix[0], matrix[1], matrix[2], 0], 
        [matrix[4], matrix[5], matrix[6], 0],
        [matrix[8], matrix[9], matrix[10], 0],
        [0,0,0,1]
    ])

    # rotation 
    # R( normZ )*R( normX )*R( normY )*R( az )*R( alt )*R( phi )*R( -ppsi )*R( -ptheta )*R( -pphi ) (all left handedness)
    #      Z          X          Z         Z       X        Z          Z           Y           Z 
    mnormY = vtk.rotation_matrix(np.radians(-normY), [0, 0, 1])
    mnormX = vtk.rotation_matrix(np.radians(-normX), [1, 0, 0])
    mnormZ = vtk.rotation_matrix(np.radians(-normZ), [0, 0, 1])
    norm = functools.reduce(np.matmul, [mnormZ, mnormX, mnormY])

    m = np.matmul(eulerZXZtoZYZ(norm), eulerZXZtoZYZ(rotation))
    m = eulerTwoZYZtoOneZYZ(m)

    mpsi = vtk.rotation_matrix(np.radians(-ppsi), [0, 0, 1])
    mtheta = vtk.rotation_matrix(np.radians(-ptheta), [0, 1, 0])
    mphi = vtk.rotation_matrix(np.radians(-pphi), [0, 0, 1])
    refine_rotation = functools.reduce(np.matmul, [mpsi, mtheta, mphi])

    m = np.matmul(m, refine_rotation)
    m = eulerTwoZYZtoOneZYZ(m)

    if m[2, 2] < 1 - np.nextafter(0, 1):

        if m[2, 2] > -1 + np.nextafter(0, 1):
            y = math.acos(m[2, 2])
            z2 = math.atan2(m[2, 1] / math.sin(y), -m[2, 0] / math.sin(y))
            z1 = math.atan2(m[1, 2] / math.sin(y), m[0, 2] / math.sin(y))
        else:
            y = math.pi
            z1 = -math.atan2(m[1, 0], m[1, 1])
            z2 = 0
    else:
        y = 0
        z1 = math.atan2(m[1, 0], m[1, 1])
        z2 = 0
  
    rot, tilt, psi = np.degrees(np.array([z2, y, z1])) 

    # translation 
    translation = np.dot(np.linalg.inv(rotation), refinement_sva)
    translation[0, 3] = -translation[0, 3]
    translation[2, 3] = -translation[2, 3]

    t = np.matmul(np.linalg.inv(reverse_rotation), translation)

    norm = functools.reduce(np.matmul, [mnormY, mnormX, mnormZ])
    t = np.matmul(np.linalg.inv(norm), t)
    
    # return 0, 0, 0, 0, 0, 0
    return rot, tilt, psi, t[0,3], t[1,3], t[2,3]


def get_tomo_binning(image_x, image_y, binned_tomo_dims=512, squared_image=True):
    if squared_image:
        return int(math.ceil(max(image_x, image_y) / binned_tomo_dims*1.0))
    else:
        return int(math.ceil(image_x / binned_tomo_dims * 1.0))

def get_vir_binning_boxsize(vir_rad, pixel_size, factor_rad2boxsize=3, binned_vir_boxsize=400):
    virion_boxsize = factor_rad2boxsize * int(vir_rad / pixel_size)
    return int(math.ceil( virion_boxsize / binned_vir_boxsize)), virion_boxsize

