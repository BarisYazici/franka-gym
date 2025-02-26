import numpy as np
import time
from scipy.spatial.transform import Rotation

def verify_joint_limits(positions):
    """Verify joint positions are within safe limits"""
    # Franka Research 3 joint limits in radians
    JOINT_LIMITS = [
        [-2.7437, 2.7437],  # Joint 1
        [-1.7837, 1.7837],  # Joint 2
        [-2.9007, 2.9007],  # Joint 3
        [-3.0421, -0.1518], # Joint 4
        [-2.8065, 2.8065],  # Joint 5
        [0.5445, 4.5169],   # Joint 6
        [-3.0159, 3.0159],  # Joint 7
    ]

    for i, (pos, (lower, upper)) in enumerate(zip(positions, JOINT_LIMITS)):
        if not (lower <= pos <= upper):
            raise ValueError(
                f"Joint {i+1} position {pos:.4f} exceeds limits [{lower:.4f}, {upper:.4f}]"
            )

def transform_to_pose(transform):
    """Convert 4x4 homogeneous transformation matrix to [x, y, z, qw, qx, qy, qz] pose.
    The input transform is a 16-element array in column-major format."""
    # Convert flat array to 4x4 matrix (column-major to row-major)
    transform_matrix = np.array(transform).reshape(4, 4, order='F')
    
    # Extract position (translation vector)
    position = transform_matrix[0:3, 3]
    
    # Extract rotation matrix and convert to quaternion
    rotation_matrix = transform_matrix[0:3, 0:3]
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    
    # scipy gives quaternion as [x, y, z, w], we want [w, x, y, z]
    quaternion = np.roll(quaternion, 1)
    
    return np.concatenate([position, quaternion])

def pose_distance(pose1, pose2, position_weight=1.0, orientation_weight=0.5):
    """
    Calculate weighted distance between two poses [x, y, z, qw, qx, qy, qz].
    
    Args:
        pose1: First pose [x, y, z, qw, qx, qy, qz]
        pose2: Second pose [x, y, z, qw, qx, qy, qz]
        position_weight: Weight for position error (default: 1.0)
        orientation_weight: Weight for orientation error (default: 0.5)
        
    Returns:
        Weighted distance scalar
    """
    # Position distance (Euclidean)
    pos_distance = np.linalg.norm(pose1[:3] - pose2[:3])
    
    # Orientation distance (quaternion difference)
    q1 = pose1[3:]  # [qw, qx, qy, qz]
    q2 = pose2[3:]
    
    # Ensure quaternions are normalized
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Calculate angular distance (arccos of dot product, handling sign)
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = min(1.0, max(-1.0, dot_product))  # Numerical stability
    angle_distance = 2 * np.arccos(dot_product)
    
    # Return weighted combination
    return position_weight * pos_distance + orientation_weight * angle_distance
            
def wait_for_motion_completion(rt_control, target_pos, timeout=30.0, threshold=0.01):
    """Wait for motion to complete with timeout"""
    start_time = time.time()
    while True:
        current_pos = rt_control.get_current_position()
        error = np.linalg.norm(np.array(current_pos) - np.array(target_pos))

        if error < threshold:
            return True

        if time.time() - start_time > timeout:
            print("Motion timed out!")
            return False

        time.sleep(0.1) 