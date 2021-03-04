from . import frame
from . import chain
from . import transform

JOINT_TYPE_MAP = {'free': 'floating', 
                  'hinge': 'revolute'}
MJ_JOINT_TYPE = {0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}
MJ_GEOM_TYPE = {0: 'plane',
                2: 'sphere',
                3: 'capsule',
                4: 'ellipsoid',
                5: 'cylinder',
                6: 'box'}


def geoms_to_visuals(model, geom, base=transform.Transform()):
    visuals = []
    for g in geom:
        gtype = MJ_GEOM_TYPE[model.geom_type[g]]
        if gtype == 'capsule':
            param = (model.geom_size[g])#, g.fromto)
        elif gtype == 'sphere':
            param = model.geom_size[g]
        else:
            raise ValueError('Invalid geometry type %s.' % gtype)
        quat = model.geom_quat[g]
        pos = model.geom_pos[g]

        visuals.append(frame.Visual(offset=base * transform.Transform(quat, pos),
                                    geom_type=gtype,
                                    geom_param=param))
    return visuals


def body_to_link(model, body_id, base=transform.Transform()):
    return frame.Link(str(model.body_id2name(body_id)),
                      offset=base * transform.Transform(model.body_quat[body_id], model.body_pos[body_id]))

def joint_to_joint(model, joint, base=transform.Transform()):
    joint_type = MJ_JOINT_TYPE[model.jnt_type[joint]]
    n = model.joint_id2name(joint)
    return frame.Joint(model.joint_id2name(joint),
                       offset=base * transform.Transform(pos=model.jnt_pos[joint]),
                       joint_type=JOINT_TYPE_MAP[joint_type],
                       axis=model.jnt_axis[joint])

def add_composite_joint(model, root_frame, joints, base=transform.Transform()):
    if len(joints) > 0:
        root_frame.children = root_frame.children + [frame.Frame(link=frame.Link(name=root_frame.link.name + '_child'),
                                                                 joint=joint_to_joint(model, joints[0], base))]
        ret, offset = add_composite_joint(model, root_frame.children[-1], joints[1:])
        return ret, root_frame.joint.offset * offset
    else:
        return root_frame, root_frame.joint.offset

def _get_body_joints(model, body_id):
    n = model.body_jntnum[body_id]
    if n > 0:
        start = model.body_jntadr[body_id]
        return list(range(start, start+n))
    return []

def _get_body_geoms(model, body_id):
    n = model.body_geomnum[body_id]
    if n > 0:
        start = model.body_geomadr[body_id]
        return list(range(start, start+n))
    return []


def _get_body_bodies(model, body_id):
    children = [[] for _ in range(model.nbody)]
    for i, p in enumerate(model.body_parentid):
        children[p].append(i)
    return children[body_id]
    
def _build_chain_recurse(model, root_frame, root_body):
    base = root_frame.link.offset

    joints = _get_body_joints(model, root_body)
    cur_frame, cur_base = add_composite_joint(model, root_frame, joints, base)
    jbase = cur_base.inverse() * base

    geoms = _get_body_geoms(model, root_body)
    if len(joints) > 0:
        cur_frame.link.visuals = geoms_to_visuals(model, geoms, jbase)
    else:
        cur_frame.link.visuals = geoms_to_visuals(model, geoms)
    
    for b in _get_body_bodies(model, root_body):
        cur_frame.children = cur_frame.children + [frame.Frame()]
        next_frame = cur_frame.children[-1]
        next_frame.name = str(model.body_id2name(b)) + "_frame"
        next_frame.link = body_to_link(model, b, jbase)
        _build_chain_recurse(model, next_frame, b)


def build_chain_from_mujoco_py(model):
    """
    Build a Chain object from MJCF data.

    Parameters
    ----------
    data : str
        MJCF string data.

    Returns
    -------
    chain.Chain
        Chain object created from MJCF.
    """
    root_body_id = 1
    root_frame = frame.Frame(str(model.body_id2name(root_body_id)) + "_frame",
                             link=body_to_link(model, root_body_id),
                             joint=frame.Joint())
    _build_chain_recurse(model, root_frame, root_body_id)
    return chain.Chain(root_frame)


def build_serial_chain_from_mjcf(data, end_link_name, root_link_name=""):
    """
    Build a SerialChain object from MJCF data.

    Parameters
    ----------
    data : str
        MJCF string data.
    end_link_name : str
        The name of the link that is the end effector.
    root_link_name : str, optional
        The name of the root link.

    Returns
    -------
    chain.SerialChain
        SerialChain object created from MJCF.
    """
    mjcf_chain = build_chain_from_mjcf(data)
    return chain.SerialChain(mjcf_chain, end_link_name + "_frame",
                             "" if root_link_name == "" else root_link_name + "_frame")
