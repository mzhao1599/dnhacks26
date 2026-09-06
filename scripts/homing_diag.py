import os, sys, numpy as np, json
os.environ.setdefault("MUJOCO_GL", "egl")
from fleet_memory.envs.libero_env import LiberoEnv
from fleet_memory.execution.homing import Homing, HomingTarget, rotation_error
from fleet_memory.execution.envelope import DEFAULT_ENVELOPE, POS_SCALE_M
np.set_printoptions(precision=4, suppress=True)
env = LiberoEnv("libero_10", 3); obs = env.reset(0)
print("start ee_pos", obs.ee_pos, "quat", obs.ee_quat)
# 1) does a pure +x delta move +x?  2) does a pure +rz rotation change quat the way rotation_error expects?
for name, a in [("+x", [1,0,0,0,0,0,-1]), ("+z", [0,0,1,0,0,0,-1]), ("+rz", [0,0,0,0,0,1,-1]), ("+rx", [0,0,0,1,0,0,-1])]:
    o0 = env.reset(0)
    for _ in range(5): o1, _, _ = env.step(np.array(a, np.float32))
    dp = o1.ee_pos - o0.ee_pos; re = rotation_error(o0.ee_quat, o1.ee_quat)
    print(f"{name}: dpos={dp} rotvec(cur->new)={re}")
# 3) homing from the perturbed-ish start: take 8 random steps away, then home to the start pose
o0 = env.reset(0); tgt = HomingTarget(pos=o0.ee_pos.copy(), quat_xyzw=o0.ee_quat.copy())
rng = np.random.default_rng(0); obs = o0
for _ in range(8): obs, _, _ = env.step(np.r_[rng.uniform(-0.6,0.6,6), -1].astype(np.float32))
print("displaced: pos_err %.4f rot_err %.4f" % (np.linalg.norm(tgt.pos-obs.ee_pos), np.linalg.norm(rotation_error(obs.ee_quat, tgt.quat_xyzw))))
h = Homing(tgt, DEFAULT_ENVELOPE, max_steps=40)
for i in range(40):
    a, pe, re = h.action_for(obs)
    if i % 5 == 0 or (pe < h.tol_pos and re < h.tol_rot): print(f"  step {i}: pos_err {pe:.4f} rot_err {re:.4f} action {a}")
    if pe < h.tol_pos and re < h.tol_rot: print("  REACHED"); break
    obs, _, _ = env.step(a)
print("final pos_err %.4f rot_err %.4f" % (np.linalg.norm(tgt.pos-obs.ee_pos), np.linalg.norm(rotation_error(obs.ee_quat, tgt.quat_xyzw))))
