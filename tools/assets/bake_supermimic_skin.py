"""Prepare a SIH2300 material/skin bundle using the independently calibrated skin rig."""
import argparse
from pathlib import Path
from omega_playground.policy.supermimic.skin_rig import SkinRig
from omega_playground.viewer.mjwarp_skin_asset import prepare_skin

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('xml','calibration','scene','skin','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--armed',action='store_true')
    a=p.parse_args()
    prepare_skin(SkinRig(a.xml,a.calibration),a.skin,a.output,scene_path=a.scene,armed=a.armed)
