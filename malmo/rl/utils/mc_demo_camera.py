"""Put the live Minecraft window in demo view: F5 third person, F1 hide GUI/chat."""

from __future__ import print_function

import os
import subprocess

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "send_mc_keys")


def apply_demo_camera(press_f5=True, press_f1=True):
    if not os.path.isfile(_BIN):
        print("mc_demo_camera: missing", _BIN)
        return False
    args = [_BIN]
    if press_f5:
        args.append("F5")
    if press_f1:
        args.append("F1")
    rc = subprocess.call(args)
    return rc == 0
