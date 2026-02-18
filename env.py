import os

CRDS_PATH = "/Users/parke/crds_cache"
if "CRDS_PATH" not in os.environ:
    os.environ["CRDS_PATH"] = CRDS_PATH
    os.environ.setdefault("CRDS_SERVER_URL", "https://hst-crds.stsci.edu")
    os.environ.setdefault("iref", f"{CRDS_PATH}/references/hst/iref/")
    os.environ.setdefault("jref", f"{CRDS_PATH}/references/hst/jref/")
    os.environ.setdefault("oref", f"{CRDS_PATH}/references/hst/oref/")
    os.environ.setdefault("lref", f"{CRDS_PATH}/references/hst/lref/")
    os.environ.setdefault("nref", f"{CRDS_PATH}/references/hst/nref/")