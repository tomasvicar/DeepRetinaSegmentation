"""Explicit completion-family contracts; no ML imports or execution permission."""

PROFILES = {
    "pool_6b": (
        "trusted_pool_completion_qa",
        "6b3dfad605a256e904a877efcdcc1f3cc6135dd0",
        "cb5c4f98e0181a13b166c128a632faf502de1a3bf770fbee4a5b7fa649e651cd",
        "9075246883f938ba80b4fe282b365da10ddb8c834bd3106d846465ac79ce9473",
    ),
    "standalone_0b": (
        "standalone_unified_completion_qa",
        "0b8762e667c997a2dfd129dc65988c1f26f6b2ad",
        "3ffe949e787045546a9470750b1aedc27c8ae9bee3603db568f719e4b407f24e",
        "85441afac96f47165df55505d06a08742183b9686fb1c608225166810a23a00f",
    ),
    "standalone_72": (
        "standalone_unified_completion_qa",
        "72fb321187eca517823083aca80cedace9605570",
        "347f78ffd2f4a352c0870faad34327945f7d1c64b865dacb7b7d86e5ca4e0aaa",
        "3725ad32a76449b2a3383485cbf14ef050a75f945b4c25e98f4b5b511a332f28",
    ),
}
PROFILES["pool_72_v3"] = ("v3_rung_evidence", *PROFILES["standalone_72"][1:])
ROLES = {"training_best": "best.pt", "fixed_terminal60": "last.pt"}
VERIFIER = "4b568b7e5e66229e378fd8712000af90a05190364563bad6311687e419eb56fe"
MANIFEST = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
CONTENT = "00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52"
GEOMETRY = "2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d"
