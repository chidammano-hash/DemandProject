"""Reusable inventory-planning math.

Pure, DB-free formula functions extracted from the inventory CLI scripts so
they can be imported and unit-tested without pulling in a script's argument
parser, database connection, or pipeline plumbing.

Submodules:
    safety_stock — safety-stock, reorder-point, guard-rail, XYZ, outlier,
                   and seasonal-adjustment formulas (IPfeature3 / IPfeature11).
"""
