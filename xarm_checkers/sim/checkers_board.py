from typing import List
import os
import numpy as np
from PIL import Image
import pybullet as pb

def checkerboard_array(n_grids: int,
                       px_per_grid: int=10,
                       colors: List=((0,0,0),(255,0,0)),
                      ) -> np.ndarray:
    # you need to repeat the grids to make rendering look better
    cb = np.indices((n_grids, n_grids)).sum(axis=0) % 2
    cb = np.repeat(cb, px_per_grid, axis=0)
    cb = np.repeat(cb, px_per_grid, axis=1)

    array = np.zeros((n_grids*px_per_grid, n_grids*px_per_grid, 3),
                      dtype=np.uint8)
    array[cb == 0] = colors[0]
    array[cb == 1] = colors[1]
    return array

def add_checkerboard(
    grid_size: float=0.1, # CENTIMETERS!
    n_grids: int=8,
    px_per_grid: int=10,
    grid_colors: List=((0,0,0),(255,0,0)),
) -> int:
    # create body
    coll_id = pb.createCollisionShape(
        pb.GEOM_BOX,
        halfExtents=(grid_size/2, grid_size/2, 0.01)
    )
    vis_id = pb.createVisualShape(pb.GEOM_BOX,
                                  halfExtents=(grid_size/2, grid_size/2, 0.01))
    obj_id = pb.createMultiBody(0, coll_id, vis_id)

    # TODO add collision object

    # place pattern in lower right corner of image, this seems to work
    # for adding textures to the top of any GEOM_BOX
    pattern = checkerboard_array(n_grids, px_per_grid, grid_colors)
    padding = 3*pattern.shape[0]
    pattern = np.pad(pattern, ((padding,0),(padding,0),(0,0)))

    # save pattern as png so it can be imported by pybullet
    tex_fname = 'tmp_board_texture.png'
    Image.fromarray(pattern, mode="RGB").save(tex_fname)
    tex_id = pb.loadTexture(tex_fname)
    pb.changeVisualShape(obj_id, -1, textureUniqueId=tex_id,
                         rgbaColor=(1,1,1,1))

    os.remove(tex_fname)

    return obj_id
