# Import required libraries 
import pyopencl as cl
import numpy as np

# CELLS
cells = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], 
    np.int32)

# Prepare OpenCL context
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
cells_buffer = cl.Buffer(ctx, mf.READ_WRITE, cells.nbytes)
cells_buffer_out = cl.Buffer(ctx, mf.READ_WRITE, cells.nbytes)
cl.enqueue_copy(queue, cells_buffer, cells).wait()

prg = cl.Program(ctx,
r"""
int checkField(__global uint *cells, int x, int y, int size){
    x = (x + size) % size;
    y = (y + size) % size;

    if (cells[x*size+y] > 0)  return 1;
    else                    return 0;
}

__kernel void cgol(__global uint *cells_in, __global uint *cells_out, int size){
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    uint neighbours = 0;
    int isAlive = cells_in[x*size+y];
    for (int i = -1; i<2; i++){
        for (int j = -1; j<2; j++){
            if(i!=0 || j!=0){
                neighbours += checkField(cells_in, x+i, y+j, size);
            }
        }
    }

    //Any live cell with fewer than two live neighbours dies, as if by underpopulation.
    if(isAlive > 0 && neighbours < 2) isAlive = 0;

    //Any live cell with two or three live neighbours lives on to the next generation.
    else if(isAlive > 0 && neighbours < 4) isAlive = 1;

    //Any live cell with more than three live neighbours dies, as if by overpopulation.
    else if(isAlive > 0 && neighbours > 3) isAlive = 0;

    //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
    else if(isAlive == 0 && neighbours == 3)isAlive = 1;

    cells_out[x*size+y] = (uint) isAlive;
}
""")


try:
    prg.build()
except Exception:
    print("Error:")
    print(prg.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
    raise


for i in range(10_000):
    prg.cgol(queue, (len(cells),len(cells)), None, cells_buffer, cells_buffer_out, np.int32(len(cells)))
    cells_buffer_out, cells_buffer = cells_buffer, cells_buffer_out
cl.enqueue_copy(queue, cells, cells_buffer).wait()
print("Last Generation")
for res in cells:
    print(res)