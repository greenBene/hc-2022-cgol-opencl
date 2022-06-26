import pyopencl as cl
import numpy as np
import time


class CGOL():
    kernel = r"""
        int checkField(__global int *cells, int x, int y, int size){
            x = (x + size) % size;
            y = (y + size) % size;

            if (cells[x*size+y] > 0)  return 1;
            else                    return 0;
        }

        __kernel void cgol(__global int *cells_in, __global int *cells_out, int size){
            int x = get_global_id(0);
            int y = get_global_id(1);
            
            int neighbours = 0;
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
            cells_out[x*size+y] = (int) isAlive;
        }
        """

    def __init__(self, size):
        self.cells = np.random.randint(low=2, size=(size,size), dtype=np.int32)
        self.size = size
        # Prep OpelCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)

        self.cells_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.cells.nbytes)
        self.cells_buffer_out = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.cells.nbytes)
        cl.enqueue_copy(self.queue, self.cells_buffer, self.cells).wait()

        self.programm = cl.Program(self.context, self.kernel)

        try:
            self.programm.build()
        except Exception:
            print("Error:")
            print(self.programm.get_build_info(self.context.devices[0], cl.program_build_info.LOG))
            raise
    
    def calculate_next_generation(self):
        self.programm.cgol(
            self.queue, #queue
            (self.size,self.size), #global_size
            None, #local_size
            self.cells_buffer, # Argument 0 (cells_in)
            self.cells_buffer_out, # Argument 1 (cells_out)
            np.int32(self.size)) # Argument 2 (size)
        self.cells_buffer_out, self.cells_buffer = self.cells_buffer, self.cells_buffer_out

    def get_cells(self):
        cl.enqueue_copy(self.queue, self.cells, self.cells_buffer).wait()
        return self.cells

    def print_current_generation(self):
        cells = self.get_cells()
        s = ""
        for row in cells:
            for element in row:
                s+= f"\033[91mX \033[0m" if element == 1 else "\033[92mX \033[0m"
            s+='\n'
        print(s)


if __name__ == '__main__':
    cgol = CGOL(size=10)

    while(True):
        cgol.calculate_next_generation()
        cgol.print_current_generation()
        try:
            input("Press Enter to continue...")
        except EOFError:
            break