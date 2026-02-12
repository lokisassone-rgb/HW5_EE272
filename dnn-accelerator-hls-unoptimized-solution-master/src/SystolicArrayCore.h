#ifndef SYSTOLIC_ARRAY_CORE_H
#define SYSTOLIC_ARRAY_CORE_H

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/comparison/not_equal.hpp>
#include <boost/preprocessor/repetition/for.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/arithmetic/dec.hpp>

#include "ProcessingElement.h"
#include "Fifo.h"

// Define this macro for debug logging
#define HLS_DEBUG 0
#if HLS_DEBUG
#ifndef __SYNTHESIS__
#include <iostream>
#include <fstream>
#include <string>

// Only works for square arrays
template <typename T>

        // Read in the params and loop indices from the channel
        Params params = paramsIn.read();
        LoopIndices loopIndices = loopIndicesIn.read();

        // Flattened loop for IC1, FY, FX, OY0, OX0
        int IC1 = params.IC1;
        int FY  = params.FY;
        int FX  = params.FX;
        int OY0 = params.OY0;
        int OX0 = params.OX0;
        int total_iters = IC1 * FY * FX * OY0 * OX0 + IC0_MAX + OC0_MAX - 1;

        #pragma hls_pipeline_init_interval 1
        for (int flat_idx = 0; flat_idx < total_iters; ++flat_idx) {
            // Calculate each index from the flat index
            int ic1_idx = (flat_idx / (FY * FX * OY0 * OX0)) % IC1;
            int fy_idx  = (flat_idx / (FX * OY0 * OX0)) % FY;
            int fx_idx  = (flat_idx / (OY0 * OX0)) % FX;
            int oy0_idx = (flat_idx / OX0) % OY0;
            int ox0_idx = flat_idx % OX0;
#ifndef __SYNTHESIS__
            if (flat_idx < 10) {
                std::cout << "flat_idx=" << flat_idx
                          << " ic1_idx=" << ic1_idx
                          << " fy_idx=" << fy_idx
                          << " fx_idx=" << fx_idx
                          << " oy0_idx=" << oy0_idx
                          << " ox0_idx=" << ox0_idx << std::endl;
            }
#endif

            // Example: Use these indices for your computation
            // You may need to update the logic below to use these indices

            // Read weights if needed (e.g., at the start of a tile)
            if (flat_idx < IC0) {
                PackedInt<WEIGHT_PRECISION, OC0> w_row = weight.read();
                #pragma hls_unroll yes
                for(int j = 0; j < OC0; j++){
                    weight_reg[flat_idx][j] = w_row.value[j];
                }
#ifndef __SYNTHESIS__
                if (flat_idx < 10) {
                    std::cout << "Weight: ";
                    for (int j = 0; j < OC0; ++j) std::cout << int(weight_reg[flat_idx][j]) << " ";
                    std::cout << std::endl;
                }
#endif
            }

            PackedInt<INPUT_PRECISION, IC0> in_col;
            if (flat_idx < (OX0 * OY0)) {
                in_col = input.read();
#ifndef __SYNTHESIS__
                if (flat_idx < 10) {
                    std::cout << "Input: ";
                    for (int i = 0; i < IC0; ++i) std::cout << int(in_col.value[i]) << " ";
                    std::cout << std::endl;
                }
#endif
            }

            PackedInt<INPUT_PRECISION, IC0> input_buf;
            #define INPUT_FIFO_BODY(z,i,unused) \
                IDTYPE BOOST_PP_CAT(input_fifo_output_, i); \
                IDTYPE BOOST_PP_CAT(input_fifo_input_, i) = in_col.value[i]; \
                BOOST_PP_CAT(input_fifo_, i).run( BOOST_PP_CAT(input_fifo_input_, i) , BOOST_PP_CAT(input_fifo_output_, i) ); \
                input_buf.value[i] = BOOST_PP_CAT(input_fifo_output_, i);
            REPEAT(INPUT_FIFO_BODY)

            #pragma hls_unroll yes
            LABEL(INIT_IN) for(int i = 0; i < IC0; ++i) {
                input_reg[i][0] = input_buf.value[i];
            }

            PackedInt<OUTPUT_PRECISION, OC0> psum_buf;
            if(flat_idx < (OX0 * OY0)){
                if(ic1_idx == 0 && fx_idx == 0 && fy_idx == 0) {
                    #pragma hls_unroll yes
                    for(int j = 0; j < OC0; j++){
                        psum_buf.value[j].template set_val<AC_VAL_0>();
                    }
                }
                else{
                    #pragma hls_unroll yes
                    for(int j = 0; j < OC0; j++){
                        psum_buf.value[j] = accumulation_buffer[flat_idx][j];
                    }
#ifndef __SYNTHESIS__
                    if (flat_idx < 10) {
                        std::cout << "AccumBuf before: ";
                        for (int j = 0; j < OC0; ++j) std::cout << int(accumulation_buffer[flat_idx][j]) << " ";
                        std::cout << std::endl;
                    }
#endif
                }
            }

            PackedInt<OUTPUT_PRECISION, OC0> output_buf;
            #define ACCUM_FIFO_BODY(z,i,unused) \
                ODTYPE BOOST_PP_CAT(psum_fifo_output_, i); \
                ODTYPE BOOST_PP_CAT(psum_fifo_input_, i) = psum_buf.value[i]; \
                BOOST_PP_CAT(psum_fifo_, i).run( BOOST_PP_CAT(psum_fifo_input_, i) , BOOST_PP_CAT(psum_fifo_output_, i) ); \
                output_buf.value[i] = BOOST_PP_CAT(psum_fifo_output_, i);
            REPEAT(ACCUM_FIFO_BODY)

            #pragma hls_unroll yes
            LABEL(INIT_OUT) for(int j = 0; j < OC0; ++j) {
                psum_reg[0][j] = output_buf.value[j];
            }

            #pragma hls_unroll yes
            LABEL(COL) for (int j=0; j < OC0; ++j) {
                #pragma hls_unroll yes
                LABEL(ROW) for (int i=0; i < IC0; ++i) {
                    pe[i][j].run(input_reg[i][j], psum_reg[i][j], weight_reg[i][j], input_reg2[i][j], psum_reg2[i][j]);
                }
            }

            PackedInt<OUTPUT_PRECISION, OC0> output_row;
            #define FIFO_WRITE_BODY_NEW(z,i,unused)\
                ODTYPE BOOST_PP_CAT(accum_fifo_output_, i); \
                BOOST_PP_CAT(accum_fifo_, i).run( psum_reg[IC0][i] , BOOST_PP_CAT(accum_fifo_output_, i) );\
                output_row.value[i] = BOOST_PP_CAT(accum_fifo_output_,i); \
            REPEAT(FIFO_WRITE_BODY_NEW)

            if(flat_idx >= OC0+IC0-1){
                #pragma hls_unroll yes
                for(int i = 0; i < OC0; i++){
                    accumulation_buffer[flat_idx-(IC0+OC0-1)][i] = output_row.value[i];
                }
#ifndef __SYNTHESIS__
                if (flat_idx < 10) {
                    std::cout << "AccumBuf after: ";
                    for (int j = 0; j < OC0; ++j) std::cout << int(accumulation_buffer[flat_idx-(IC0+OC0-1)][j]) << " ";
                    std::cout << std::endl;
                }
#endif
                if (ic1_idx == IC1-1 && fx_idx == FX-1 && fy_idx == FY-1) {   
                    output.write(output_row);
#ifndef __SYNTHESIS__
                    std::cout << "Output written at flat_idx=" << flat_idx << ": ";
                    for (int j = 0; j < OC0; ++j) std::cout << int(output_row.value[j]) << " ";
                    std::cout << std::endl;
#endif
                }
            }

            #pragma hls_unroll yes
            for(int j = 0; j < OC0; j++){
                #pragma hls_unroll yes
                for(int i = 0; i < IC0; i++){
                    input_reg[i][j+1] = input_reg2[i][j];
                    psum_reg[i+1][j] = psum_reg2[i][j];
                }
            }
        }
        // Debug example:
        // printf("outputs written: %d\n", output.size());
    }
                // Your code starts here
                // -------------------------------
                #pragma hls_unroll yes
                LABEL(COL) for (int j=0; j < OC0; ++j) {
                    #pragma hls_unroll yes
                    LABEL(ROW) for (int i=0; i < IC0; ++i) {
                        pe[i][j].run(input_reg[i][j], psum_reg[i][j], weight_reg[i][j], input_reg2[i][j], psum_reg2[i][j]);
                    } //ROW
                } //COL
                // -------------------------------
                // Your code ends here
                // -------------------------------

                // Captures PE register state into log files
                #if HLS_DEBUG
                #ifndef __SYNTHESIS__
                log_matrix(input_file, input_reg, step, OC0);
                log_matrix(weight_file, weight_reg, step, OC0);
                log_matrix(psum_file, psum_reg, step, OC0);
                #endif
                #endif
                

                /*
                 * FIFOs for partial outputs coming out of the systolic array
                 * The skewed version will be in the variable output_row
                 */
                PackedInt<OUTPUT_PRECISION, OC0> output_row;

                #define FIFO_WRITE_BODY_NEW(z,i,unused)\
                    ODTYPE BOOST_PP_CAT(accum_fifo_output_, i); \
                    BOOST_PP_CAT(accum_fifo_, i).run( psum_reg[IC0][i] , BOOST_PP_CAT(accum_fifo_output_, i) );\
                    output_row.value[i] = BOOST_PP_CAT(accum_fifo_output_,i); \
                
                REPEAT(FIFO_WRITE_BODY_NEW)

                // -------------------------------
                // After a certain number of cycles, you will have valid output from the systolic array
                // Depending on the loop indices, this valid output will either be written into the accumulation buffer or written out
                // Your code starts here
                // -------------------------------
                if(step >= OC0+IC0-1){
                    #pragma hls_unroll yes
                    for(int i = 0; i < OC0; i++){
                        accumulation_buffer[step-(IC0+OC0-1)][i] = output_row.value[i];
                    }
                    if (loopIndices.ic1_idx==params.IC1-1 && loopIndices.fx_idx == params.FX-1 && loopIndices.fy_idx == params.FY-1) {   
                        output.write(output_row);
                    }
                }
                // -------------------------------
                // Your code ends here
                // -------------------------------
                
                // -------------------------------
                // Cycle the input/psum registers
                // That is, the outputs that a PE wrote to should now become the input for the next PE
                // Your code starts here
                // -------------------------------
                #pragma hls_unroll yes
                for(int j = 0; j < OC0; j++){
                    #pragma hls_unroll yes
                    for(int i = 0; i < IC0; i++){
                        input_reg[i][j+1] = input_reg2[i][j];
                        psum_reg[i+1][j] = psum_reg2[i][j];
                    }
                }

                // -------------------------------
                // Your code ends here
                // -------------------------------
                if (step == step_bound-1) break;
            }
        }
    
        // Debug example:
        // printf("outputs written: %d\n", output.size());
    }

private:
    
    // -------------------------------
    // Create the following:
    //  - PE array
    //  - accumulation buffer
    //  - weight registers
    //  - input registers (two sets, one at the input of the PE and one at the output) 
    //  - psum registers (two sets, one at the input of the PE and one at the output) 
    // Your code starts here
    // -------------------------------
    ProcessingElement<IDTYPE, WDTYPE, ODTYPE> pe[IC0][OC0];

    ODTYPE accumulation_buffer[ACCUMULATION_BUFFER_SIZE][OC0];
    WDTYPE weight_reg[IC0][OC0];
    IDTYPE input_reg[IC0][OC0+1];
    IDTYPE input_reg2[IC0][OC0];
    ODTYPE psum_reg[IC0+1][OC0];
    ODTYPE psum_reg2[IC0][OC0];
    // -------------------------------
    // Your code ends here
    // -------------------------------
    

#define INPUT_FIFOS_INIT(z, i, unused) \
    Fifo<IDTYPE, i + 1> BOOST_PP_CAT(input_fifo_, i);

    REPEAT(INPUT_FIFOS_INIT)

#define ACCUM_FIFOS_INIT(z, i, unused) \
    Fifo<ODTYPE, i + 1> BOOST_PP_CAT(psum_fifo_, i);

    REPEAT(ACCUM_FIFOS_INIT)
    

#define OUTPUT_FIFOS_INIT(z, i, unused) \
    Fifo<ODTYPE, OC0 - i> BOOST_PP_CAT(accum_fifo_, i);
    
    REPEAT(OUTPUT_FIFOS_INIT)
};

#endif

