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
void log_matrix(std::ofstream& file, T* data, int iteration, int side_length) {
    file << "Iteration: " << iteration << '\n';
    for (int r = 0; r < side_length; r++) {
        for (int c = 0; c < side_length; c++) {
            file << int(data[r][c].to_int()) << ' ';
        }
        file << '\n';
    }
    file << '\n';
}
#endif
#endif


struct LoopIndices{
    uint_16 ic1_idx;
    uint_16 fx_idx;
    uint_16 fy_idx;
};



template <typename IDTYPE, typename WDTYPE, typename ODTYPE, int OC0, int IC0>
class SystolicArrayCore
{
    #if HLS_DEBUG
    #ifndef __SYNTHESIS__
    // Create log file information
    std::ofstream input_file;
    std::ofstream weight_file;
    std::ofstream psum_file;
    #endif
    #endif


public:
    SystolicArrayCore() {
        #if HLS_DEBUG
        #ifndef __SYNTHESIS__

        // Creates filenames
        std::string input_filename = "input_file";
        std::string weight_filename = "weight_file";
        std::string psum_filename = "psum_file";

        // Opens log files when debugging
        input_file.open(input_filename.c_str());
        weight_file.open(weight_filename.c_str());
        psum_file.open(psum_filename.c_str());
        bool open_success = true;
        open_success = open_success && input_file.is_open();
        open_success = open_success && weight_file.is_open();
        open_success = open_success && psum_file.is_open();

        if (!open_success) {
            std::cerr << "Failed to open one or more log files." << std::endl;
        }
        #endif
        #endif
    }

#pragma hls_design interface
    #pragma hls_pipeline_init_interval 1
    void CCS_BLOCK(run)(
        ac_channel<PackedInt<INPUT_PRECISION, IC0> > &input, 
        ac_channel<PackedInt<WEIGHT_PRECISION, OC0> > &weight, 
        ac_channel<PackedInt<OUTPUT_PRECISION, OC0> > &output,
        ac_channel<Params> &paramsIn)
    {
        #ifndef __SYNTHESIS__
        // assert(params.OX0 * params.OY0 <= ACCUMULATION_BUFFER_SIZE);
        #endif

        #ifndef __SYNTHESIS__
        while(paramsIn.available(1))
        #endif
        {
            // -------------------------------
            // Read in the params from the channel
            // -------------------------------
            Params params = paramsIn.read();

            // -------------------------------
            // HW5 Section 5: Flatten all inner loops into one INNER_LOOP
            // Total MAC iterations = IC1 × FX × FY × OX0 × OY0
            // Total steps = mac_iters + ramp + flush
            // -------------------------------
            uint_32 mac_iters = params.IC1 * params.FX * params.FY * params.OX0 * params.OY0;
            uint_32 ramp = IC0 + OC0 - 2;
            uint_32 total_steps = mac_iters + ramp;
            
            #pragma hls_pipeline_init_interval 1
            LABEL(INNER_LOOP) for (uint_32 step = 0; step < IC1_MAX * FX_MAX * FY_MAX * OX0_MAX * OY0_MAX + IC0_MAX + OC0_MAX - 2; ++step) {
                // -------------------------------
                // Decode step into loop indices (ic1, fx, fy, pix)
                // -------------------------------
                uint_32 mac_step =
                    (step < mac_iters)
                        ? (uint_32)step
                        : (uint_32)(mac_iters - 1);

                
                uint_32 pix = mac_step % (params.OX0 * params.OY0);
                uint_32 tmp = mac_step / (params.OX0 * params.OY0);
                
                uint_16 fy = tmp % params.FY;
                tmp /= params.FY;
                
                uint_16 fx = tmp % params.FX;
                tmp /= params.FX;
                
                uint_16 ic1 = tmp;
                
                uint_16 oy0 = pix / params.OX0;
                uint_16 ox0 = pix % params.OX0;

                // -------------------------------
                // Load weights only when starting new pixel (pix == 0)
                // This happens once per IC1×FX×FY combination
                // -------------------------------
                if (pix == 0 && step < mac_iters) {
                    for(int i = 0; i < IC0_MAX; i++){
                        if (i < IC0) {
                            PackedInt<WEIGHT_PRECISION, OC0> w_row = weight.read();
                            #pragma hls_unroll yes
                            for(int j = 0; j < OC0_MAX; j++){
                                weight_reg[i][j] = w_row.value[j];
                                if (j == OC0 - 1) break;
                            }
                        }
                        if (i == IC0 - 1) break;
                    }
                }
                
                PackedInt<INPUT_PRECISION, IC0> in_col;

                // -------------------------------
                // Read inputs every step during active MAC iterations
                // -------------------------------
                if (step < mac_iters) {        
                    in_col = input.read();
                }
                // -------------------------------
                // Your code ends here
                // -------------------------------

                // Debug example:        
                // printf("in_col: %s\n", in_col.to_string().c_str());


                /*
                 * FIFOs for inputs coming in to the systolic array
                 * assign values to in_col, and the skewed version will be in input_buf
                 */
                PackedInt<INPUT_PRECISION, IC0> input_buf;

                #define INPUT_FIFO_BODY(z,i,unused) \
                    IDTYPE BOOST_PP_CAT(input_fifo_output_, i); \
                    IDTYPE BOOST_PP_CAT(input_fifo_input_, i) = in_col.value[i]; \
                    BOOST_PP_CAT(input_fifo_, i).run( BOOST_PP_CAT(input_fifo_input_, i) , BOOST_PP_CAT(input_fifo_output_, i) ); \
                    input_buf.value[i] = BOOST_PP_CAT(input_fifo_output_, i);
                
                REPEAT(INPUT_FIFO_BODY)

                // -------------------------------
                // Assign values from input_buf into the registers for the first column of PEs
                // Your code starts here
                // -------------------------------
                #pragma hls_unroll yes
                LABEL(INIT_IN) for(int i = 0; i < IC0_MAX; ++i) {
                    input_reg[i][0] = input_buf.value[i];
                    if (i == IC0 - 1) {
                        break;
                    }
                }
                // -------------------------------
                // Your code ends here
                // -------------------------------

                PackedInt<OUTPUT_PRECISION, OC0> psum_buf;
                
                // -------------------------------
                // Set partial outputs for the array to psum_buf.
                // Initialize to 0 only at first IC1/FX/FY, otherwise read from accumulation buffer
                // -------------------------------
                if(step < mac_iters){
                    // initial partial output of 0
                    if(ic1 == 0 && fx == 0 && fy == 0) {
                        #pragma hls_unroll yes
                        for(int j = 0; j < OC0_MAX; j++){
                            psum_buf.value[j].template set_val<AC_VAL_0>();
                            if (j == OC0 - 1) {
                                break;
                            }
                        }
                    }
                    else{ // read partial output from accumulation buffer using pixel index
                        #pragma hls_unroll yes
                        for(int j = 0; j < OC0_MAX; j++){
                            psum_buf.value[j] = accumulation_buffer[pix][j];
                            if (j == OC0 - 1) {
                                break;
                            }
                        }
                    }
                }
                // -------------------------------
                // Your code ends here
                // -------------------------------
                
                // Debug example:
                // printf("psum_buf: %s\n", psum_buf.to_string().c_str());

                /*
                 * FIFOs for partial outputs coming in to the systolic array
                 * assign values to psum_buf, and the skewed version will be in output_buf
                 */
                PackedInt<OUTPUT_PRECISION, OC0> output_buf;
                #define ACCUM_FIFO_BODY(z,i,unused) \
                    ODTYPE BOOST_PP_CAT(psum_fifo_output_, i); \
                    ODTYPE BOOST_PP_CAT(psum_fifo_input_, i) = psum_buf.value[i]; \
                    BOOST_PP_CAT(psum_fifo_, i).run( BOOST_PP_CAT(psum_fifo_input_, i) , BOOST_PP_CAT(psum_fifo_output_, i) ); \
                    output_buf.value[i] = BOOST_PP_CAT(psum_fifo_output_, i);
                
                REPEAT(ACCUM_FIFO_BODY)
        
                // -------------------------------
                // Assign values from output_buf into the partial sum registers for the first row of PEs
                // Your code starts here
                // -------------------------------
                #pragma hls_unroll yes
                LABEL(INIT_OUT) for(int j = 0; j < OC0_MAX; ++j) {
                    psum_reg[0][j] = output_buf.value[j];
                    if (j == OC0 - 1) {
                        break;
                    }
                }
                // -------------------------------
                // Your code ends here
                // -------------------------------
            

                // -------------------------------
                // Run the 16x16 PE array
                // Make sure that the correct registers are given to the PE
                // Your code starts here
                // -------------------------------
                #pragma hls_unroll yes
                LABEL(COL) for (int j=0; j < OC0_MAX; ++j) {
                    #pragma hls_unroll yes
                    LABEL(ROW) for (int i=0; i < IC0_MAX; ++i) {
                        pe[i][j].run(input_reg[i][j], psum_reg[i][j], weight_reg[i][j], input_reg2[i][j], psum_reg2[i][j]);
                        if (i == IC0 - 1) {
                            break;
                        }
                    } //ROW
                    if (j == OC0 - 1) {
                        break;
                    }
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
                // After ramp-up, write valid outputs to accumulation buffer
                // Write to output channel only at final IC1×FX×FY
                // -------------------------------
                if(step >= ramp && step < mac_iters + ramp){
                    #pragma hls_unroll yes
                    for(int i = 0; i < OC0_MAX; i++){
                        accumulation_buffer[pix][i] = output_row.value[i];
                        if (i == OC0 - 1) {
                            break;
                        }
                    }
                    // Only write output at the last IC1, FX, FY
                    if (ic1 == params.IC1-1 && fx == params.FX-1 && fy == params.FY-1) {   
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
                for(int j = 0; j < OC0_MAX; j++){
                    #pragma hls_unroll yes
                    for(int i = 0; i < IC0_MAX; i++){
                        input_reg[i][j+1] = input_reg2[i][j];
                        psum_reg[i+1][j] = psum_reg2[i][j];
                        if (i == IC0 - 1) {
                            break;
                        }
                    }
                    if (j == OC0 - 1) {
                        break;
                    }
                }

                // -------------------------------
                // Your code ends here
                // -------------------------------
                if (step == total_steps-1) break;
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
