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
        // Initialize internal buffers to 0
        for (int pix = 0; pix < ACCUMULATION_BUFFER_SIZE; pix++) {
            for (int j = 0; j < OC0; j++) {
                accumulation_buffer[pix][j] = 0;
            }
        }

        for (int i = 0; i < IC0; i++) {
            for (int j = 0; j < OC0+1; j++) input_reg[i][j] = 0;
            for (int j = 0; j < OC0; j++) input_reg2[i][j] = 0;
            for (int j = 0; j < OC0; j++) psum_reg2[i][j] = 0;
            for (int j = 0; j < OC0; j++) weight_reg[i][j] = 0;
        }

        for (int i = 0; i < IC0+1; i++) {
            for (int j = 0; j < OC0; j++) psum_reg[i][j] = 0;
        }
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
           // -------------------------------
        // -------------------------------
        uint_32 tile_size = params.OX0 * params.OY0;
        uint_32 groups = params.IC1 * params.FX * params.FY;
        uint_32 ramp = IC0 + OC0 - 1;
        uint_32 group_span = tile_size + ramp;
        uint_32 total_steps = groups * group_span;

        #pragma hls_pipeline_init_interval 1
        LABEL(INNER_LOOP)
        uint_32 group_idx = 0;
        uint_32 in_group_step = 0;
        uint_16 in_ic1 = 0;
        uint_16 in_fy = 0;
        uint_16 in_fx = 0;
        for (uint_32 step = 0;
            step < IC1_MAX * FX_MAX * FY_MAX * (OX0_MAX * OY0_MAX + IC0_MAX + OC0_MAX - 1);
            ++step)
        {
            // Stop after required cycles
            if (step == total_steps) break;
            bool group_active = in_group_step < tile_size;

            uint_32 input_pix = in_group_step;

            // -------------------------------
            // Load weights once per IC1×FX×FY
            // -------------------------------
            if (group_idx < groups && in_group_step == 0) {
                for (int i = 0; i < IC0_MAX; i++) {
                    if (i < IC0) {
                        PackedInt<WEIGHT_PRECISION, OC0> w_row = weight.read();
                        #pragma hls_unroll yes
                        for (int j = 0; j < OC0_MAX; j++) {
                            weight_reg[i][j] = w_row.value[j];
                            if (j == OC0 - 1) break;
                        }
                    }
                    if (i == IC0 - 1) break;
                }
            }

            // -------------------------------
            // Read input
            // -------------------------------
            PackedInt<INPUT_PRECISION, IC0> in_col;
            if (group_idx < groups && group_active) {
                in_col = input.read();
            } else {
                #pragma hls_unroll yes
                for (int k = 0; k < IC0_MAX; k++) {
                    in_col.value[k].template set_val<AC_VAL_0>();
                    if (k == IC0 - 1) break;
                }
            }

            // -------------------------------
            // Input FIFOs
            // -------------------------------
            PackedInt<INPUT_PRECISION, IC0> input_buf;

            #define INPUT_FIFO_BODY(z,i,unused) \
                IDTYPE BOOST_PP_CAT(input_fifo_output_, i); \
                IDTYPE BOOST_PP_CAT(input_fifo_input_, i) = in_col.value[i]; \
                BOOST_PP_CAT(input_fifo_, i).run( BOOST_PP_CAT(input_fifo_input_, i), BOOST_PP_CAT(input_fifo_output_, i) ); \
                input_buf.value[i] = BOOST_PP_CAT(input_fifo_output_, i);

            REPEAT(INPUT_FIFO_BODY)

            #pragma hls_unroll yes
            for (int i = 0; i < IC0_MAX; i++) {
                input_reg[i][0] = input_buf.value[i];
                if (i == IC0 - 1) break;
            }

            // -------------------------------
            // Partial sum init / read
            // -------------------------------
            PackedInt<OUTPUT_PRECISION, OC0> psum_buf;

            if (group_idx < groups && group_active) {
                if (in_ic1 == 0 && in_fx == 0 && in_fy == 0) {
                    #pragma hls_unroll yes
                    for (int j = 0; j < OC0_MAX; j++) {
                        psum_buf.value[j].template set_val<AC_VAL_0>();
                        if (j == OC0 - 1) break;
                    }
                } else {
                    #pragma hls_unroll yes
                    for (int j = 0; j < OC0_MAX; j++) {
                        psum_buf.value[j] = accumulation_buffer[input_pix][j];
                        if (j == OC0 - 1) break;
                    }
                }
            } else {
                #pragma hls_unroll yes
                for (int j = 0; j < OC0_MAX; j++) {
                    psum_buf.value[j].template set_val<AC_VAL_0>();
                    if (j == OC0 - 1) break;
                }
            }

            // -------------------------------
            // Accum FIFOs
            // -------------------------------
            PackedInt<OUTPUT_PRECISION, OC0> output_buf;

            #define ACCUM_FIFO_BODY(z,i,unused) \
                ODTYPE BOOST_PP_CAT(psum_fifo_output_, i); \
                ODTYPE BOOST_PP_CAT(psum_fifo_input_, i) = psum_buf.value[i]; \
                BOOST_PP_CAT(psum_fifo_, i).run( BOOST_PP_CAT(psum_fifo_input_, i), BOOST_PP_CAT(psum_fifo_output_, i) ); \
                output_buf.value[i] = BOOST_PP_CAT(psum_fifo_output_, i);

            REPEAT(ACCUM_FIFO_BODY)

            #pragma hls_unroll yes
            for (int j = 0; j < OC0_MAX; j++) {
                psum_reg[0][j] = output_buf.value[j];
                if (j == OC0 - 1) break;
            }

            // -------------------------------
            // Run PE array
            // -------------------------------
            #pragma hls_unroll yes
            for (int j = 0; j < OC0_MAX; j++) {
                #pragma hls_unroll yes
                for (int i = 0; i < IC0_MAX; i++) {
                    pe[i][j].run(input_reg[i][j], psum_reg[i][j],
                                weight_reg[i][j],
                                input_reg2[i][j], psum_reg2[i][j]);
                    if (i == IC0 - 1) break;
                }
                if (j == OC0 - 1) break;
            }

            // -------------------------------
            // Output FIFOs
            // -------------------------------
            PackedInt<OUTPUT_PRECISION, OC0> output_row;

            #define FIFO_WRITE_BODY(z,i,unused) \
                ODTYPE BOOST_PP_CAT(accum_fifo_output_, i); \
                BOOST_PP_CAT(accum_fifo_, i).run( psum_reg[IC0][i], BOOST_PP_CAT(accum_fifo_output_, i) ); \
                output_row.value[i] = BOOST_PP_CAT(accum_fifo_output_, i);

            REPEAT(FIFO_WRITE_BODY)

            // -------------------------------
            // Write back / output
            // -------------------------------
            if (group_idx < groups && in_group_step >= ramp && in_group_step < group_span) {
                uint_32 output_pix = in_group_step - ramp;

                #pragma hls_unroll yes
                for (int i = 0; i < OC0_MAX; i++) {
                    accumulation_buffer[output_pix][i] = output_row.value[i];
                    if (i == OC0 - 1) break;
                }

                if (group_idx == groups - 1) {
                    output.write(output_row);
                }
            }

            // -------------------------------
            // Shift registers
            // -------------------------------
            #pragma hls_unroll yes
            for (int j = 0; j < OC0_MAX; j++) {
                #pragma hls_unroll yes
                for (int i = 0; i < IC0_MAX; i++) {
                    input_reg[i][j+1] = input_reg2[i][j];
                    psum_reg[i+1][j]  = psum_reg2[i][j];
                    if (i == IC0 - 1) break;
                }
                if (j == OC0 - 1) break;
            }

            if (in_group_step == group_span - 1) {
                in_group_step = 0;
                group_idx++;

                if (in_fx == params.FX - 1) {
                    in_fx = 0;
                    if (in_fy == params.FY - 1) {
                        in_fy = 0;
                        in_ic1++;
                    } else {
                        in_fy++;
                    }
                } else {
                    in_fx++;
                }
            } else {
                in_group_step++;
            }

            // Input/output indices are derived from step; no manual counter update needed.
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
