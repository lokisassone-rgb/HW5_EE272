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
            for (int j = 0; j < OC0; j++) {
                weight_reg[0][i][j] = 0;
                weight_reg[1][i][j] = 0;
            }
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
        uint_32 total_mac_ops = groups * tile_size;  // Total MAC operations
        // Pipeline depth: input FIFO (1) + IC0 PE rows + OC0 accum FIFO = IC0 + OC0 + 1
        // But we index from 0, so ramp_cycles = IC0 + OC0
        uint_32 ramp_cycles = IC0 + OC0;
        uint_32 total_cycles = total_mac_ops + ramp_cycles;

        uint_16 active_bank = 0;
        uint_32 global_idx = 0;  // NEW: tracks which MAC operation we're on (0 to total_mac_ops-1)

        if (groups > 0) {
            for (int i = 0; i < IC0_MAX; i++) {
                if (i < IC0) {
                    PackedInt<WEIGHT_PRECISION, OC0> w_row = weight.read();
                    #pragma hls_unroll yes
                    for (int j = 0; j < OC0_MAX; j++) {
                        weight_reg[0][i][j] = w_row.value[j];
                        if (j == OC0 - 1) break;
                    }
                }
                if (i == IC0 - 1) break;
            }
        }

        #pragma hls_pipeline_init_interval 1
        LABEL(INNER_LOOP)
        // REMOVE: input_group, input_pix, output_group, output_pix counters from here
        for (uint_32 step = 0; step < IC1_MAX * FX_MAX * FY_MAX * OX0_MAX * OY0_MAX + IC0_MAX + OC0_MAX; ++step)
        {
            if (step == total_cycles) break;  // Use total_cycles instead
            bool mac_active = (step < total_mac_ops);  // Use total_mac_ops
            
            // NEW: Decode global_idx into logical indices (inside the loop)
            uint_32 current_pix = global_idx % tile_size;
            uint_32 current_group = global_idx / tile_size;
            
            if (mac_active && (current_group + 1 < groups)) {
                uint_32 pixels_left_in_group = tile_size - current_pix;
                if (pixels_left_in_group <= IC0) {
                    uint_32 weight_row_idx = IC0 - pixels_left_in_group;
                    PackedInt<WEIGHT_PRECISION, OC0> w_row = weight.read();
                    #pragma hls_unroll yes
                    for (int j = 0; j < OC0_MAX; j++) {
                        weight_reg[1 - active_bank][weight_row_idx][j] = w_row.value[j];
                        if (j == OC0 - 1) break;
                    }
                }
            }

            // -------------------------------
            // Read input
            // -------------------------------
            PackedInt<INPUT_PRECISION, IC0> in_col;
            if (mac_active) {
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

            if (mac_active) {
                if (current_group == 0) {
                    #pragma hls_unroll yes
                    for (int j = 0; j < OC0_MAX; j++) {
                        psum_buf.value[j].template set_val<AC_VAL_0>();
                        if (j == OC0 - 1) break;
                    }
                } else {
                    #pragma hls_unroll yes
                    for (int j = 0; j < OC0_MAX; j++) {
                        psum_buf.value[j] = accumulation_buffer[current_pix][j];
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
                    uint_16 bank_sel = active_bank;
                    if (current_group > 0 && current_pix < (uint_32)i) {
                        bank_sel = 1 - active_bank;
                    }
                    pe[i][j].run(input_reg[i][j], psum_reg[i][j],
                                weight_reg[bank_sel][i][j],
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
            uint_32 output_idx = step - ramp_cycles;
            if (step >= ramp_cycles && step < total_mac_ops + ramp_cycles) {
                uint_32 output_pix = output_idx % tile_size;
                uint_32 output_group = output_idx / tile_size;
                #pragma hls_unroll yes
                for (int i = 0; i < OC0_MAX; i++) {
                    accumulation_buffer[output_pix][i] = output_row.value[i];
                    if (i == OC0 - 1) break;
                }
                if (output_group == groups - 1) {
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

            if (mac_active) {
                global_idx++;  
                if (current_pix == tile_size - 1 && current_group + 1 < groups) {
                    active_bank = 1 - active_bank;
                }
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
    WDTYPE weight_reg[2][IC0][OC0];
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